"""Measure exact text overlap across completed web profiles and an optional baseline.

This computes candidate volume before semantic filtering and near-deduplication.
Baseline hashes are computed from text; baseline token counts remain reported
stored counts, so this is not the final release integrity audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from scripts.youtube.download import _directory_lock, _write_json
from .profile import _query_rows, load_inputs, sha256
from .sources import SOURCES, data_files


def profile_paths(root: Path, profile_name: str, source: str) -> list[str]:
    manifest, digest = load_inputs(root, source)
    output = root / profile_name
    report = json.loads((output / "summary.json").read_text())
    if (report.get("complete") is not True or report.get("manifest_sha256") != digest
            or report.get("run_config_sha256") != sha256(output / "run-config.json")):
        raise ValueError(f"Profile is incomplete or provenance changed: {output}")
    entries = data_files(manifest)
    shards = report["shards"]
    if len(shards) != len(entries):
        raise ValueError("Profile does not cover every shard")
    paths = []
    for index, (entry, shard) in enumerate(zip(entries, shards)):
        path = output / "metadata" / f"{index:05d}.parquet"
        if (shard["source_shard"] != entry["path"]
                or shard["metadata_sha256"] != sha256(path)):
            raise ValueError(f"Profile metadata changed or reordered: {path}")
        paths.append(str(path))
    return paths


def compare(paths: list[str], baseline: list[str], output: Path, workers: int) -> dict:
    with duckdb.connect() as connection:
        connection.execute("SET memory_limit = '4GB'")
        connection.execute(f"SET threads = {workers}")
        connection.execute("SET temp_directory = ?", [str(output.parent / "compare-temp")])
        connection.read_parquet(paths, hive_partitioning=False).create_view("candidates")
        connection.execute("""
            CREATE TEMP TABLE candidate_unique AS
            SELECT content_hash, max(content_length) tokens, count(*) copies,
                   count(DISTINCT source) sources
            FROM candidates WHERE content_hash IS NOT NULL GROUP BY content_hash
        """)
        report = _query_rows(connection, """
            SELECT count(*) exact_unique_texts, coalesce(sum(tokens),0) exact_unique_tokens,
                   coalesce(sum(copies-1),0) extra_exact_copies,
                   count(*) FILTER (WHERE sources>1) cross_source_exact_texts,
                   coalesce(sum(tokens) FILTER (WHERE sources>1),0) cross_source_exact_tokens
            FROM candidate_unique
        """)[0]
        report["per_source"] = _query_rows(connection, """
            SELECT source, count(*) exact_unique_texts, sum(tokens) exact_unique_tokens
            FROM (SELECT source, content_hash, max(content_length) tokens
                  FROM candidates WHERE content_hash IS NOT NULL GROUP BY source, content_hash)
            GROUP BY source ORDER BY source
        """)
        report["baseline_comparison"] = None
        if baseline:
            connection.read_parquet(baseline, union_by_name=True, hive_partitioning=False).create_view("baseline")
            invalid = connection.execute("""
                SELECT count(*) FROM baseline WHERE content IS NULL OR trim(content)=''
                    OR try_cast(content_length AS BIGINT) IS NULL OR content_length <= 0
            """).fetchone()[0]
            if invalid:
                raise ValueError(f"Baseline has {invalid} invalid text/token records")
            totals = _query_rows(connection, """
                SELECT count(*) records, sum(content_length) stored_tokens FROM baseline
            """)[0]
            connection.execute("CREATE TEMP TABLE baseline_hashes AS SELECT DISTINCT sha256(content) content_hash FROM baseline")
            overlap = _query_rows(connection, """
                SELECT count(*) FILTER (WHERE b.content_hash IS NOT NULL) matching_texts,
                       coalesce(sum(c.tokens) FILTER (WHERE b.content_hash IS NOT NULL),0) matching_tokens,
                       coalesce(sum(c.tokens) FILTER (WHERE b.content_hash IS NULL),0) novel_exact_unique_tokens
                FROM candidate_unique c LEFT JOIN baseline_hashes b USING (content_hash)
            """)[0]
            report["baseline_comparison"] = {**overlap, **totals, "baseline_paths": baseline,
                "baseline_tokens_recomputed": False, "aspirational_target_tokens": 3_000_000_000,
                "target_is_release_gate": False,
                "gap_to_aspirational_target": max(0, 3_000_000_000 - totals["stored_tokens"])}
    report.update(complete=True, tokenizer="cl100k_base",
                  scope="Raw candidate exact-text overlap only. No quality filtering or near-deduplication. "
                        "Novel exact-unique tokens are an upper bound on retained additions. "
                        "Per-source counts overlap and must not be added together.")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--profile-name", default="profile-v1")
    parser.add_argument("--source-profile", action="append", default=[], metavar="SOURCE=NAME",
                        help="Override one source's profile directory, preserving completed profiles from other versions")
    parser.add_argument("--baseline", type=Path, nargs="+", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    profiles = dict.fromkeys(SOURCES, args.profile_name)
    overrides = set()
    for setting in args.source_profile:
        source, separator, name = setting.partition("=")
        if not separator or source not in SOURCES or source in overrides:
            parser.error("source-profile must be SOURCE=NAME for a known source, once per source")
        profiles[source] = name
        overrides.add(source)
    for name in profiles.values():
        if not name or Path(name).name != name or name in {".", "..", "raw"}:
            parser.error("Profile names must be single directory names other than raw")
    root = args.data_dir.expanduser().resolve()
    output = (args.output or root / "comparison-v1.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    baseline = set()
    for candidate in args.baseline:
        found = ([candidate] if candidate.is_file() and candidate.suffix == ".parquet" else
                 [p for p in candidate.rglob("*.parquet") if not p.name.startswith("._")]
                 if candidate.is_dir() else [])
        if not found:
            parser.error(f"No baseline Parquet files: {candidate}")
        baseline.update(str(p.resolve()) for p in found)
    # Hold both source locks through validation and query so profiles cannot be
    # replaced while comparing them. No Slurm tasks should use the same roots.
    from contextlib import ExitStack
    with ExitStack() as stack:
        for source in SOURCES:
            stack.enter_context(_directory_lock(root / source))
        paths = [p for source in SOURCES for p in profile_paths(root / source, profiles[source], source)]
        if str(output) in paths or str(output) in baseline:
            parser.error("Output cannot overwrite input")
        _write_json(output, {"complete": False, "status": "running"})
        report = compare(paths, sorted(baseline), output, args.workers)
        report["profile_reports"] = {
            source: sha256(root / source / profiles[source] / "summary.json") for source in SOURCES
        }
        report["profile_names"] = profiles
        _write_json(output, report)
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
