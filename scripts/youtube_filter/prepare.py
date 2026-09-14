"""Extract a reproducible English transcript pilot from the completed profile.

The random arm samples raw English rows without relevance/length/channel gates.
Previously inspected cases form a separate diagnostic arm. This selects pilot
examples, not the production corpus. Every selected raw row retains lineage.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path

import duckdb

from ingest.utils import compute_content_hash, compute_token_count
from scripts.youtube.download import _directory_lock, _shard_path, _write_json
from scripts.youtube.inspect_profile import read_verified_sample
from scripts.youtube.profile import _load_inputs, _records, _sample_texts, _sha256, _source_stat


LOG = logging.getLogger("youtube-pilot")


def profile_metadata(root: Path, profile: Path) -> tuple[dict, list[str], dict]:
    manifest, digest = _load_inputs(root)
    summary = json.loads((profile / "summary.json").read_text())
    config = json.loads((profile / "run-config.json").read_text())
    if (summary.get("complete") is not True or summary["dataset_revision"] != manifest["revision"]
            or config.get("manifest_sha256") != digest):
        raise ValueError("Completed profile does not match the downloaded snapshot")
    paths, checkpoints = [], {}
    for index, entry in enumerate(manifest["files"]):
        checkpoint = json.loads((profile / "checkpoints" / f"{index:05d}.json").read_text())
        metadata = profile / "metadata" / f"{index:05d}.parquet"
        if (checkpoint["source_shard"] != entry["path"] or checkpoint["upstream_hash"] != entry["hash"]
                or checkpoint["metadata_sha256"] != _sha256(metadata)):
            raise ValueError(f"Profile metadata changed: {entry['path']}")
        paths.append(str(metadata))
        checkpoints[entry["path"]] = checkpoint
    return manifest, paths, checkpoints


def select_rows(paths: list[str], sample_size: int, case_ids: list[str], case_variants: int,
                seed: int, temp_dir: Path) -> tuple[list[dict], dict]:
    with duckdb.connect(config={"memory_limit": "4GB", "threads": 2,
                               "temp_directory": str(temp_dir)}) as con:
        con.read_parquet(paths, hive_partitioning=False).create_view("metadata")
        # Exact label en is the approved pilot frame, including translated rows.
        con.execute("CREATE VIEW english AS SELECT * FROM metadata WHERE transcription_language='en'")
        total = con.execute("SELECT count(*) FROM english").fetchone()[0]
        random_rows = _records(con.execute("""
            SELECT * FROM english ORDER BY md5(? || source_shard || ':' || CAST(source_row AS VARCHAR)),
                source_shard, source_row LIMIT ?
        """, [f"{seed}:", sample_size]))
        cases = _records(con.execute("""
            SELECT * FROM english WHERE video_id IN (SELECT unnest(?))
            QUALIFY row_number() OVER (PARTITION BY video_id ORDER BY
                md5(? || source_shard || ':' || CAST(source_row AS VARCHAR)), source_shard, source_row) <= ?
            ORDER BY video_id, source_shard, source_row
        """, [case_ids, f"{seed}:case:", case_variants])) if case_ids else []
    selected = {}
    for arm, rows in (("random_english_rows", random_rows), ("diagnostic_cases", cases)):
        for row in rows:
            key = (row["source_shard"], row["source_row"])
            item = selected.setdefault(key, dict(row, sample_arms=[], random_inclusion_probability=None))
            item["sample_arms"].append(arm)
            if arm == "random_english_rows":
                item["random_inclusion_probability"] = min(sample_size, total) / total
    return sorted(selected.values(), key=lambda r: (r["source_shard"], r["source_row"])), {
        "english_raw_rows": total, "random_rows": len(random_rows), "diagnostic_rows": len(cases),
        "missing_case_video_ids": sorted(set(case_ids) - {row["video_id"] for row in cases}),
    }


def write_pilot(rows: list[dict], output: Path, revision: str, counts: dict) -> dict:
    candidates, lineage = {}, []
    for row in rows:
        text = row.get("text")
        status = "blank_or_missing" if text is None or not text.strip() else "candidate"
        digest = compute_content_hash(text) if status == "candidate" else None
        lineage.append({**row, "content_hash": digest, "status": status})
        if digest:
            record = candidates.setdefault(digest, {"content_hash": digest, "text": text,
                "content_length": compute_token_count(text), "dataset_revision": revision,
                "locations": [], "sample_arms": []})
            record["locations"].append({k: row.get(k) for k in (
                "source_shard", "source_row", "video_id", "video_link", "channel_id", "channel",
                "license", "transcription_language", "original_language", "sample_arms",
                "random_inclusion_probability")})
            record["sample_arms"] = sorted(set(record["sample_arms"]) | set(row["sample_arms"]))
    for filename, values in (("candidates.jsonl", list(candidates.values())), ("lineage.jsonl", lineage)):
        path = output / filename
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        temporary.replace(path)
    report = {"complete": True, "dataset_revision": revision, "sampling": counts,
              "selected_raw_rows": len(rows), "unique_candidate_texts": len(candidates),
              "blank_or_missing_rows": sum(r["status"] != "candidate" for r in lineage),
              "exact_duplicate_extra_rows": len(rows) - len(candidates) - sum(r["status"] != "candidate" for r in lineage),
              "candidate_tokens": sum(r["content_length"] for r in candidates.values()),
              "files": {name: _sha256(output / name) for name in ("candidates.jsonl", "lineage.jsonl")},
              "scope": "Pilot only. No corpus keep/drop decisions or yield estimate. Random sampling is over raw "
                       "English rows; duplicate/translation clusters affect uncertainty. Diagnostic cases are enriched."}
    _write_json(output / "summary.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inspection-only", action="store_true",
                        help="Use the existing English diagnostic samples; no raw shards needed. Smoke test only.")
    # RESEARCHER: tune pilot sample size; it does not set any production threshold.
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--case-variants", type=int, default=2)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("pilot_cases.json"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.sample_size < 1 or args.case_variants < 1:
        parser.error("sample-size and case-variants must be positive")
    if not args.data_dir and not (args.inspection_only and args.profile_dir):
        parser.error("data-dir is required unless using inspection-only with profile-dir")
    profile = (args.profile_dir or args.data_dir / "profile-v1").resolve()
    output = args.output_dir.resolve()
    if output == profile or output.is_relative_to(profile):
        parser.error("Keep pilot output separate from the completed profile")
    if args.data_dir and (output == args.data_dir.resolve() or output.is_relative_to(args.data_dir.resolve() / "raw")):
        parser.error("Keep pilot output separate from the download root and raw data")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output.mkdir(parents=True, exist_ok=True)
    with _directory_lock(output):
        summary, inspected = read_verified_sample(profile)
        config = {"profile_summary_sha256": _sha256(profile / "summary.json"),
                  "inspection_only": args.inspection_only, "sample_size": args.sample_size,
                  "case_variants": args.case_variants, "cases_sha256": _sha256(args.cases),
                  "seed": args.seed, "prepare_sha256": _sha256(Path(__file__)),
                  "duckdb_version": duckdb.__version__}
        config_path = output / "run-config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("Pilot preparation configuration changed; choose a new output-dir")
        _write_json(config_path, config)
        _write_json(output / "summary.json", {"complete": False, "status": "preparing"})
        if args.inspection_only:
            rows = [dict(row, sample_arms=["existing_diagnostic_sample"], random_inclusion_probability=None)
                    for row in inspected if row["transcription_language"] == "en"]
            counts = {"inspection_only": True, "not_a_random_english_sample": True}
        else:
            root = args.data_dir.resolve()
            with _directory_lock(root):
                manifest, paths, checkpoints = profile_metadata(root, profile)
                case_ids = [r["video_id"] for r in json.loads(args.cases.read_text())["cases"]]
                rows, counts = select_rows(paths, args.sample_size, case_ids, args.case_variants,
                                           args.seed, output / "duckdb-temp")
                if not rows:
                    raise ValueError("No English rows found in the completed profile")
                groups = defaultdict(list)
                for row in rows:
                    groups[row["source_shard"]].append(row)
                entries = {e["path"]: e for e in manifest["files"]}
                cache = output / "extracted"
                cache.mkdir(exist_ok=True)
                for index, (name, group) in enumerate(sorted(groups.items())):
                    path = _shard_path(root / "raw", entries[name])
                    if _source_stat(path) != checkpoints[name]["source_stat"]:
                        raise ValueError(f"Raw source changed since profiling: {name}")
                    saved = cache / f"{index:05d}.json"
                    locations = [r["source_row"] for r in group]
                    # Cache is advisory: its own checksum is recorded separately.
                    saved_hash = saved.with_suffix(".sha256")
                    if saved.exists() and saved_hash.exists() and _sha256(saved) == saved_hash.read_text().strip():
                        cached = json.loads(saved.read_text())
                    else:
                        cached = None
                    if cached is None or cached["source_shard"] != name or cached["rows"] != locations:
                        texts = _sample_texts(path, locations)
                        cached = {"source_shard": name, "rows": locations, "texts": [texts[i] for i in locations]}
                        _write_json(saved, cached)
                        saved_hash.write_text(_sha256(saved) + "\n")
                    for row, text in zip(group, cached["texts"]):
                        row["text"] = text
                    if _source_stat(path) != checkpoints[name]["source_stat"]:
                        raise ValueError(f"Raw source changed during extraction: {name}")
                    LOG.info("[%d/%d] Retrieved %d English transcripts from %s", index+1, len(groups), len(group), name)
        report = write_pilot(rows, output, summary["dataset_revision"], counts)
        LOG.info("PILOT READY: %d distinct texts, %s tokens. %s", report["unique_candidate_texts"],
                 f"{report['candidate_tokens']:,}", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
