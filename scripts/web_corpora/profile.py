"""CPU inventory of every raw record: exact tokens/hashes and diagnostic samples.

No content selection, canonical schema change, or rewrite of the source data.
The Parquet outputs are diagnostic sidecars. Completed shards are checksummed
and reusable; a preempted shard is scanned again on the next run.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import logging
import multiprocessing
import os
from pathlib import Path
import random
from urllib.parse import urlsplit

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken

from ingest.utils import compute_content_hash, compute_token_count
from scripts.youtube.download import _directory_lock, _shard_path, _verify_file, _write_json
from .download import load_manifest
from .sources import SOURCES, data_files


LOG = logging.getLogger("web-profile")
FEATURE_SCHEMA = pa.schema([
    ("source", pa.string()), ("source_shard", pa.string()), ("source_row", pa.int64()),
    ("upstream_id", pa.string()), ("text_state", pa.string()), ("content_hash", pa.string()),
    ("content_length", pa.int64()), ("characters", pa.int64()), ("url", pa.string()),
    ("domain", pa.string()), ("language", pa.string()), ("upstream_relevant", pa.string()),
    ("upstream_probability", pa.float64()), ("upstream_token_count", pa.string()),
    ("upstream_license", pa.string()),
])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def iter_rows(path: Path):
    """Stream actual upstream formats. Invalid JSON fails rather than losing rows."""
    if path.name.endswith(".parquet"):
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=128, use_threads=False):
                yield from batch.to_pylist()
    elif path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    value = json.loads(line)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Invalid JSON at {path.name}:{line_number}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"Expected a JSON object at {path.name}:{line_number}")
                yield value
    else:
        raise ValueError(f"Unsupported data format: {path.name}")


def _string(value):
    if value is None:
        return None
    return value if isinstance(value, str) else json.dumps(value, default=str, sort_keys=True)


def text_field(source: str) -> str:
    # Confirmed from the pinned Primus shard on Marlowe (2026-09-12).
    # These are source field mappings, not alternative content filters.
    return "content" if source == "primus-fineweb" else "text"


def features(row: dict, source: str, shard: str, index: int) -> dict:
    metadata = row.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except ValueError:
            metadata = None
    if not isinstance(metadata, dict):
        metadata = {}

    def supplied(key):
        return row[key] if row.get(key) is not None else metadata.get(key)

    text = row.get(text_field(source))
    state = ("missing" if text is None else "non_string" if not isinstance(text, str)
             else "blank" if not text.strip() else "valid")
    url = _string(supplied("url"))
    try:
        domain = urlsplit(url).hostname if url else None
    except ValueError:
        domain = None
    probability = supplied("probability")
    return {
        "source": source, "source_shard": shard, "source_row": index,
        "upstream_id": _string(supplied("id")), "text_state": state,
        "content_hash": compute_content_hash(text) if state == "valid" else None,
        "content_length": compute_token_count(text) if isinstance(text, str) else None,
        "characters": len(text) if isinstance(text, str) else None,
        "url": url, "domain": domain, "language": _string(supplied("language")),
        "upstream_relevant": _string(supplied("relevant")),
        "upstream_probability": float(probability) if type(probability) in (int, float) else None,
        "upstream_token_count": _string(supplied("token_count")),
        "upstream_license": _string(supplied("license")),
    }


def load_inputs(root: Path, source: str) -> tuple[dict, str]:
    manifest = load_manifest(root, source, offline=True)
    digest = sha256(root / "manifest.json")
    summary = json.loads((root / "download-summary.json").read_text())
    if (summary.get("complete") is not True or summary.get("failed")
            or summary.get("revision") != manifest["revision"]
            or summary.get("repo_id") != manifest["repo_id"]
            or summary.get("manifest_sha256") != digest):
        raise ValueError("Download must be complete and match the pinned manifest")
    verified = summary.get("verified", [])
    by_path = {item["path"]: item for item in verified}
    if len(by_path) != len(verified) or set(by_path) != {e["path"] for e in manifest["files"]}:
        raise ValueError("Download report does not cover all files exactly once")
    for entry in manifest["files"]:
        observed = by_path[entry["path"]]
        if observed["bytes"] != entry["size"] or (
                entry["hash_algorithm"] == "sha256" and observed["sha256"] != entry["hash"]):
            raise ValueError(f"Invalid download evidence: {entry['path']}")
    return manifest, digest


def _profile_one(task: tuple) -> dict:
    index, entry, root, output, source, revision, sample_size, seed = task
    pa.set_cpu_count(1)
    path = _shard_path(root / "raw", entry)
    raw_stat = _verify_stat(path)
    verified = _verify_file(path, entry)  # Detect same-size corruption, including on resume.
    stem = f"{index:05d}"
    sidecar = output / "metadata" / f"{stem}.parquet"
    samples_path = output / "samples" / f"{stem}.jsonl"
    checkpoint = output / "checkpoints" / f"{stem}.json"
    if checkpoint.exists():
        try:
            cached = json.loads(checkpoint.read_text())
            if (cached["raw_sha256"] == verified["sha256"]
                    and cached["metadata_sha256"] == sha256(sidecar)
                    and cached["samples_sha256"] == sha256(samples_path)):
                return {**cached, "resumed": True}
        except (OSError, ValueError, KeyError):
            pass  # Rebuild incomplete/corrupt local products from verified raw bytes.

    rng = random.Random(f"{seed}:{source}:{entry['path']}")
    samples, buffer, column_types = [], [], {}
    count = 0
    text_seen = False
    temporary = sidecar.with_suffix(".tmp")
    try:
        with pq.ParquetWriter(temporary, FEATURE_SCHEMA, compression="zstd") as writer:
            for count, row in enumerate(iter_rows(path), 1):
                text_seen |= text_field(source) in row
                for key, value in row.items():
                    column_types.setdefault(key, set()).add(type(value).__name__)
                feature = features(row, source, entry["path"], count - 1)
                buffer.append(feature)
                slot = len(samples) if len(samples) < sample_size else rng.randrange(count)
                if slot < sample_size:
                    sample = {"source": source, "dataset_revision": revision,
                              "source_shard": entry["path"], "source_row": count - 1,
                              "features": feature, "raw_record": row}
                    if slot == len(samples):
                        samples.append(sample)
                    else:
                        samples[slot] = sample
                if len(buffer) >= 1024:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=FEATURE_SCHEMA))
                    buffer.clear()
                if count % 10000 == 0:
                    print(f"{source}/{entry['path']}: {count:,} records tokenized", flush=True)
            if buffer:
                writer.write_table(pa.Table.from_pylist(buffer, schema=FEATURE_SCHEMA))
        if count and not text_seen:
            raise ValueError(f"No {text_field(source)!r} column in nonempty shard {entry['path']}; inspect upstream schema")
        if _verify_stat(path) != raw_stat:
            raise ValueError(f"Raw shard changed while profiling: {entry['path']}")
        os.replace(temporary, sidecar)
    finally:
        temporary.unlink(missing_ok=True)

    temporary_sample = samples_path.with_suffix(".tmp")
    with temporary_sample.open("w", encoding="utf-8") as handle:
        for sample in sorted(samples, key=lambda item: item["source_row"]):
            sample["selection_probability"] = min(sample_size, count) / count
            handle.write(json.dumps(sample, ensure_ascii=False, default=str) + "\n")
    os.replace(temporary_sample, samples_path)
    result = {"source_shard": entry["path"], "rows": count, "samples": len(samples),
              "raw_sha256": verified["sha256"], "metadata_sha256": sha256(sidecar),
              "samples_sha256": sha256(samples_path),
              "column_types": {key: sorted(value) for key, value in column_types.items()}}
    _write_json(checkpoint, result)
    return result


def _verify_stat(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _query_rows(connection, sql):
    cursor = connection.execute(sql)
    names = [entry[0] for entry in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def summarize(output: Path, results: list[dict], manifest: dict, workers: int) -> dict:
    paths = [str(output / "metadata" / f"{index:05d}.parquet") for index in range(len(results))]
    with duckdb.connect() as connection:
        connection.execute("SET memory_limit = '4GB'")
        connection.execute(f"SET threads = {workers}")
        connection.execute("SET temp_directory = ?", [str(output / "duckdb-temp")])
        connection.read_parquet(paths, hive_partitioning=False).create_view("records")
        totals = _query_rows(connection, """
            SELECT count(*) records, count(*) FILTER (WHERE text_state='valid') valid_text_records,
                   coalesce(sum(content_length), 0) raw_tokens,
                   coalesce(sum(content_length) FILTER (WHERE text_state='valid'), 0) valid_text_tokens,
                   count(*) FILTER (WHERE url IS NULL OR trim(url)='') missing_urls,
                   count(*) FILTER (WHERE upstream_license IS NULL OR trim(upstream_license)='') missing_page_licenses,
                   quantile_cont(content_length, [0.5, 0.9, 0.99]) token_percentiles
            FROM records
        """)[0]
        unique = _query_rows(connection, """
            SELECT count(*) unique_texts, coalesce(sum(tokens), 0) exact_unique_tokens,
                   coalesce(sum(copies-1), 0) extra_exact_copies
            FROM (SELECT content_hash, max(content_length) tokens, count(*) copies
                  FROM records WHERE content_hash IS NOT NULL GROUP BY content_hash)
        """)[0]
        strata = _query_rows(connection, """
            SELECT upstream_relevant, text_state, count(*) records,
                   coalesce(sum(content_length), 0) tokens
            FROM records GROUP BY ALL ORDER BY records DESC
        """)
        languages = _query_rows(connection, """
            SELECT language, count(*) records, coalesce(sum(content_length),0) tokens
            FROM records GROUP BY language ORDER BY records DESC
        """)
        domains = _query_rows(connection, """
            SELECT domain, count(*) records, coalesce(sum(content_length),0) tokens
            FROM records GROUP BY domain ORDER BY tokens DESC LIMIT 100
        """)
    combined = output / "inspection_sample.jsonl"
    temporary = combined.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        for index in range(len(results)):
            with (output / "samples" / f"{index:05d}.jsonl").open("rb") as source:
                while block := source.read(1024 * 1024):
                    handle.write(block)
    os.replace(temporary, combined)
    return {
        "complete": True, "source": manifest["source"], "repo_id": manifest["repo_id"],
        "revision": manifest["revision"], "completed_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": "cl100k_base", "text_field": text_field(manifest["source"]),
        "totals": totals, "exact_deduplication": unique,
        "upstream_relevance_strata": strata, "languages": languages, "top_domains": domains,
        "inspection_sample": {"records": sum(r["samples"] for r in results),
                              "sha256": sha256(combined), "text_truncated": False,
                              "design": "Uniform reservoir within each shard, with selection probability; "
                                        "diagnostic only, not an unweighted corpus yield estimate."},
        "scope": "All raw records, no filtering. Exact uniqueness within this source only. "
                 "No near-dedup, factual verification, or redistribution decision. "
                 "Upstream relevance and licenses are supplied values, not our judgments.",
        "shards": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source", choices=[*SOURCES, "all"], default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--profile-name", default="profile-v1")
    # Diagnostic sampling only, not a research quality threshold.
    parser.add_argument("--samples-per-shard", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.samples_per_shard < 1:
        parser.error("workers and samples-per-shard must be positive")
    if (not args.profile_name or Path(args.profile_name).name != args.profile_name
            or args.profile_name in {".", "..", "raw"}):
        parser.error("profile-name must be a single directory name other than raw")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sources = list(SOURCES) if args.source == "all" else [args.source]
    try:
        for source in sources:
            root = args.data_dir.expanduser().resolve() / source
            with _directory_lock(root):
                manifest, digest = load_inputs(root, source)
                output = root / args.profile_name
                output.mkdir(exist_ok=True)
                config = {"manifest_sha256": digest, "seed": args.seed, "text_field": text_field(source),
                          "samples_per_shard": args.samples_per_shard,
                          "code": {str(path.relative_to(Path(__file__).parents[2])): sha256(path) for path in (
                              Path(__file__), Path(__file__).with_name("sources.py"),
                              Path(__file__).with_name("download.py"),
                              Path(__file__).parents[1] / "youtube/download.py",
                              Path(__file__).parents[2] / "src/ingest/utils.py")},
                          "versions": {"pyarrow": pa.__version__, "duckdb": duckdb.__version__,
                                       "tiktoken": tiktoken.__version__}}
                config_path = output / "run-config.json"
                if config_path.exists() and json.loads(config_path.read_text()) != config:
                    raise ValueError("Profile code/configuration changed; choose a new --profile-name")
                _write_json(config_path, config)
                for directory in ("metadata", "samples", "checkpoints"):
                    (output / directory).mkdir(exist_ok=True)
                _write_json(output / "summary.json", {"complete": False, "status": "running"})
                entries = data_files(manifest)
                results = [None] * len(entries)
                tasks = [(index, entry, root, output, source, manifest["revision"],
                          args.samples_per_shard, args.seed) for index, entry in enumerate(entries)]
                LOG.info("Tokenizing %s: %d shards, %d CPU workers", source, len(entries), args.workers)
                if args.workers == 1:
                    for index, task in enumerate(tasks):
                        results[index] = _profile_one(task)
                        LOG.info("[%d/%d] %s", index + 1, len(entries), results[index]["source_shard"])
                else:
                    with ProcessPoolExecutor(max_workers=args.workers,
                                             mp_context=multiprocessing.get_context("spawn")) as executor:
                        futures = {executor.submit(_profile_one, task): task[0] for task in tasks}
                        done = 0
                        try:
                            for future in as_completed(futures):
                                results[futures[future]] = future.result()
                                done += 1
                                LOG.info("[%d/%d] %s", done, len(entries), results[futures[future]]["source_shard"])
                        except BaseException:
                            for future in futures:
                                future.cancel()
                            raise
                report = summarize(output, results, manifest, args.workers)
                report["manifest_sha256"] = digest
                report["run_config_sha256"] = sha256(config_path)
                _write_json(output / "summary.json", report)
                LOG.info("PROFILE COMPLETE: %s; %s raw tokens; %s exact-unique tokens. Report: %s",
                         source, f"{report['totals']['raw_tokens']:,}",
                         f"{report['exact_deduplication']['exact_unique_tokens']:,}", output / "summary.json")
        return 0
    except KeyboardInterrupt:
        LOG.warning("Interrupted; rerun to resume completed shards")
        return 130
    except Exception as exc:
        LOG.error("Profile failed (%s): %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
