#!/usr/bin/env python3
"""Profile all downloaded shards and export full-text diagnostic examples.

This is a metadata inventory and inspection sample, not normalized ingestion,
a quality filter, or a token-count pass. Raw Parquet files are never rewritten.
Completed per-shard indexes and samples are reusable after interruption.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

if __package__:
    from .download import _directory_lock, _shard_path, _validate_manifest, _write_json
else:
    from download import _directory_lock, _shard_path, _validate_manifest, _write_json


LOG = logging.getLogger("youtube-profile")
TEXT_COLUMNS = (
    "video_id", "video_link", "channel_id", "channel", "title", "date", "license",
    "transcription_language", "original_language", "source_language", "language_id_method",
)
COUNT_COLUMNS = ("word_count", "character_count")


def _sql(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_stat(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _load_inputs(root: Path) -> tuple[dict, str]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    _validate_manifest(manifest)
    digest = _sha256(manifest_path)
    summary = json.loads((root / "download-summary.json").read_text())
    if (summary.get("complete") is not True or summary.get("failed")
            or summary.get("revision") != manifest["revision"]
            or summary.get("manifest_sha256") != digest):
        raise ValueError("Download is incomplete or does not match manifest.json")
    verified = summary.get("verified", [])
    by_path = {row["path"]: row for row in verified}
    if len(by_path) != len(verified) or set(by_path) != {row["path"] for row in manifest["files"]}:
        raise ValueError("Download verification does not cover every shard exactly once")
    for entry in manifest["files"]:
        row = by_path[entry["path"]]
        if row["bytes"] != entry["size"] or (
            entry["hash_algorithm"] == "sha256" and row["sha256"] != entry["hash"]
        ):
            raise ValueError(f"Download checksum evidence mismatch: {entry['path']}")
        path = _shard_path(root / "raw", entry)
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise ValueError(f"Downloaded shard missing or size changed: {entry['path']}")
    return manifest, digest


def _sample_texts(path: Path, rows: list[int]) -> dict[int, str | None]:
    """Read only row groups containing selected examples, in bounded batches."""
    selected = set(rows)
    result = {}
    with pq.ParquetFile(path) as file:
        offset = 0
        for group in range(file.metadata.num_row_groups):
            stop = offset + file.metadata.row_group(group).num_rows
            targets = {row for row in selected if offset <= row < stop}
            if targets:
                cursor = offset
                for batch in file.iter_batches(
                    row_groups=[group], columns=["text"], batch_size=16, use_threads=False,
                ):
                    for row in sorted(targets):
                        if cursor <= row < cursor + batch.num_rows:
                            result[row] = batch.column(0)[row - cursor].as_py()
                    cursor += batch.num_rows
                    if targets <= result.keys():
                        break
            offset = stop
    if result.keys() != selected:
        raise ValueError(f"Could not retrieve requested source rows from {path}")
    return result


def _profile_shard(index: int, entry: dict, root: Path, output: Path,
                   revision: str, samples_per_shard: int, seed: int) -> dict:
    source = _shard_path(root / "raw", entry)
    source_stat = _source_stat(source)
    checkpoint = output / "checkpoints" / f"{index:05d}.json"
    metadata = output / "metadata" / f"{index:05d}.parquet"
    samples = output / "samples" / f"{index:05d}.jsonl"
    identity = {"source_shard": entry["path"], "source_stat": source_stat,
                "upstream_hash": entry["hash"]}
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError(f"Source changed since profiling: {entry['path']}")
        if all(path.is_file() and _sha256(path) == saved.get(key) for path, key in (
            (metadata, "metadata_sha256"), (samples, "samples_sha256"),
        )):
            return {**saved, "resumed": True}

    with pq.ParquetFile(source) as file:
        schema = {field.name: str(field.type) for field in file.schema_arrow}
        expected_rows = file.metadata.num_rows
    if "text" not in schema:
        raise ValueError(f"No transcript text column: {entry['path']}")
    if schema["text"] not in {"string", "large_string", "null"}:
        raise ValueError(f"Unexpected text type in {entry['path']}: {schema['text']}")

    fields = [f"{_sql(entry['path'])} AS source_shard", "file_row_number AS source_row"]
    for column in TEXT_COLUMNS:
        value = f'CAST("{column}" AS VARCHAR)' if column in schema else "NULL::VARCHAR"
        fields.append(f'{value} AS "{column}"')
    for column in COUNT_COLUMNS:
        value = f'TRY_CAST("{column}" AS BIGINT)' if column in schema else "NULL::BIGINT"
        fields.append(f'{value} AS "{column}"')
        # Preserve malformed/negative/fractional supplied lengths as diagnostics.
        invalid = (
            f'"{column}" IS NOT NULL AND ({value} IS NULL OR {value} < 0 '
            f'OR TRY_CAST("{column}" AS DOUBLE) IS DISTINCT FROM CAST({value} AS DOUBLE))'
            if column in schema else "false"
        )
        fields.append(f'{invalid} AS "{column}_invalid"')

    temp_meta = metadata.with_suffix(".tmp")
    temp_samples = samples.with_suffix(".tmp")
    try:
        with duckdb.connect(config={"threads": 1, "memory_limit": "1GB"}) as con:
            con.execute(f"COPY (SELECT {', '.join(fields)} FROM read_parquet("
                        f"{_sql(str(source))}, file_row_number=true, hive_partitioning=false)) "
                        f"TO {_sql(str(temp_meta))} (FORMAT PARQUET, COMPRESSION ZSTD)")
            con.execute(f"CREATE VIEW records AS SELECT * FROM read_parquet({_sql(str(temp_meta))})")
            count = con.execute("SELECT count(*) FROM records").fetchone()[0]
            if count != expected_rows:
                raise ValueError(f"Row count changed during profiling: {entry['path']}")
            cursor = con.execute(
                "SELECT * FROM records ORDER BY md5(? || CAST(source_row AS VARCHAR)), source_row LIMIT ?",
                [f"{seed}:{entry['path']}:", samples_per_shard],
            )
            names = [field[0] for field in cursor.description]
            chosen = [dict(zip(names, row)) for row in cursor.fetchall()]
        texts = _sample_texts(source, [row["source_row"] for row in chosen])
        with temp_samples.open("w", encoding="utf-8") as handle:
            for row in chosen:
                row.update({"text": texts[row["source_row"]], "dataset_revision": revision,
                            "sample_seed": seed, "shard_rows": count,
                            "selection_probability": len(chosen) / count})
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if _source_stat(source) != source_stat:
            raise ValueError(f"Source changed while profiling: {entry['path']}")
        os.replace(temp_meta, metadata)
        os.replace(temp_samples, samples)
        result = {**identity, "rows": count, "samples": len(chosen), "schema": schema,
                  "missing_metadata_columns": sorted(set(TEXT_COLUMNS + COUNT_COLUMNS) - schema.keys()),
                  "metadata_sha256": _sha256(metadata), "samples_sha256": _sha256(samples)}
        _write_json(checkpoint, result)
        return result
    finally:
        temp_meta.unlink(missing_ok=True)
        temp_samples.unlink(missing_ok=True)


def _records(cursor) -> list[dict]:
    names = [field[0] for field in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def _summarize(output: Path, results: list[dict], manifest: dict, workers: int) -> dict:
    files = [str(output / "metadata" / f"{index:05d}.parquet") for index in range(len(results))]
    with duckdb.connect(config={"threads": workers, "memory_limit": "8GB",
                               "temp_directory": str(output / "duckdb-tmp")}) as con:
        con.read_parquet(files, hive_partitioning=False).create_view("records")
        totals = _records(con.execute("""
            SELECT count(*) AS records,
                count(DISTINCT nullif(trim(video_id), '')) AS distinct_video_ids,
                count(DISTINCT nullif(trim(channel_id), '')) AS distinct_channel_ids,
                count(*) FILTER (WHERE nullif(trim(video_id), '') IS NULL) AS missing_video_ids,
                count(*) FILTER (WHERE nullif(trim(channel_id), '') IS NULL) AS missing_channel_ids,
                count(*) FILTER (WHERE nullif(trim(license), '') IS NULL) AS missing_licenses,
                count(*) FILTER (WHERE word_count IS NULL) AS missing_or_unparseable_word_counts,
                count(*) FILTER (WHERE word_count_invalid) AS invalid_word_counts,
                count(*) FILTER (WHERE character_count_invalid) AS invalid_character_counts,
                sum(word_count) FILTER (WHERE NOT word_count_invalid) AS reported_word_count_sum,
                sum(character_count) FILTER (WHERE NOT character_count_invalid) AS reported_character_count_sum
            FROM records
        """))[0]
        if totals["records"] != sum(result["rows"] for result in results):
            raise ValueError("Aggregate metadata row count does not match shard checkpoints")
        languages = _records(con.execute("""
            SELECT transcription_language, count(*) AS records,
                count(DISTINCT nullif(trim(video_id), '')) AS distinct_video_ids,
                sum(word_count) FILTER (WHERE NOT word_count_invalid) AS reported_words,
                quantile_cont(word_count, [0.5, 0.9, 0.99])
                    FILTER (WHERE NOT word_count_invalid) AS reported_word_count_p50_p90_p99,
                max(word_count) FILTER (WHERE NOT word_count_invalid) AS max_reported_word_count
            FROM records GROUP BY transcription_language ORDER BY records DESC, transcription_language
        """))
        language_pairs = _records(con.execute("""
            SELECT transcription_language, original_language, source_language, count(*) AS records
            FROM records GROUP BY ALL ORDER BY records DESC, transcription_language, original_language, source_language
        """))
        licenses = _records(con.execute("""
            SELECT license, count(*) AS records FROM records GROUP BY license ORDER BY records DESC, license
        """))
        repeated = _records(con.execute("""
            SELECT count(*) AS groups, coalesce(sum(copies - 1), 0) AS additional_rows
            FROM (SELECT video_id, transcription_language, count(*) AS copies
                  FROM records WHERE nullif(trim(video_id), '') IS NOT NULL
                  GROUP BY video_id, transcription_language HAVING count(*) > 1)
        """))[0]
    combined = output / "inspection_sample.jsonl"
    temporary = combined.with_suffix(".tmp")
    try:
        with temporary.open("wb") as destination:
            for index in range(len(results)):
                with (output / "samples" / f"{index:05d}.jsonl").open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
        os.replace(temporary, combined)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "complete": True, "dataset_revision": manifest["revision"],
        "completed_at": datetime.now(timezone.utc).isoformat(), "shards": len(results),
        "scan_scope": "All rows: metadata only. Full text: diagnostic samples only.",
        "length_units": "Source-reported words and characters, not tokenizer counts.",
        "totals": totals, "transcription_languages": languages, "language_pairs": language_pairs,
        "licenses": licenses, "repeated_video_language_keys": repeated,
        "repeat_note": "Repeated video/language keys do not establish duplicate text; no rows removed.",
        "inspection_sample": {"path": str(combined), "records": sum(r["samples"] for r in results),
                              "sha256": _sha256(combined), "text_truncated": False,
                              "purpose": "Diagnostic inspection, not a labeled classifier validation set.",
                              "design": "Deterministic uniform selection within each shard; per-row selection probability included."},
        "shard_schemas": [{"shard": r["source_shard"], "schema": r["schema"],
                           "missing_metadata_columns": r["missing_metadata_columns"]} for r in results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Default: DATA_DIR/profile-v1")
    parser.add_argument("--workers", type=int, default=4)
    # Diagnostic display size only: this does not select training records or
    # set the researcher's classifier calibration/acceptance sample size.
    parser.add_argument("--samples-per-shard", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.samples_per_shard < 1:
        parser.error("workers and samples-per-shard must be positive")
    root = args.data_dir.expanduser().resolve()
    output = (args.output_dir or root / "profile-v1").expanduser().resolve()
    if output == root or output.is_relative_to(root / "raw"):
        parser.error("Profile output must be separate from the download root and raw shards")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        manifest, digest = _load_inputs(root)
        output.mkdir(parents=True, exist_ok=True)
        with _directory_lock(root), _directory_lock(output):
            config = {"manifest_sha256": digest, "data_dir": str(root), "seed": args.seed,
                      "samples_per_shard": args.samples_per_shard,
                      "profile_code_sha256": _sha256(Path(__file__)),
                      "duckdb_version": duckdb.__version__,
                      "pyarrow_version": __import__("pyarrow").__version__}
            config_path = output / "run-config.json"
            if config_path.exists() and json.loads(config_path.read_text()) != config:
                raise ValueError("Profile configuration changed; choose a new --output-dir")
            _write_json(config_path, config)
            for directory in ("checkpoints", "metadata", "samples"):
                (output / directory).mkdir(exist_ok=True)
            summary_path = output / "summary.json"
            _write_json(summary_path, {"complete": False, "status": "running"})
            LOG.info("Profiling %d shards with %d CPU workers; output: %s",
                     len(manifest["files"]), args.workers, output)
            results = [None] * len(manifest["files"])
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(_profile_shard, index, entry, root, output,
                                           manifest["revision"], args.samples_per_shard, args.seed): index
                           for index, entry in enumerate(manifest["files"])}
                done = 0
                try:
                    for future in as_completed(futures):
                        index = futures[future]
                        results[index] = future.result()
                        done += 1
                        LOG.info("[%d/%d] %s: %s rows%s", done, len(results),
                                 results[index]["source_shard"], f"{results[index]['rows']:,}",
                                 " (resumed)" if results[index].get("resumed") else "")
                except BaseException:
                    for future in futures:
                        future.cancel()
                    _write_json(summary_path, {"complete": False, "status": "interrupted_or_failed"})
                    raise
            LOG.info("All shards indexed; aggregating languages, lengths, IDs, and inspection sample")
            report = _summarize(output, results, manifest, args.workers)
            _write_json(summary_path, report)
            LOG.info("PROFILE COMPLETE: %s records, %s video IDs; %d diagnostic samples. Report: %s",
                     f"{report['totals']['records']:,}", f"{report['totals']['distinct_video_ids']:,}",
                     report["inspection_sample"]["records"], summary_path)
        return 0
    except KeyboardInterrupt:
        LOG.warning("Interrupted; rerun the same command to reuse completed shard profiles")
        return 130
    except Exception as exc:
        LOG.error("Profile failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
