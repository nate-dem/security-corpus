#!/usr/bin/env python3
"""Stream exact candidate files, recompute hashes/tokens, and report integrity.

No records are altered or filtered. Duplicate content is reported for researcher
policy decisions. A successful integrity check alone is not publication approval:
Qwen review, source permissions, attribution, and release policy remain separate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import duckdb
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ingest.utils import compute_content_hash, compute_token_count  # noqa: E402


REQUIRED = ("source_id", "source_record_id", "record_id", "content", "content_hash",
            "content_length", "license", "source_url", "ingested_at")


def _files(inputs: Sequence[Path]) -> list[Path]:
    files = set()
    for input_path in inputs:
        path = input_path.resolve()
        if path.is_file() and path.suffix == ".parquet":
            found = [path]
        elif path.is_dir():
            found = [file for file in path.rglob("*.parquet") if not file.name.startswith("._")]
        else:
            raise FileNotFoundError(f"Not a Parquet file or directory: {path}")
        if not found:
            raise FileNotFoundError(f"No Parquet files under {path}")
        files.update(found)
    return sorted(files)


def _audit_file(task: tuple[Path, bool]) -> dict:
    path, check_tokens = task
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    result = {"path": str(path), "bytes": before.st_size, "sha256": digest.hexdigest(),
              "records": 0, "stored_tokens": 0, "recomputed_tokens": 0 if check_tokens else None}
    issues = Counter()
    sources = {}
    examples = []
    try:
        parquet = pq.ParquetFile(path)
        result["schema"] = str(parquet.schema_arrow)
        missing = set(REQUIRED) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
        for batch in parquet.iter_batches(batch_size=256, columns=list(REQUIRED)):
            for row in batch.to_pylist():
                result["records"] += 1
                errors = []
                for field in REQUIRED:
                    if field in {"content_length", "ingested_at"}:
                        continue
                    if not isinstance(row[field], str) or not row[field].strip():
                        errors.append(f"invalid_{field}")
                length = row["content_length"]
                if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
                    errors.append("invalid_content_length")
                    length = 0
                result["stored_tokens"] += length
                timestamp = row["ingested_at"]
                if not isinstance(timestamp, datetime) or timestamp.utcoffset() is None:
                    errors.append("invalid_ingested_at")
                if row["record_id"] != f"{row['source_id']}:{row['source_record_id']}":
                    errors.append("record_id_namespace_mismatch")
                content = row["content"]
                if isinstance(content, str):
                    if compute_content_hash(content) != row["content_hash"]:
                        errors.append("content_hash_mismatch")
                    if check_tokens:
                        actual = compute_token_count(content)
                        result["recomputed_tokens"] += actual
                        if actual != row["content_length"]:
                            errors.append("content_length_mismatch")
                source = sources.setdefault(str(row["source_id"]), {"records": 0, "tokens": 0})
                source["records"] += 1
                source["tokens"] += length
                issues.update(errors)
                if errors and len(examples) < 10:
                    examples.append({"row": result["records"], "record_id": row["record_id"], "issues": errors})
    except Exception as error:
        issues["unreadable_or_invalid_file"] += 1
        result["error"] = f"{type(error).__name__}: {error}"
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        issues["file_changed_during_audit"] += 1
    if not result["records"]:
        issues["empty_file"] += 1
    result.update(issues=dict(issues), examples=examples, sources=sources)
    return result


def _duplicates(files: list[Path]) -> dict:
    with duckdb.connect() as connection:
        connection.execute("SET memory_limit = '2GB'")
        connection.read_parquet([str(path) for path in files], union_by_name=True,
                                hive_partitioning=False).create_view("corpus")
        duplicate_ids = connection.execute("""
            SELECT coalesce(sum(n - 1), 0) FROM (
                SELECT count(*) n FROM corpus GROUP BY source_id, record_id HAVING count(*) > 1
            )
        """).fetchone()[0]
        groups, extra, cross_source = connection.execute("""
            SELECT count(*), coalesce(sum(n - 1), 0), count(*) FILTER (WHERE sources > 1)
            FROM (SELECT count(*) n, count(DISTINCT source_id) sources FROM corpus
                  GROUP BY content_hash HAVING count(*) > 1)
        """).fetchone()
    return {"duplicate_record_ids": int(duplicate_ids), "exact_content_groups": int(groups),
            "extra_content_copies": int(extra), "cross_source_content_groups": int(cross_source)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-token-check", action="store_true",
                        help="Diagnostic hash/structure pass only; cannot pass release integrity.")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    files = _files(args.inputs)
    if args.output.resolve() in files:
        raise ValueError("Audit report cannot overwrite a corpus input")
    tasks = [(path, not args.skip_token_check) for path in files]
    results = []
    if args.workers == 1:
        for task in tasks:
            results.append(_audit_file(task))
            print(f"Audited {len(results)}/{len(files)} files", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(_audit_file, tasks):
                results.append(result)
                print(f"Audited {len(results)}/{len(files)} files", flush=True)
    issues, sources = Counter(), {}
    for result in results:
        issues.update(result["issues"])
        for source, counts in result["sources"].items():
            total = sources.setdefault(source, {"records": 0, "tokens": 0})
            total["records"] += counts["records"]
            total["tokens"] += counts["tokens"]
    duplicates = None
    if not issues.get("unreadable_or_invalid_file"):
        duplicates = _duplicates(files)
        if duplicates["duplicate_record_ids"]:
            issues["duplicate_record_ids"] = duplicates["duplicate_record_ids"]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": "cl100k_base", "token_counts_recomputed": not args.skip_token_check,
        "integrity_passed": not issues and not args.skip_token_check,
        "records": sum(result["records"] for result in results),
        "stored_tokens": sum(result["stored_tokens"] for result in results),
        "recomputed_tokens": None if args.skip_token_check else sum(result["recomputed_tokens"] for result in results),
        "issues": dict(issues), "sources": sources, "duplicates": duplicates, "files": results,
        "scope": "Integrity only. Duplicate precedence, semantic quality, attribution and permission require separate review.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))
    return 2 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
