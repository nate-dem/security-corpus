"""Resumable CPU hash/token checks for mixed canonical and prepared documents."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import importlib.metadata
import json
import logging
from pathlib import Path
import sqlite3

import pyarrow.parquet as pq

from ingest.utils import compute_content_hash, compute_token_count
from scripts.youtube.download import _directory_lock, _write_json
from scripts.youtube.profile import _sha256

LOG = logging.getLogger(__name__)


def read(path):
    return json.loads(Path(path).read_text())


def check_file(entry):
    path = Path(entry["path"])
    if _sha256(path) != entry["sha256"]:
        raise ValueError(f"Manifest checksum mismatch: {path}")
    issues, counts, examples = Counter(), Counter(), []
    with pq.ParquetFile(path) as file:
        names = set(file.schema_arrow.names)
        canonical = "content" in names
        text = "content" if canonical else "text"
        source = "source_id" if canonical else "source"
        identity = (
            ["record_id", "source_record_id"]
            if canonical
            else ["dataset_revision", "source_shard", "source_row"]
        )
        required = {source, text, "content_hash", "content_length", *identity}
        if not required <= names:
            raise ValueError(
                f"Missing document fields in {path}: {sorted(required - names)}"
            )
        for batch in file.iter_batches(
            batch_size=128, columns=sorted(required), use_threads=False
        ):
            for row in batch.to_pylist():
                errors = []
                counts["records"] += 1
                for name in required - {"content_length", "source_row"}:
                    if not isinstance(row[name], str) or not row[name].strip():
                        errors.append(f"invalid_{name}")
                length = row["content_length"]
                if (
                    not isinstance(length, int)
                    or isinstance(length, bool)
                    or length <= 0
                ):
                    errors.append("invalid_content_length")
                else:
                    counts["tokens"] += length
                if (
                    canonical
                    and row["record_id"] != f"{row[source]}:{row['source_record_id']}"
                ):
                    errors.append("record_id_namespace_mismatch")
                if not canonical and (
                    type(row["source_row"]) is not int or row["source_row"] < 0
                ):
                    errors.append("invalid_source_row")
                if isinstance(row[text], str):
                    if compute_content_hash(row[text]) != row["content_hash"]:
                        errors.append("content_hash_mismatch")
                    tokens = compute_token_count(row[text])
                    counts["recomputed_tokens"] += tokens
                    if tokens != length:
                        errors.append("content_length_mismatch")
                issues.update(errors)
                if errors and len(examples) < 10:
                    examples.append(
                        {"row_index": counts["records"] - 1, "issues": errors}
                    )
    for name in ("records", "tokens"):
        if counts[name] != entry[name]:
            issues[f"manifest_{name}_mismatch"] += 1
    if _sha256(path) != entry["sha256"]:
        raise ValueError(f"Input changed during integrity audit: {path}")
    return {
        "path": str(path),
        "sha256": entry["sha256"],
        "bytes": path.stat().st_size,
        "counts": dict(counts),
        "issues": dict(issues),
        "examples": examples,
    }


def execute(manifest, output, rights, workers=7):
    if not 1 <= workers <= 8:
        raise ValueError("Use 1-8 CPU workers")
    output.mkdir(parents=True, exist_ok=True)
    report = read(rights / "summary.json")
    digest = _sha256(manifest)
    if (
        not report["audit_complete"]
        or report["manifest_sha256"] != digest
        or report["ledger_sha256"] != _sha256(rights / "records.sqlite")
    ):
        raise ValueError("Rights ledger is not bound to this manifest")
    binding = {
        "manifest_sha256": digest,
        "code_sha256": _sha256(Path(__file__)),
        "tokenizer": "cl100k_base",
        "tiktoken": importlib.metadata.version("tiktoken"),
    }
    files = read(manifest)["files"]
    if len({f["path"] for f in files}) != len(files):
        raise ValueError("Duplicate manifest paths")
    with _directory_lock(output):
        target = output / "binding.json"
        if target.exists() and read(target) != binding:
            raise ValueError("Integrity audit binding changed")
        _write_json(target, binding)
        _write_json(
            output / "summary.json", {"audit_complete": False, "release_ready": False}
        )
        with sqlite3.connect(output / "checkpoints.sqlite") as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS checked(path TEXT PRIMARY KEY, result TEXT)"
            )
            pending, results = [], []
            for f in files:
                saved = con.execute(
                    "SELECT result FROM checked WHERE path=?", (f["path"],)
                ).fetchone()
                if saved:
                    result = json.loads(saved[0])
                    if (
                        result["sha256"] != f["sha256"]
                        or _sha256(Path(f["path"])) != f["sha256"]
                    ):
                        raise ValueError("Cached integrity input changed")
                    results.append(result)
                else:
                    pending.append(f)
            # I/O task batches bound the executor queue and checkpoint each file.
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for offset in range(0, len(pending), workers * 8):
                    for result in pool.map(
                        check_file, pending[offset : offset + workers * 8]
                    ):
                        con.execute(
                            "INSERT INTO checked VALUES (?,?)",
                            (result["path"], json.dumps(result)),
                        )
                        con.commit()
                        results.append(result)
                        LOG.info(
                            "Integrity %s/%s files; recomputed cl100k_base tokens",
                            len(results),
                            len(files),
                        )
            totals, issues = Counter(), Counter()
            for r in results:
                totals.update(r["counts"])
                issues.update(r["issues"])
        with sqlite3.connect(
            f"file:{(rights / 'records.sqlite').resolve()}?mode=ro", uri=True
        ) as con:
            duplicate_ids = con.execute(
                "SELECT source,record_id,COUNT(*) FROM records WHERE record_id IS NOT NULL GROUP BY source,record_id HAVING COUNT(*)>1"
            ).fetchall()
            duplicate_hashes = con.execute(
                "SELECT COUNT(*),COALESCE(SUM(n-1),0) FROM (SELECT COUNT(*) n FROM records GROUP BY content_hash HAVING COUNT(*)>1)"
            ).fetchone()
        if duplicate_ids:
            issues["duplicate_record_ids"] = sum(r[2] - 1 for r in duplicate_ids)
        saved_totals = read(manifest).get("totals")
        if saved_totals and any(
            totals[k] != saved_totals[k] for k in ("records", "tokens")
        ):
            issues["manifest_total_mismatch"] += 1
        if _sha256(manifest) != digest:
            raise ValueError("Manifest changed during integrity audit")
        result = {
            **binding,
            "audit_complete": True,
            "integrity_passed": not issues,
            "release_ready": False,
            "token_counts_recomputed": True,
            "totals": dict(totals),
            "payload_bytes": sum(r["bytes"] for r in results),
            "issues": dict(issues),
            "duplicate_record_ids": duplicate_ids,
            "exact_content_groups": duplicate_hashes[0],
            "extra_content_copies": duplicate_hashes[1],
            "files": sorted(results, key=lambda r: r["path"]),
            "scope": "Document identity, hashes and token counts only. Exact duplicate precedence, permissions, attribution, human quality review, and publication packaging remain separate.",
        }
        _write_json(output / "summary.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output", "rights"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    execute(args.manifest, args.output, args.rights, args.workers)


if __name__ == "__main__":
    main()
