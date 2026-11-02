"""Apply approved MITRE version exclusions and audit the final local corpus.

Canonical/license export remains separately bound to the frozen CPU snapshot.
This derivative removes only approved physical rows. Unchanged Parquet files
use hard links on the same filesystem; all report writes use new files. No
existing input or remote file is modified. Run as a module from the repo.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import gzip
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.release.audit_snapshot import check_file
from scripts.release.correct_licenses import read, require, sha
from scripts.release.export_canonical import BASE_FIELDS
from scripts.youtube.download import _directory_lock, _write_json

LOG = logging.getLogger(__name__)
VERSION = "final-local-corpus-v1"


def contained(root, relative):
    root = root.resolve()
    name = Path(relative)
    path = root / name
    require(
        not name.is_absolute()
        and ".." not in name.parts
        and path.resolve() == path.absolute()
        and path.resolve().is_relative_to(root)
        and path != root,
        f"Unsafe local file path: {relative}",
    )
    return path


def verified_export(root):
    marker = read(root / "canonical-export-complete.json")
    require(marker["export_complete"], "Canonical export incomplete")
    for key, name in (
        ("manifest_sha256", "corpus-manifest.json"),
        ("audit_sha256", "audit.json"),
        ("schema_sha256", "schema.json"),
        ("record_metadata_sha256", "record-metadata.jsonl.gz"),
    ):
        require(sha(root / name) == marker[key], f"Canonical report changed: {name}")
    for name, digest in marker["evidence_sha256"].items():
        require(sha(contained(root, name)) == digest, "Canonical evidence changed")
    manifest = read(root / "corpus-manifest.json")
    seen = set()
    for entry in manifest["files"]:
        path = contained(root, entry["relative_path"])
        require(path not in seen, "Duplicate canonical path")
        seen.add(path)
        require(sha(path) == entry["sha256"], f"Canonical payload changed: {path}")
    return marker, manifest


def validate_variant(table, proof):
    index = proof["row_index"]
    require(type(index) is int and 0 <= index < len(table), "Invalid approved row")
    row = table.slice(index, 1).to_pylist()[0]
    raw = json.loads(row["raw"]) if isinstance(row["raw"], str) else row["raw"]
    require(row["source_id"] == "mitre-attack", "Exclusion must be MITRE ATT&CK")
    for key, expected in (
        ("record_id", proof["record_id"]),
        ("content_hash", proof["content_hash"]),
        ("content_length", proof["tokens"]),
        ("content", proof["content"]),
    ):
        require(row[key] == expected, f"Approved row differs: {key}")
    for key, expected in (
        ("id", proof["stix_id"]),
        ("modified", proof["stix_modified"]),
        ("x_mitre_version", proof["mitre_version"]),
        ("description", row["content"]),
    ):
        require(raw[key] == expected, f"Approved source version differs: {key}")


def approved_rows(plan_path, manifest, canonical):
    require(
        sha(plan_path) == plan_path.with_suffix(".sha256").read_text().strip(),
        "Approved exclusion plan changed",
    )
    plan = read(plan_path)
    require(plan["status"] == "approved_by_researcher", "Exclusions not approved")
    for key in ("records", "tokens"):
        require(
            manifest["totals"][key] == plan["before"][key], "Unexpected corpus scope"
        )
    by_digest = defaultdict(list)
    for ordinal, entry in enumerate(manifest["files"]):
        by_digest[entry["original_sha256"]].append((ordinal, entry))
    drops, retained = {}, set()
    for group in plan["groups"]:
        old, new = group["older"], group["newer"]
        require(
            old["stix_id"] == new["stix_id"]
            and old["record_id"] == new["record_id"] == group["record_id"]
            and datetime.fromisoformat(old["stix_modified"].replace("Z", "+00:00"))
            < datetime.fromisoformat(new["stix_modified"].replace("Z", "+00:00")),
            "Invalid source version precedence",
        )
        for label, proof in (("older", old), ("newer", new)):
            matches = by_digest[proof["input_sha256"]]
            require(len(matches) == 1, "Approved source file missing or ambiguous")
            ordinal, entry = matches[0]
            with pq.ParquetFile(contained(canonical, entry["relative_path"])) as pf:
                validate_variant(pf.read(), proof)
            key = (ordinal, proof["row_index"])
            if label == "older":
                require(key not in drops, "Repeated exclusion")
                drops[key] = proof
            else:
                retained.add(key)
    require(not set(drops) & retained, "Retained version also excluded")
    require(
        len(drops) == plan["proposed_reduction"]["records"], "Exclusion count differs"
    )
    require(
        sum(p["tokens"] for p in drops.values())
        == plan["proposed_reduction"]["tokens"],
        "Exclusion token total differs",
    )
    return plan, drops


def link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def prepare(canonical, plan_path, output):
    canonical, plan_path, output = (p.resolve() for p in (canonical, plan_path, output))
    require(not output.exists(), "Final output must be a new directory")
    require(not output.is_relative_to(canonical), "Output overlaps canonical export")
    marker, manifest = verified_export(canonical)
    plan, drops = approved_rows(plan_path, manifest, canonical)
    output.mkdir(parents=True)
    (output / "records").mkdir()
    evidence = output / "evidence"
    shutil.copytree(canonical / "evidence", evidence, copy_function=link_or_copy)
    for name in (
        "schema.json",
        "canonical-export-complete.json",
        "audit.json",
        "corpus-manifest.json",
    ):
        shutil.copyfile(canonical / name, evidence / ("canonical-" + name))
    shutil.copyfile(plan_path, evidence / "approved-mitre-exclusions.json")
    shutil.copyfile(canonical / "schema.json", output / "schema.json")
    files = []
    for ordinal, entry in enumerate(manifest["files"]):
        source = contained(canonical, entry["relative_path"])
        target = contained(output, entry["relative_path"])
        require(
            sha(source) == entry["sha256"], "Canonical input changed during preparation"
        )
        excluded = sorted(row for file, row in drops if file == ordinal)
        if excluded:
            with pq.ParquetFile(source) as pf:
                before = pf.read()
            kept = [i for i in range(len(before)) if i not in excluded]
            after = before.take(pa.array(kept, type=pa.int64()))
            pq.write_table(after, target, compression="zstd")
            with pq.ParquetFile(target) as pf:
                require(
                    pf.read().equals(after, check_metadata=True),
                    "Exclusion roundtrip changed rows",
                )
            pq.write_table(
                before.take(excluded),
                evidence / f"excluded-{ordinal:06d}.parquet",
                compression="zstd",
            )
        else:
            link_or_copy(source, target)
        files.append(
            dict(
                entry,
                path=str(target),
                sha256=sha(target),
                bytes=target.stat().st_size,
                records=entry["records"] - len(excluded),
                tokens=entry["tokens"]
                - sum(drops[(ordinal, r)]["tokens"] for r in excluded),
                canonical_sha256=entry["sha256"],
                excluded_original_rows=excluded,
            )
        )
    seen_drops, counts, output_rows = set(), Counter(), Counter()
    with (
        gzip.open(canonical / "record-metadata.jsonl.gz", "rt") as source,
        gzip.open(output / "record-metadata.jsonl.gz", "wt") as target,
    ):
        for line in source:
            note = json.loads(line)
            ordinal, row = note["file"], note["row_index"]
            require(
                0 <= ordinal < len(files) and row == counts[ordinal],
                "Metadata row sequence differs",
            )
            counts[ordinal] += 1
            key = (ordinal, row)
            if key in drops:
                require(
                    note["record_id"] == drops[key]["record_id"]
                    and note["content_hash"] == drops[key]["content_hash"],
                    "Excluded metadata identity differs",
                )
                seen_drops.add(key)
                continue
            note["output_row_index"] = output_rows[ordinal]
            output_rows[ordinal] += 1
            target.write(json.dumps(note, ensure_ascii=False) + "\n")
    require(seen_drops == set(drops), "Excluded metadata missing")
    for ordinal, entry in enumerate(files):
        require(
            counts[ordinal] == manifest["files"][ordinal]["records"],
            "Metadata input count differs",
        )
        require(
            output_rows[ordinal] == entry["records"], "Metadata output count differs"
        )
    require(
        sha(canonical / "record-metadata.jsonl.gz") == marker["record_metadata_sha256"],
        "Metadata changed during preparation",
    )
    totals = {key: sum(f[key] for f in files) for key in ("records", "tokens", "bytes")}
    require(
        all(totals[k] == plan["projected_after"][k] for k in ("records", "tokens")),
        "Final totals differ from approved correction",
    )
    _write_json(
        output / "corpus-manifest.json",
        dict(
            files=files,
            totals=totals,
            scope=manifest.get("scope"),
            original_manifest_sha256=manifest["original_manifest_sha256"],
            canonical_manifest_sha256=marker["manifest_sha256"],
            record_metadata="record-metadata.jsonl.gz",
            release_ready=False,
        ),
    )
    result = dict(
        version=VERSION,
        prepared=True,
        totals=totals,
        manifest_sha256=sha(output / "corpus-manifest.json"),
        record_metadata_sha256=sha(output / "record-metadata.jsonl.gz"),
        schema_sha256=sha(output / "schema.json"),
        evidence_sha256={
            str(p.relative_to(output)): sha(p)
            for p in evidence.rglob("*")
            if p.is_file()
        },
        excluded_records=len(drops),
        excluded_tokens=plan["proposed_reduction"]["tokens"],
        retained_text_and_quality_columns_unchanged=True,
        code_sha256=sha(Path(__file__)),
        metadata_fully_resolved=False,
        release_ready=False,
    )
    _write_json(output / "final-prepared.json", result)
    return result


def verify_prepared(root):
    prepared = read(root / "final-prepared.json")
    require(
        prepared["prepared"] and prepared["version"] == VERSION,
        "Incomplete final preparation",
    )
    for key, name in (
        ("manifest_sha256", "corpus-manifest.json"),
        ("record_metadata_sha256", "record-metadata.jsonl.gz"),
        ("schema_sha256", "schema.json"),
    ):
        require(sha(root / name) == prepared[key], f"Final report changed: {name}")
    for name, digest in prepared["evidence_sha256"].items():
        require(sha(contained(root, name)) == digest, "Final evidence changed")
    manifest = read(root / "corpus-manifest.json")
    paths = []
    for entry in manifest["files"]:
        path = contained(root, entry["relative_path"])
        require(path.is_file(), f"Missing final file: {path}")
        paths.append(path)
        entry["path"] = str(path)
    require(len(set(paths)) == len(paths), "Duplicate final file paths")
    return prepared, manifest


def quality_file(entry):
    result = check_file(entry)  # Recomputes every content hash and cl100k_base length.
    issues = Counter(result["issues"])
    licenses, sources, identities = Counter(), defaultdict(Counter), []
    columns = {
        "source_id",
        "record_id",
        "content_hash",
        "content_length",
        "license",
        "ingested_at",
    }
    with pq.ParquetFile(entry["path"]) as pf:
        require(
            set(BASE_FIELDS) <= set(pf.schema_arrow.names),
            "Final canonical fields missing",
        )
        for batch in pf.iter_batches(batch_size=512, columns=sorted(columns)):
            for row in batch.to_pylist():
                label = row["license"]
                if not isinstance(label, str) or not label.strip():
                    issues["missing_license_value"] += 1
                stamp = row["ingested_at"]
                if not isinstance(stamp, datetime) or stamp.utcoffset() is None:
                    issues["invalid_ingested_at"] += 1
                licenses[str(label)] += 1
                sources[str(row["source_id"])]["records"] += 1
                sources[str(row["source_id"])]["tokens"] += row["content_length"]
                identities.append(
                    (row["record_id"], row["content_hash"], row["source_id"], label)
                )
    require(sha(entry["path"]) == entry["sha256"], "File changed during quality check")
    result.update(issues=dict(issues), licenses=dict(licenses), sources=dict(sources))
    return result, identities


def audit(root, workers=4):
    root = root.resolve()
    require(1 <= workers <= 8, "Use 1-8 CPU workers")
    prepared, manifest = verify_prepared(root)
    output = root / "quality"
    output.mkdir(exist_ok=True)
    binding = dict(
        manifest_sha256=prepared["manifest_sha256"],
        code_sha256=sha(Path(__file__)),
        token_auditor_sha256=sha(Path(__file__).with_name("audit_snapshot.py")),
        tiktoken=importlib.metadata.version("tiktoken"),
        tokenizer="cl100k_base",
    )
    with _directory_lock(output), sqlite3.connect(output / "checkpoints.sqlite") as db:
        if (output / "binding.json").exists():
            require(
                read(output / "binding.json") == binding,
                "Quality checkpoint binding changed",
            )
        _write_json(output / "binding.json", binding)
        _write_json(
            output / "summary.json", dict(audit_complete=False, integrity_passed=False)
        )
        # A new attempt must not leave an earlier success marker active.
        (root / "final-corpus-complete.json").unlink(missing_ok=True)
        db.executescript("""
        CREATE TABLE IF NOT EXISTS checked(file INTEGER PRIMARY KEY, result TEXT);
        CREATE TABLE IF NOT EXISTS identities(file INTEGER,row_index INTEGER,record_id TEXT,content_hash TEXT,source TEXT,license TEXT,PRIMARY KEY(file,row_index));
        """)
        pending, results = [], []
        for ordinal, entry in enumerate(manifest["files"]):
            saved = db.execute(
                "SELECT result FROM checked WHERE file=?", (ordinal,)
            ).fetchone()
            if saved:
                result = json.loads(saved[0])
                require(
                    result["sha256"] == entry["sha256"] == sha(entry["path"]),
                    "Cached quality input changed",
                )
                results.append(result)
            else:
                pending.append((ordinal, entry))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for offset in range(0, len(pending), workers * 2):
                group = pending[offset : offset + workers * 2]
                for (ordinal, entry), (result, identities) in zip(
                    group, pool.map(quality_file, [e for _, e in group])
                ):
                    with db:
                        db.executemany(
                            "INSERT INTO identities VALUES (?,?,?,?,?,?)",
                            ((ordinal, i, *row) for i, row in enumerate(identities)),
                        )
                        db.execute(
                            "INSERT INTO checked VALUES (?,?)",
                            (ordinal, json.dumps(result)),
                        )
                    results.append(result)
                    if len(results) % 100 == 0 or len(results) == len(
                        manifest["files"]
                    ):
                        LOG.info(
                            "Final quality %s/%s files",
                            len(results),
                            len(manifest["files"]),
                        )
        issues, totals, licenses, sources = (
            Counter(),
            Counter(),
            Counter(),
            defaultdict(Counter),
        )
        for result in results:
            issues.update(result["issues"])
            totals.update(result["counts"])
            licenses.update(result["licenses"])
            for source, counts in result["sources"].items():
                sources[source].update(counts)
        duplicates = {}
        for column, name in (
            ("record_id", "duplicate_record_ids"),
            ("content_hash", "extra_exact_content_copies"),
        ):
            count = db.execute(
                f"SELECT COALESCE(SUM(n-1),0) FROM (SELECT COUNT(*) n FROM identities GROUP BY {column} HAVING COUNT(*)>1)"
            ).fetchone()[0]
            duplicates[name] = count
            if count:
                issues[name] = count
        metadata_count = 0
        last_file, expected_row, cursor = -1, 0, None
        with gzip.open(root / "record-metadata.jsonl.gz", "rt") as handle:
            for line in handle:
                note = json.loads(line)
                ordinal = note["file"]
                if ordinal != last_file:
                    require(
                        ordinal > last_file
                        and (cursor is None or cursor.fetchone() is None),
                        "Metadata file sequence differs",
                    )
                    cursor = db.execute(
                        "SELECT row_index,record_id,content_hash,source,license FROM identities WHERE file=? ORDER BY row_index",
                        (ordinal,),
                    )
                    last_file, expected_row = ordinal, 0
                expected = (
                    expected_row,
                    note["record_id"],
                    note["content_hash"],
                    note["source"],
                    note["license"],
                )
                require(
                    note["output_row_index"] == expected_row
                    and cursor.fetchone() == expected,
                    "Final metadata/Parquet mismatch",
                )
                expected_row += 1
                metadata_count += 1
        require(
            cursor is None or cursor.fetchone() is None, "Missing trailing metadata"
        )
        require(metadata_count == totals["records"], "Final metadata coverage differs")
        for key in ("records", "tokens"):
            if totals[key] != manifest["totals"][key]:
                issues[f"manifest_{key}_mismatch"] += 1
        require(
            verify_prepared(root)[0] == prepared,
            "Prepared evidence changed during quality check",
        )
        report = dict(
            **binding,
            audit_complete=True,
            integrity_passed=not issues,
            issues=dict(issues),
            duplicates=duplicates,
            totals=dict(totals),
            payload_bytes=sum(r["bytes"] for r in results),
            licenses=dict(licenses),
            by_source=dict(sources),
            files_checked=len(results),
            metadata_records_checked=metadata_count,
            token_counts_recomputed=True,
            canonical_fields_checked=list(BASE_FIELDS),
            retained_text_and_quality_columns_unchanged=prepared[
                "retained_text_and_quality_columns_unchanged"
            ],
            semantic_quality_reassessed=False,
            metadata_fully_resolved=False,
            release_ready=False,
            scope="Final structural, duplicate, metadata, hash and token audit. Existing semantic selection is preserved; unresolved license evidence remains explicit.",
        )
        _write_json(output / "summary.json", report)
        if not issues:
            _write_json(
                root / "final-corpus-complete.json",
                dict(
                    complete=True,
                    integrity_passed=True,
                    release_ready=False,
                    manifest_sha256=prepared["manifest_sha256"],
                    prepared_sha256=sha(root / "final-prepared.json"),
                    quality_summary_sha256=sha(output / "summary.json"),
                    totals=manifest["totals"],
                    metadata_fully_resolved=False,
                ),
            )
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare")
    for name in ("canonical", "plan", "output"):
        prepare_parser.add_argument("--" + name, type=Path, required=True)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--corpus", type=Path, required=True)
    audit_parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.action == "prepare":
        result = prepare(args.canonical, args.plan, args.output)
    else:
        result = audit(args.corpus, args.workers)
    print(json.dumps(result, indent=2))
    return 0 if result.get("prepared") or result.get("integrity_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
