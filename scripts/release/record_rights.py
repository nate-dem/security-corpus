"""Create a checksum-bound, per-record rights evidence ledger, without relicensing.

This is an engineering inventory, not a legal clearance engine. Every physical
row is covered, including duplicate record IDs. No corpus records are rewritten,
deleted, or approved for publication. SQLite keeps the scratch inode cost low.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3

import pyarrow.parquet as pq

from scripts.youtube.download import _directory_lock, _write_json
from scripts.youtube.profile import _sha256

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "rights_sources.json"
LOG = logging.getLogger(__name__)
FIELDS = {
    "source_id",
    "source",
    "record_id",
    "content_hash",
    "content_length",
    "license",
    "source_url",
    "title",
    "authors",
    "author",
    "rule_source",
    "dataset_revision",
    "source_shard",
    "source_row",
    "video_id",
    "video_link",
    "channel_id",
    "channel",
    "upstream_license",
    "url",
}


def read(path):
    return json.loads(Path(path).read_text())


def family(source):
    if source == "stackoverflow" or source.startswith("stackexchange-"):
        return "stackexchange"
    if source == "reddit" or source.startswith("reddit-"):
        return "reddit"
    return "capec" if source == "mitre-capec" else source


def assess(source, row):
    """Record evidence gaps; never convert a collection license into text rights."""
    kind = family(source)
    declared = row.get("license") or row.get("upstream_license")
    if kind in {"primus-fineweb", "redsage-cfw"}:
        return "page_permission_review" if declared else "missing_page_license"
    if kind == "youtube-commons":
        return (
            "video_license_version_and_attribution_review"
            if declared
            else "missing_video_license"
        )
    if kind == "stackexchange":
        return "missing_contribution_licenses_and_attribution"
    if kind in {"reddit", "cloudtrail-flaws"}:
        return "redistribution_basis_not_established"
    if kind == "arxiv":
        if not declared or declared == "arXiv Perpetual Non-Exclusive License":
            return "redistribution_basis_not_established"
        if "NC" in declared or "ND" in declared or "by-nc" in declared.lower():
            return "restricted_license_and_transformation_review"
        if "publicdomain" in declared.lower() or declared == "Public Domain":
            return "ambiguous_public_domain_evidence"
        return "paper_version_attribution_and_license_review"
    if kind in {"nvd", "cisa-kev", "mitre-attack", "mitre-cwe", "capec", "sigma"}:
        return "source_version_notice_and_label_review"
    return "unknown_source_or_license"


def rows(path):
    with pq.ParquetFile(path) as file:
        columns = sorted(FIELDS.intersection(file.schema_arrow.names))
        for batch in file.iter_batches(
            batch_size=512, columns=columns, use_threads=False
        ):
            yield from batch.to_pylist()


def input_files(manifest, project):
    saved = read(manifest)
    if "baseline_files" in saved:
        if project is None:
            raise ValueError("--project is required for a baseline input manifest")
        files = []
        for f in saved["baseline_files"]:
            path = (project / f["project_relative_path"]).resolve()
            if not path.is_relative_to(project.resolve()):
                raise ValueError("Baseline input escapes project")
            files.append(
                {
                    "path": str(path),
                    "sha256": f["sha256"],
                    "records": f["records"],
                    "tokens": f["recomputed_tokens"],
                }
            )
        return files
    return saved["files"]


def load_payload(con, files):
    seen = set()
    for i, entry in enumerate(files):
        path = Path(entry["path"]).resolve()
        if path in seen:
            raise ValueError(f"Repeated input: {path}")
        seen.add(path)
        digest = _sha256(path)
        if digest != entry["sha256"]:
            raise ValueError(f"Payload checksum mismatch: {path}")
        con.execute(
            "INSERT INTO files VALUES (?,?,?,?)",
            (str(path), digest, entry["records"], path.stat().st_size),
        )
        count, tokens = 0, 0
        for row_index, row in enumerate(rows(path)):
            source = row.get("source_id") or row.get("source")
            if (
                not source
                or not row.get("content_hash")
                or not isinstance(row.get("content_length"), int)
            ):
                raise ValueError(f"Missing source/hash/tokens: {path}:{row_index}")
            locator = (
                row.get("dataset_revision"),
                row.get("source_shard"),
                row.get("source_row"),
            )
            if any(v is not None for v in locator) and any(v is None for v in locator):
                raise ValueError(f"Incomplete source locator: {path}:{row_index}")
            declared = row.get("license")
            con.execute(
                "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(path),
                    row_index,
                    source,
                    row.get("record_id"),
                    row["content_hash"],
                    row["content_length"],
                    *locator,
                    declared,
                    declared,
                    row.get("source_url"),
                    json.dumps(row, ensure_ascii=False),
                    None,
                    assess(source, row),
                    0,
                ),
            )
            count += 1
            tokens += row["content_length"]
        if (
            count != entry["records"]
            or tokens != entry["tokens"]
            or _sha256(path) != digest
        ):
            raise ValueError(f"Payload changed or counts disagree: {path}")
        con.commit()
        LOG.info("Rights inventory [%s/%s]: %s records", i + 1, len(files), count)


def metadata_binding(entry, config):
    path = Path(entry["metadata"])
    if entry["source"] == "youtube-commons":
        return path, entry["sha256"]
    summary_path = path.parent.parent / "summary.json"
    if _sha256(summary_path) != config["source_summaries"][entry["source"]]:
        raise ValueError(f"Source profile summary changed: {summary_path}")
    matches = [
        s
        for s in read(summary_path)["shards"]
        if s["source_shard"] == entry["source_shard"]
    ]
    if len(matches) != 1:
        raise ValueError("Metadata shard is not uniquely bound to source profile")
    return path, matches[0]["metadata_sha256"]


def recover_metadata(con, config_path):
    """Join only the selected representative, never a different copy's license."""
    config = read(config_path)
    config_digest = _sha256(config_path)
    # Bind this configuration to the prepared input manifest used by the scorer.
    summary = read(config_path.parent / "summary.json")
    if summary["run_config_sha256"] != config_digest:
        raise ValueError("Prepared metadata configuration changed")
    for entry in config["inputs"]:
        source, revision, shard = (
            entry[k] for k in ("source", "dataset_revision", "source_shard")
        )
        selected = con.execute(
            "SELECT source_row, content_hash FROM records WHERE source=? AND dataset_revision=? AND source_shard=?",
            (source, revision, shard),
        ).fetchall()
        if not selected:
            continue
        expected = {}
        for index, digest in selected:
            if index in expected and expected[index] != digest:
                raise ValueError("Conflicting selected hashes at one source locator")
            expected[index] = digest
        path, digest = metadata_binding(entry, config)
        if _sha256(path) != digest:
            raise ValueError(f"Metadata checksum mismatch: {path}")
        found = set()
        for row in rows(path):
            index = row.get("source_row")
            if index not in expected:
                continue
            if (
                index in found
                or row.get("source_shard") != shard
                or row.get("content_hash") != expected[index]
            ):
                raise ValueError(f"Metadata locator/hash mismatch: {path}:{index}")
            if (
                row.get("dataset_revision", revision) != revision
                or row.get("source", source) != source
            ):
                raise ValueError("Metadata source/revision mismatch")
            found.add(index)
            con.execute(
                "UPDATE records SET evidence_license=?, source_url=?, metadata_json=?, metadata_file=?, status=? WHERE source=? AND dataset_revision=? AND source_shard=? AND source_row=?",
                (
                    row.get("license") or row.get("upstream_license"),
                    row.get("video_link") or row.get("url"),
                    json.dumps(row, ensure_ascii=False),
                    str(path),
                    assess(source, row),
                    source,
                    revision,
                    shard,
                    index,
                ),
            )
        if found != set(expected) or _sha256(path) != digest:
            raise ValueError(f"Metadata coverage changed or incomplete: {path}")
        con.execute(
            "INSERT INTO metadata_files VALUES (?,?,?)",
            (str(path), digest, config_digest),
        )
        con.commit()
        LOG.info(
            "Restored rights metadata: %s %s (%s selected rows)",
            source,
            shard,
            len(found),
        )
    missing = con.execute(
        "SELECT COUNT(*) FROM records WHERE source_shard IS NOT NULL AND metadata_file IS NULL"
    ).fetchone()[0]
    if missing:
        raise ValueError(f"{missing} selected records have no matching source metadata")
    if _sha256(config_path) != config_digest:
        raise ValueError("Prepared configuration changed during metadata recovery")


def execute(manifest, output, project=None, prepared_config=None):
    manifest, output = manifest.resolve(), output.resolve()
    files = input_files(manifest, project)
    for f in files:
        if Path(f["path"]).resolve().is_relative_to(output):
            raise ValueError("Rights output must not contain corpus inputs")
    output.mkdir(parents=True, exist_ok=True)
    with _directory_lock(output):
        _write_json(
            output / "summary.json", {"audit_complete": False, "release_ready": False}
        )
        temporary = output / "records.building.sqlite"
        # This is only our derived incomplete ledger, never source data.
        temporary.unlink(missing_ok=True)
        con = sqlite3.connect(temporary)
        try:
            con.executescript("""
                CREATE TABLE files(path TEXT PRIMARY KEY, sha256 TEXT, records INTEGER, bytes INTEGER);
                CREATE TABLE records(
                    input_path TEXT, row_index INTEGER, source TEXT, record_id TEXT,
                    content_hash TEXT, tokens INTEGER, dataset_revision TEXT,
                    source_shard TEXT, source_row INTEGER, original_license TEXT,
                    evidence_license TEXT, source_url TEXT, metadata_json TEXT,
                    metadata_file TEXT, status TEXT, public_release_approved INTEGER,
                    PRIMARY KEY(input_path,row_index));
                CREATE INDEX locator ON records(source,dataset_revision,source_shard,source_row);
                CREATE TABLE metadata_files(path TEXT PRIMARY KEY, sha256 TEXT, prepared_config_sha256 TEXT);
                CREATE TABLE source_terms(source_family TEXT PRIMARY KEY, evidence_json TEXT);
            """)
            registry = read(REGISTRY)
            for source, evidence in registry["sources"].items():
                con.execute(
                    "INSERT INTO source_terms VALUES (?,?)",
                    (source, json.dumps(evidence)),
                )
            load_payload(con, files)
            if prepared_config:
                recover_metadata(con, prepared_config)
            report = []
            for source, status, license_, count, tokens in con.execute(
                "SELECT source,status,evidence_license,COUNT(*),SUM(tokens) FROM records GROUP BY source,status,evidence_license ORDER BY source,status,evidence_license"
            ):
                report.append(
                    {
                        "source": source,
                        "status": status,
                        "evidence_license": license_,
                        "records": count,
                        "tokens": tokens,
                    }
                )
            totals = Counter()
            for row in report:
                totals.update(records=row["records"], tokens=row["tokens"])
            missing_metadata = con.execute(
                "SELECT COUNT(*) FROM records WHERE source_shard IS NOT NULL AND metadata_file IS NULL"
            ).fetchone()[0]
            duplicate_ids = con.execute(
                "SELECT source,record_id,COUNT(*) FROM records WHERE record_id IS NOT NULL GROUP BY source,record_id HAVING COUNT(*)>1"
            ).fetchall()
            con.commit()
            if con.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise ValueError("Rights ledger integrity check failed")
        finally:
            con.close()
        target = output / "records.sqlite"
        temporary.replace(target)
        _write_json(output / "source-terms.json", registry)
        result = {
            "audit_complete": True,
            "release_ready": False,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "scope": "Every selected physical row inventoried; rights evidence only, not legal clearance, relicensing, or a content-integrity/token recomputation audit.",
            "manifest_sha256": _sha256(manifest),
            "manifest_path": str(manifest),
            "auditor_sha256": _sha256(Path(__file__)),
            "source_registry_sha256": _sha256(REGISTRY),
            "prepared_config_sha256": _sha256(prepared_config)
            if prepared_config
            else None,
            "ledger_sha256": _sha256(target),
            "totals": dict(totals),
            "payload_bytes": sum(Path(f["path"]).stat().st_size for f in files),
            "prepared_records_without_restored_metadata": missing_metadata,
            "duplicate_record_ids": duplicate_ids,
            "groups": report,
            "public_release_approved_records": 0,
            "next": "Resolve recorded rights/attribution gaps and obtain an appropriate publication assessment. Unknown rights are not fair-use licenses. This audit does not change the research corpus.",
        }
        _write_json(output / "summary.json", result)
        LOG.info(
            "Rights ledger complete: %s records, %s tokens; publication review remains",
            totals["records"],
            totals["tokens"],
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--prepared-config", type=Path)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    execute(args.manifest, args.output, args.project, args.prepared_config)


if __name__ == "__main__":
    main()
