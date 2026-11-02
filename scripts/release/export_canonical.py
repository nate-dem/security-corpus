"""Restore canonical fields after the frozen CPU snapshot and license corrections.

This export changes metadata only. It never selects records or approves release.
Prepared text/source aliases become content/source_id; every other input column
is retained. Physical file/row identities, not possibly duplicated record IDs,
bind source metadata. Run with python -m scripts.release.export_canonical.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import zip_longest
import json
import logging
from pathlib import Path
import shutil
import sqlite3
from urllib.parse import quote

import pyarrow as pa
import pyarrow.parquet as pq

from ingest.connectors.base import NormalizedData
from scripts.release.apply_upstream_license_labels import declarations
from scripts.release.correct_licenses import (
    completed_manifest,
    read,
    require,
    sha,
    write,
)
from scripts.release.reconcile_arxiv_rights import canonical_license

LOG = logging.getLogger(__name__)
BASE_FIELDS = tuple(NormalizedData.model_fields)
ALIASES = {"source": "source_id", "text": "content"}
TIMESTAMP = pa.timestamp("us", tz="UTC")


def connect(path):
    con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def arxiv_evidence(review, baseline_ledger):
    """Index modest per-paper metadata; never load corpus text into memory."""
    summary = read(review / "summary.json")
    require(sha(baseline_ledger) == summary["ledger"]["sha256"], "arXiv ledger changed")
    require(
        sha(review / "records.jsonl") == summary["output"]["sha256"],
        "arXiv review changed",
    )
    with connect(baseline_ledger) as con:
        files = {r["path"]: r["sha256"] for r in con.execute("SELECT * FROM files")}
    result = {}
    with (review / "records.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (files[row["input_path"]], row["row_index"])
            require(key not in result, "Duplicate arXiv evidence address")
            result[key] = row
    require(len(result) == summary["records"], "arXiv evidence count differs")
    return result


def license_value(source, old, rule, paper):
    """Return the recorded declaration and its basis, without a permissions gate."""
    if rule:
        return rule["license"], rule["license_scope"]
    if source == "stackoverflow" or source.startswith("stackexchange-"):
        # Legacy selected threads lack the IDs/licenses of their individual
        # answer revisions. A question date cannot resolve the whole thread.
        return "CC-BY-SA (version unknown)", "contribution_versions_unresolved"
    if (
        source == "reddit"
        or source.startswith("reddit-")
        or source == "cloudtrail-flaws"
    ):
        return "NOASSERTION", "content_license_not_established"
    if source == "arxiv":
        require(paper is not None, "Missing physical-row arXiv review")
        status = paper["evidence_status"]
        if status == "public_domain_label_resolved_to_cc0":
            require(
                paper["observed_license_candidate"] == "CC0-1.0",
                "Unexpected arXiv candidate",
            )
            return "CC0-1.0", "paper_metadata_declaration_version_unbound"
        if status in {"stored_raw_license_conflict", "multiple_observed_licenses"}:
            return "NOASSERTION", "conflicting_paper_license_evidence"
        normalized = canonical_license(old)
        if (
            not normalized
            or normalized == "UNSPECIFIED_PUBLIC_DOMAIN"
            or normalized.startswith("UNRECOGNIZED:")
        ):
            return "NOASSERTION", "paper_license_unresolved"
        return normalized, "paper_label_" + status + "_version_unbound"
    if source in {"sigma", "cisa-kev"}:
        expected = "DRL-1.1" if source == "sigma" else "CC0-1.0"
        require(old == expected, "Verified source correction has not been applied")
        return old, "verified_source_correction"
    if source == "nvd":
        return "CVE Terms of Use / NIST public data", "source_terms_declaration"
    terms = {
        "mitre-attack": "MITRE ATT&CK custom terms",
        "mitre-cwe": "MITRE CWE Terms of Use",
        "capec": "MITRE CAPEC Terms of Use",
        "mitre-capec": "MITRE CAPEC Terms of Use",
    }
    if source in terms:
        return terms[source], "source_terms_declaration"
    return (
        (old, "stored_declaration_unverified")
        if old and old.strip()
        else ("NOASSERTION", "missing_license_evidence")
    )


def canonicalize(table, evidence, rules, papers, original_sha, start, exported_at):
    """Normalize one bounded batch and return a parallel metadata audit."""
    prepared = "source" in table.column_names
    require(
        not prepared or "source_id" not in table.column_names,
        "Ambiguous source columns",
    )
    require(
        not prepared or "content" not in table.column_names, "Ambiguous content columns"
    )
    names = [ALIASES.get(c, c) if prepared else c for c in table.column_names]
    after = table.rename_columns(names)
    require(len(evidence) == len(table), "Missing source metadata rows")
    if prepared and not len(table):
        for name in ("source_record_id", "record_id", "license", "ingested_at"):
            if name not in after.column_names:
                after = after.append_column(
                    name,
                    pa.array(
                        [], type=TIMESTAMP if name == "ingested_at" else pa.string()
                    ),
                )
    meta_cols = [
        c
        for c in after.column_names
        if c not in {"content", "raw", "rule_source", "qwen_raw_response"}
    ]
    metadata = after.select(meta_cols).to_pylist()
    additions = defaultdict(list)
    audits = []
    for offset, (row, proof) in enumerate(zip(metadata, evidence)):
        index = start + offset
        source = row["source_id"]
        require(
            proof["row_index"] == index and proof["source"] == source,
            "Metadata physical-row/source mismatch",
        )
        require(
            proof["content_hash"] == row["content_hash"]
            and proof["tokens"] == row["content_length"],
            "Metadata content identity mismatch",
        )
        content = after["content"][offset].as_py()
        require(isinstance(content, str) and content.strip(), "Empty/non-text content")
        require(
            hashlib.sha256(content.encode()).hexdigest() == row["content_hash"],
            "Content hash differs",
        )
        require(
            type(row["content_length"]) is int and row["content_length"] > 0,
            "Invalid token count",
        )
        upstream = json.loads(proof["metadata_json"])
        rule = rules.get(source) if prepared else None
        if prepared:
            require(rule is not None, "Unmapped prepared source")
            require(
                row["dataset_revision"] == rule["dataset_revision"],
                "Dataset revision mismatch",
            )
            require(proof["metadata_file"], "Prepared metadata was not restored")
            for key in ("dataset_revision", "source_shard", "source_row"):
                require(
                    row[key] == proof[key] and row[key] is not None,
                    f"Metadata locator mismatch: {key}",
                )
            locator = f"{row['dataset_revision']}:{quote(row['source_shard'], safe='')}:{row['source_row']}"
            additions["source_record_id"].append(locator)
            additions["record_id"].append(f"{source}:{locator}")
            additions["source_url"].append(proof["source_url"])
            additions["title"].append(upstream.get("title"))
            # These records enter the canonical schema for the first time here.
            # Do not describe this as their original download/processing time.
            additions["ingested_at"].append(exported_at)
        else:
            require(
                all(
                    k in row and row[k] is not None
                    for k in ("source_record_id", "record_id", "ingested_at")
                ),
                "Missing baseline identity/timestamp",
            )
            require(
                row["record_id"] == proof["record_id"], "Baseline record ID differs"
            )
        paper = papers.get((original_sha, index)) if source == "arxiv" else None
        if paper:
            require(
                paper["record_id"] == row["record_id"]
                and paper["content_hash"] == row["content_hash"]
                and paper["original_license"] == row.get("license"),
                "arXiv record binding differs",
            )
        new, basis = license_value(source, row.get("license"), rule, paper)
        require(isinstance(new, str) and new.strip(), "Empty license value")
        additions["license"].append(new)
        audits.append(
            dict(
                row_index=index,
                source=source,
                record_id=additions["record_id"][-1] if prepared else row["record_id"],
                content_hash=row["content_hash"],
                original_license=proof["original_license"],
                input_license=row.get("license"),
                license=new,
                license_basis=basis,
                upstream_metadata=upstream,
                arxiv_review_address=[original_sha, index] if paper else None,
                ingested_at_basis="canonical_export_time_original_ingestion_unknown"
                if prepared
                else "preserved_original",
            )
        )
    for name, values in additions.items():
        column = pa.array(
            values, type=TIMESTAMP if name == "ingested_at" else pa.string()
        )
        field_index = after.schema.get_field_index(name)
        after = (
            after.set_column(field_index, name, column)
            if field_index >= 0
            else after.append_column(name, column)
        )
    for name in BASE_FIELDS:
        if name not in after.column_names:
            require(
                name in {"raw", "published_at", "title", "source_url"},
                f"Missing canonical field: {name}",
            )
            dtype = (
                TIMESTAMP
                if name == "published_at"
                else pa.null()
                if name == "raw"
                else pa.string()
            )
            after = after.append_column(name, pa.nulls(len(after), type=dtype))
    after = after.select(
        list(BASE_FIELDS) + [c for c in after.column_names if c not in BASE_FIELDS]
    )
    assert_preserved(table, after)
    return after, audits


def assert_preserved(before, after):
    for name in before.column_names:
        if name == "license":
            continue
        target = ALIASES.get(name, name) if "source" in before.column_names else name
        require(
            before[name].equals(after[target]),
            f"Non-license input column changed: {name}",
        )


def export(
    input_dir, snapshot, declarations_path, arxiv_review, baseline_ledger, output
):
    input_dir, snapshot, output = (p.resolve() for p in (input_dir, snapshot, output))
    require(not output.exists(), "Output must be a new directory")
    original, original_digest = completed_manifest(snapshot)
    marker = read(input_dir / "corrections-complete.json")
    input_manifest = input_dir / "corpus-manifest.json"
    require(
        marker["complete"]
        and marker["original_snapshot_manifest_sha256"] == original_digest,
        "Correction snapshot differs",
    )
    require(
        sha(input_manifest) == marker["manifest_sha256"]
        and sha(input_dir / "license-changes.jsonl")
        == marker["license_changes_sha256"],
        "Correction manifest/audit changed",
    )
    require(
        sha(input_dir / "correction-plan.json") == marker["correction_plan_sha256"],
        "Correction plan changed",
    )
    for name, digest in read(input_dir / "correction-plan.json")[
        "evidence_files"
    ].items():
        evidence_path = (input_dir / name).resolve()
        require(
            evidence_path.is_relative_to(input_dir) and sha(evidence_path) == digest,
            "Correction evidence changed",
        )
    manifest = read(input_manifest)
    rules = declarations(declarations_path)
    papers = arxiv_evidence(arxiv_review, baseline_ledger)
    originals = {f["path"]: f for f in original["files"]}
    require(len(originals) == len(original["files"]), "Duplicate snapshot paths")
    require(
        len(manifest["files"]) == len(originals)
        and {e["original_remote_path"] for e in manifest["files"]} == set(originals),
        "Corrected file coverage differs",
    )
    required_bytes = 0
    for entry in manifest["files"]:
        path = Path(entry["path"])
        require(
            path.is_absolute() and not path.resolve().is_relative_to(output),
            "Unsafe input path",
        )
        require(sha(path) == entry["sha256"], f"Payload changed: {path}")
        old = originals[entry["original_remote_path"]]
        require(
            entry.get("original_sha256", entry["sha256"]) == old["sha256"],
            "Correction lineage mismatch",
        )
        require(
            entry["records"] == old["records"] and entry["tokens"] == old["tokens"],
            "Correction counts differ",
        )
        required_bytes += path.stat().st_size
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    # Working-space allowance, not a corpus quality threshold.
    require(
        shutil.disk_usage(ancestor).free > int(required_bytes * 1.5) + 2 * 1024**3,
        "Insufficient disk headroom for separate canonical copy",
    )
    output.mkdir(parents=True)
    (output / "records").mkdir()
    evidence_dir = output / "evidence"
    evidence_dir.mkdir()
    for name, src in {
        "upstream-sources.json": declarations_path,
        "arxiv-summary.json": arxiv_review / "summary.json",
        "arxiv-records.jsonl": arxiv_review / "records.jsonl",
        "prior-license-changes.jsonl": input_dir / "license-changes.jsonl",
        "correction-plan.json": input_dir / "correction-plan.json",
        "cpu-checks-complete.json": snapshot / "checks-complete.json",
        "cpu-integrity-summary.json": snapshot / "integrity/summary.json",
    }.items():
        shutil.copyfile(src, evidence_dir / name)
    shutil.copytree(input_dir / "evidence", evidence_dir / "corrections")
    for rule in rules.values():
        shutil.copyfile(
            declarations_path.parent / rule["card_file"],
            evidence_dir / rule["card_file"],
        )
    by_source = defaultdict(Counter)
    missing_optional = defaultdict(Counter)
    files, duplicates = [], []
    stamp = datetime.now(timezone.utc)
    ledger = snapshot / "rights/records.sqlite"
    ledger_digest = sha(ledger)
    ids = sqlite3.connect(output / "identity-audit.sqlite")
    ids.execute(
        "CREATE TABLE identities(source TEXT,record_id TEXT,first_file INTEGER,first_row INTEGER,PRIMARY KEY(source,record_id))"
    )
    with (
        connect(ledger) as con,
        gzip.open(output / "record-metadata.jsonl.gz", "wt", encoding="utf-8") as audit,
    ):
        bound_files = {r["path"]: dict(r) for r in con.execute("SELECT * FROM files")}
        require(
            set(bound_files) == set(originals), "Rights ledger file coverage differs"
        )
        write(
            evidence_dir / "metadata-file-bindings.json",
            [dict(r) for r in con.execute("SELECT * FROM metadata_files")],
        )
        write(
            evidence_dir / "source-terms.json",
            [dict(r) for r in con.execute("SELECT * FROM source_terms")],
        )
        for ordinal, entry in enumerate(manifest["files"]):
            remote = entry["original_remote_path"]
            old = originals[remote]
            require(
                bound_files[remote]["sha256"] == old["sha256"],
                "Rights ledger file binding differs",
            )
            cursor = con.execute(
                "SELECT * FROM records WHERE input_path=? ORDER BY row_index", (remote,)
            )
            path = Path(entry["path"])
            require(sha(path) == entry["sha256"], "Input changed before export")
            target = output / "records" / f"{ordinal:06d}.parquet"
            rows, tokens = 0, 0
            writer = None
            try:
                with pq.ParquetFile(path) as parquet:
                    if parquet.metadata.num_rows == 0:
                        empty, _ = canonicalize(
                            parquet.read(), [], rules, papers, old["sha256"], 0, stamp
                        )
                        pq.write_table(empty, target, compression="zstd")
                    for batch in parquet.iter_batches(batch_size=512):
                        before = pa.Table.from_batches([batch])
                        proof = [dict(r) for r in cursor.fetchmany(len(before))]
                        after, notes = canonicalize(
                            before, proof, rules, papers, old["sha256"], rows, stamp
                        )
                        if writer is None:
                            writer = pq.ParquetWriter(
                                target, after.schema, compression="zstd"
                            )
                        writer.write_table(after)
                        for offset, note in enumerate(notes):
                            source = note["source"]
                            by_source[source]["records"] += 1
                            by_source[source]["tokens"] += before["content_length"][
                                offset
                            ].as_py()
                            by_source[source]["license=" + note["license"]] += 1
                            by_source[source]["basis=" + note["license_basis"]] += 1
                            for optional in ("source_url", "title", "published_at"):
                                missing_optional[source][optional] += int(
                                    after[optional][offset].as_py() is None
                                )
                            try:
                                ids.execute(
                                    "INSERT INTO identities VALUES (?,?,?,?)",
                                    (source, note["record_id"], ordinal, rows + offset),
                                )
                            except sqlite3.IntegrityError:
                                duplicates.append(
                                    dict(
                                        source=source,
                                        record_id=note["record_id"],
                                        file=ordinal,
                                        row=rows + offset,
                                    )
                                )
                            audit.write(
                                json.dumps(dict(note, file=ordinal), ensure_ascii=False)
                                + "\n"
                            )
                        rows += len(after)
                        tokens += sum(after["content_length"].to_pylist())
            finally:
                if writer:
                    writer.close()
            require(cursor.fetchone() is None, "Extra source metadata rows")
            require(
                rows == entry["records"] and tokens == entry["tokens"],
                "Output counts differ",
            )
            with (
                pq.ParquetFile(path) as before_file,
                pq.ParquetFile(target) as after_file,
            ):
                recheck = con.execute(
                    "SELECT * FROM records WHERE input_path=? ORDER BY row_index",
                    (remote,),
                )
                verified_rows = 0
                require(
                    set(BASE_FIELDS) <= set(after_file.schema_arrow.names),
                    "Roundtrip canonical fields missing",
                )
                for before_batch, after_batch in zip_longest(
                    before_file.iter_batches(batch_size=512),
                    after_file.iter_batches(batch_size=512),
                ):
                    require(
                        before_batch is not None and after_batch is not None,
                        "Roundtrip row count differs",
                    )
                    checked = pa.Table.from_batches([after_batch])
                    before = pa.Table.from_batches([before_batch])
                    expected, _ = canonicalize(
                        before,
                        [dict(r) for r in recheck.fetchmany(len(before))],
                        rules,
                        papers,
                        old["sha256"],
                        verified_rows,
                        stamp,
                    )
                    require(
                        expected.equals(checked), "Roundtrip canonical values differ"
                    )
                    verified_rows += len(before)
            require(sha(path) == entry["sha256"], "Input changed during export")
            files.append(
                dict(
                    path=str(target),
                    relative_path=str(target.relative_to(output)),
                    sha256=sha(target),
                    bytes=target.stat().st_size,
                    records=rows,
                    tokens=tokens,
                    input_path=str(path),
                    input_sha256=entry["sha256"],
                    original_remote_path=remote,
                    original_sha256=old["sha256"],
                )
            )
            ids.commit()
            if ordinal % 100 == 0:
                LOG.info(
                    "Canonical export %s/%s files", ordinal + 1, len(manifest["files"])
                )
    ids.close()
    require(
        sha(ledger) == ledger_digest
        and sha(input_manifest) == marker["manifest_sha256"],
        "Input evidence changed during export",
    )
    require(completed_manifest(snapshot)[1] == original_digest, "CPU snapshot changed")
    totals = dict(
        records=sum(f["records"] for f in files),
        tokens=sum(f["tokens"] for f in files),
        bytes=sum(f["bytes"] for f in files),
    )
    for key in ("records", "tokens"):
        require(
            totals[key] == sum(f[key] for f in original["files"]),
            "Corpus totals changed",
        )
    write(
        output / "schema.json",
        dict(
            canonical_fields=BASE_FIELDS,
            aliases=ALIASES,
            source_specific_columns="Preserved; schemas may include additional family/quality fields and source-specific raw structs.",
            license="Recorded declaration or explicit unresolved marker; not redistribution approval.",
            prepared_ingested_at="Time first normalized into this canonical export; original ingestion time was not retained.",
            prepared_record_ids="source_id:dataset_revision:URL-escaped-source_shard:source_row",
            optional_metadata="Unknown title, source_url, published_at and raw remain null; never fabricated.",
        ),
    )
    write(
        output / "audit.json",
        dict(
            totals=totals,
            by_source=dict(by_source),
            missing_optional=dict(missing_optional),
            duplicate_record_ids=duplicates,
            all_records_have_canonical_fields=True,
            all_records_have_nonempty_license=True,
            input_text_and_quality_columns_unchanged=True,
            missing_license_values=0,
            semantic_quality_reassessed=False,
            permission_review_performed=False,
            metadata_fully_resolved=False,
        ),
    )
    write(
        output / "corpus-manifest.json",
        dict(
            files=files,
            totals=totals,
            original_manifest_sha256=original_digest,
            corrected_manifest_sha256=marker["manifest_sha256"],
            scope=original.get("scope"),
            record_metadata="record-metadata.jsonl.gz",
            release_ready=False,
        ),
    )
    # Completion only after every file and report has been written successfully.
    evidence_hashes = {
        str(p.relative_to(output)): sha(p)
        for p in evidence_dir.rglob("*")
        if p.is_file()
    }
    result = dict(
        export_complete=True,
        release_ready=False,
        metadata_fully_resolved=False,
        records=totals["records"],
        tokens=totals["tokens"],
        duplicate_record_ids=len(duplicates),
        manifest_sha256=sha(output / "corpus-manifest.json"),
        audit_sha256=sha(output / "audit.json"),
        schema_sha256=sha(output / "schema.json"),
        record_metadata_sha256=sha(output / "record-metadata.jsonl.gz"),
        evidence_sha256=evidence_hashes,
        code_sha256={
            str(p.relative_to(Path(__file__).resolve().parents[2])): sha(p)
            for p in (
                Path(__file__).resolve(),
                Path(__file__).with_name("apply_upstream_license_labels.py").resolve(),
                Path(__file__).with_name("correct_licenses.py").resolve(),
                Path(__file__).with_name("reconcile_arxiv_rights.py").resolve(),
            )
        },
    )
    write(output / "canonical-export-complete.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "input",
        "snapshot",
        "declarations",
        "arxiv-review",
        "baseline-ledger",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    print(
        json.dumps(
            export(
                args.input,
                args.snapshot,
                args.declarations,
                args.arxiv_review,
                args.baseline_ledger,
                args.output,
            ),
            indent=2,
        )
    )
