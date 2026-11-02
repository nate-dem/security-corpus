from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.release import export_canonical as ex


DECLARATIONS = (
    Path(__file__).resolve().parents[1]
    / "tests/fixtures/upstream_licenses/sources.json"
)
STAMP = datetime(2026, 10, 21, tzinfo=timezone.utc)


def baseline(source="stackoverflow", license_="CC-BY-SA-4.0"):
    text = "Preserve this security document and code: `x = 1`."
    return dict(
        source_id=source,
        source_record_id="123",
        record_id=source + ":123",
        content=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        content_length=13,
        title="Title",
        ingested_at=STAMP,
        published_at=STAMP,
        source_url="https://example.com/123",
        license=license_,
        raw=None,
        score=17,
        qwen_quality=4.5,
    )


def prepared(source="youtube-commons"):
    rule = ex.declarations(DECLARATIONS)[source]
    row = baseline()
    return dict(
        source=source,
        text=row["content"],
        content_hash=row["content_hash"],
        content_length=row["content_length"],
        dataset_revision=rule["dataset_revision"],
        source_shard="data/a:1.parquet",
        source_row=15,
        source_copies=2,
        kind="video",
    )


def proof(row, index=0):
    prepared_ = "source" in row
    return dict(
        row_index=index,
        source=row.get("source", row.get("source_id")),
        record_id=row.get("record_id"),
        content_hash=row["content_hash"],
        tokens=row["content_length"],
        original_license=row.get("license"),
        metadata_json=json.dumps(
            dict(
                title="Upstream title",
                channel="Creator",
                license="original upstream value",
            )
        ),
        metadata_file="/metadata/part.parquet" if prepared_ else None,
        source_url="https://example.com/original",
        dataset_revision=row.get("dataset_revision"),
        source_shard=row.get("source_shard"),
        source_row=row.get("source_row"),
    )


@pytest.mark.parametrize("source", ["youtube-commons", "primus-fineweb", "redsage-cfw"])
def test_prepared_canonical_fields_license_and_content_preservation(source):
    row = prepared(source)
    before = pa.Table.from_pylist([row])
    rules = ex.declarations(DECLARATIONS)
    after, audit = ex.canonicalize(before, [proof(row)], rules, {}, "digest", 0, STAMP)
    assert set(ex.BASE_FIELDS) <= set(after.column_names)
    assert after["license"].to_pylist() == [rules[source]["license"]]
    assert after["content"].to_pylist() == [row["text"]]
    assert after["source_url"].to_pylist() == ["https://example.com/original"]
    assert after["title"].to_pylist() == ["Upstream title"]
    assert after["record_id"][0].as_py().endswith(":data%2Fa%3A1.parquet:15")
    assert audit[0]["upstream_metadata"]["channel"] == "Creator"
    assert (
        audit[0]["ingested_at_basis"]
        == "canonical_export_time_original_ingestion_unknown"
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"row_index": 4}, "physical-row"),
        ({"source_row": 9}, "locator"),
        ({"content_hash": "bad"}, "identity"),
        ({"metadata_file": None}, "not restored"),
    ],
)
def test_bad_metadata_binding_rejected(change, match):
    row = prepared()
    with pytest.raises(ValueError, match=match):
        ex.canonicalize(
            pa.Table.from_pylist([row]),
            [dict(proof(row), **change)],
            ex.declarations(DECLARATIONS),
            {},
            "digest",
            0,
            STAMP,
        )


def test_revision_mismatch_rejected():
    row = dict(prepared(), dataset_revision="changed")
    with pytest.raises(ValueError, match="revision"):
        ex.canonicalize(
            pa.Table.from_pylist([row]),
            [proof(row)],
            ex.declarations(DECLARATIONS),
            {},
            "digest",
            0,
            STAMP,
        )


def test_qa_does_not_infer_thread_license_from_question_date():
    row = baseline()
    before = pa.Table.from_pylist([row])
    after, audit = ex.canonicalize(before, [proof(row)], {}, {}, "digest", 0, STAMP)
    assert after["license"].to_pylist() == ["CC-BY-SA (version unknown)"]
    assert audit[0]["original_license"] == "CC-BY-SA-4.0"
    assert after["qwen_quality"].equals(before["qwen_quality"])
    assert after["ingested_at"].equals(before["ingested_at"])


def test_arxiv_keeps_per_paper_license_and_reports_version_limit():
    paper = dict(
        evidence_status="license_label_matches",
        observed_license_candidate="CC-BY-NC-SA-3.0",
    )
    assert ex.license_value(
        "arxiv", "http://creativecommons.org/licenses/by-nc-sa/3.0/", None, paper
    ) == ("CC-BY-NC-SA-3.0", "paper_label_license_label_matches_version_unbound")
    paper = dict(
        evidence_status="public_domain_label_resolved_to_cc0",
        observed_license_candidate="CC0-1.0",
    )
    assert ex.license_value("arxiv", "Public Domain", None, paper) == (
        "CC0-1.0",
        "paper_metadata_declaration_version_unbound",
    )
    paper = dict(
        evidence_status="missing_raw_metadata", observed_license_candidate=None
    )
    assert ex.license_value("arxiv", "Public Domain", None, paper)[0] == "NOASSERTION"
    assert (
        ex.license_value("reddit-netsec", "Reddit Terms of Service", None, None)[0]
        == "NOASSERTION"
    )


def fixture_export(tmp_path):
    source = tmp_path / "input"
    snapshot = tmp_path / "snapshot"
    review = tmp_path / "arxiv"
    for directory in (
        source,
        source / "evidence",
        snapshot,
        snapshot / "rights",
        snapshot / "integrity",
        review,
    ):
        directory.mkdir()
    rows = [
        baseline(),
        prepared(),
        baseline("arxiv", "CC-BY-4.0"),
        baseline("mitre-attack", "MITRE Terms of Use"),
        baseline("mitre-attack", "MITRE Terms of Use"),
    ]
    tables = [pa.Table.from_pylist([row]) for row in rows]
    tables.append(tables[1].slice(0, 0))
    entries, remote = [], []
    ledger = snapshot / "rights/records.sqlite"
    with sqlite3.connect(ledger) as con:
        con.executescript("""
        CREATE TABLE files(path TEXT PRIMARY KEY,sha256 TEXT,records INTEGER,bytes INTEGER);
        CREATE TABLE records(input_path TEXT,row_index INTEGER,source TEXT,record_id TEXT,content_hash TEXT,tokens INTEGER,original_license TEXT,metadata_json TEXT,metadata_file TEXT,source_url TEXT,dataset_revision TEXT,source_shard TEXT,source_row INTEGER,PRIMARY KEY(input_path,row_index));
        CREATE TABLE metadata_files(path,sha256);
        CREATE TABLE source_terms(source_family,evidence_json);
        """)
        for i, table in enumerate(tables):
            path = source / f"{i}.parquet"
            pq.write_table(table, path)
            old = dict(
                path=f"/remote/{i}.parquet",
                sha256=ex.sha(path),
                records=len(table),
                tokens=sum(table["content_length"].to_pylist()),
            )
            remote.append(old)
            entries.append(dict(old, path=str(path), original_remote_path=old["path"]))
            con.execute(
                "INSERT INTO files VALUES (?,?,?,?)",
                (old["path"], old["sha256"], len(table), path.stat().st_size),
            )
            if i < len(rows):
                p = dict(input_path=old["path"], **proof(rows[i]))
                con.execute(
                    "INSERT INTO records ("
                    + ",".join(p)
                    + ") VALUES ("
                    + ",".join("?" for _ in p)
                    + ")",
                    tuple(p.values()),
                )
    arxiv = dict(
        input_path="/remote/2.parquet",
        row_index=0,
        record_id=rows[2]["record_id"],
        content_hash=rows[2]["content_hash"],
        original_license=rows[2]["license"],
        evidence_status="license_label_matches",
        observed_license_candidate="CC-BY-4.0",
    )
    (review / "records.jsonl").write_text(json.dumps(arxiv) + "\n")
    ex.write(
        review / "summary.json",
        dict(
            ledger=dict(sha256=ex.sha(ledger)),
            output=dict(sha256=ex.sha(review / "records.jsonl")),
            records=1,
        ),
    )
    ex.write(snapshot / "corpus-manifest.json", dict(files=remote))
    digest = ex.sha(snapshot / "corpus-manifest.json")
    ex.write(
        snapshot / "summary.json", dict(complete=True, corpus_manifest_sha256=digest)
    )
    for name in ("rights", "integrity"):
        ex.write(
            snapshot / name / "summary.json",
            dict(
                audit_complete=True,
                manifest_sha256=digest,
                ledger_sha256=ex.sha(ledger),
            ),
        )
    ex.write(
        snapshot / "checks-complete.json",
        dict(
            cpu_stages_complete=True,
            summary_sha256=ex.sha(snapshot / "summary.json"),
            rights_summary_sha256=ex.sha(snapshot / "rights/summary.json"),
            integrity_summary_sha256=ex.sha(snapshot / "integrity/summary.json"),
        ),
    )
    ex.write(source / "corpus-manifest.json", dict(files=entries))
    (source / "license-changes.jsonl").write_text("")
    ex.write(source / "correction-plan.json", {"evidence_files": {}})
    ex.write(
        source / "corrections-complete.json",
        dict(
            complete=True,
            original_snapshot_manifest_sha256=digest,
            manifest_sha256=ex.sha(source / "corpus-manifest.json"),
            license_changes_sha256=ex.sha(source / "license-changes.jsonl"),
            correction_plan_sha256=ex.sha(source / "correction-plan.json"),
        ),
    )
    return source, snapshot, review, ledger


def test_export_including_empty_shard_duplicates_and_originals(tmp_path):
    source, snapshot, review, ledger = fixture_export(tmp_path)
    before = {p: ex.sha(p) for p in source.glob("*.parquet")}
    output = tmp_path / "export"
    result = ex.export(source, snapshot, DECLARATIONS, review, ledger, output)
    assert result["export_complete"] and not result["release_ready"]
    assert result["duplicate_record_ids"] == 1
    manifest = ex.read(output / "corpus-manifest.json")
    assert manifest["totals"]["records"] == 5
    assert len(manifest["files"]) == 6
    for entry in manifest["files"]:
        table = pq.ParquetFile(entry["path"]).read()
        assert set(ex.BASE_FIELDS) <= set(table.column_names)
        assert all(table["license"].to_pylist())
    with gzip.open(output / "record-metadata.jsonl.gz", "rt") as handle:
        notes = [json.loads(line) for line in handle]
    assert len(notes) == 5
    assert before == {p: ex.sha(p) for p in before}
    with pytest.raises(ValueError, match="new directory"):
        ex.export(source, snapshot, DECLARATIONS, review, ledger, output)


def test_incomplete_snapshot_cannot_produce_completion_marker(tmp_path):
    source, snapshot, review, ledger = fixture_export(tmp_path)
    ex.write(snapshot / "checks-complete.json", dict(cpu_stages_complete=False))
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="incomplete"):
        ex.export(source, snapshot, DECLARATIONS, review, ledger, output)
    assert not output.exists()


def test_changed_payload_rejected_before_writing(tmp_path):
    source, snapshot, review, ledger = fixture_export(tmp_path)
    (source / "0.parquet").write_bytes(b"changed")
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="Payload changed"):
        ex.export(source, snapshot, DECLARATIONS, review, ledger, output)
    assert not output.exists()
