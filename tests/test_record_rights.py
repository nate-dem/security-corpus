import json
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.release import record_rights as rights
from scripts.youtube.download import _write_json
from scripts.youtube.profile import _sha256


def fixture(tmp_path, rows):
    data = tmp_path / "input.parquet"
    pq.write_table(pa.Table.from_pylist(rows), data)
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "files": [
                {
                    "path": str(data),
                    "sha256": _sha256(data),
                    "records": len(rows),
                    "tokens": sum(r["content_length"] for r in rows),
                }
            ]
        },
    )
    return manifest, data


def test_every_physical_row_preserved_no_fake_licenses(tmp_path):
    records = [
        {
            "source_id": "stackoverflow",
            "record_id": "same",
            "license": "CC-BY-SA-4.0",
            "content_hash": "a",
            "content_length": 10,
        },
        {
            "source_id": "stackoverflow",
            "record_id": "same",
            "license": "CC-BY-SA-4.0",
            "content_hash": "b",
            "content_length": 11,
        },
        {
            "source_id": "cloudtrail-flaws",
            "record_id": "log",
            "license": "Public Domain (flaws.cloud)",
            "content_hash": "c",
            "content_length": 12,
        },
    ]
    manifest, data = fixture(tmp_path, records)
    before = data.read_bytes()
    output = tmp_path / "rights"
    report = rights.execute(manifest, output)
    assert report["audit_complete"] and not report["release_ready"]
    assert report["totals"] == {"records": 3, "tokens": 33}
    assert report["duplicate_record_ids"] == [("stackoverflow", "same", 2)]
    assert report["public_release_approved_records"] == 0
    assert data.read_bytes() == before
    with sqlite3.connect(output / "records.sqlite") as con:
        assert con.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 3
        assert con.execute(
            "SELECT original_license FROM records ORDER BY row_index"
        ).fetchall() == [(r["license"],) for r in records]


@pytest.mark.parametrize(
    "source,license_,status",
    [
        ("primus-fineweb", None, "missing_page_license"),
        ("redsage-cfw", "ODC-By-1.0", "page_permission_review"),
        (
            "youtube-commons",
            "Creative Commons Attribution license (reuse allowed)",
            "video_license_version_and_attribution_review",
        ),
        (
            "arxiv",
            "arXiv Perpetual Non-Exclusive License",
            "redistribution_basis_not_established",
        ),
        ("arxiv", "CC-BY-NC-ND-4.0", "restricted_license_and_transformation_review"),
        (
            "arxiv",
            "http://creativecommons.org/publicdomain/",
            "ambiguous_public_domain_evidence",
        ),
        ("arxiv", "CC-BY-4.0", "paper_version_attribution_and_license_review"),
    ],
)
def test_rights_classification_does_not_clear_publication(source, license_, status):
    assert rights.assess(source, {"license": license_}) == status


def test_modified_payload_rejected(tmp_path):
    manifest, data = fixture(
        tmp_path, [{"source_id": "arxiv", "content_hash": "a", "content_length": 10}]
    )
    with data.open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        rights.execute(manifest, tmp_path / "rights")


def test_metadata_join_is_source_revision_row_and_hash_bound(tmp_path):
    selected = {
        "source": "youtube-commons",
        "dataset_revision": "pinned",
        "source_shard": "data/shard.parquet",
        "source_row": 2,
        "content_hash": "abc",
        "content_length": 10,
    }
    manifest, _ = fixture(tmp_path, [selected])
    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    **selected,
                    "license": "Creative Commons Attribution license (reuse allowed)",
                    "video_link": "https://www.youtube.com/watch?v=example",
                    "channel": "Example creator",
                }
            ]
        ),
        source,
    )
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    config = prepared / "run-config.json"
    _write_json(
        config,
        {"inputs": [{**selected, "metadata": str(source), "sha256": _sha256(source)}]},
    )
    _write_json(prepared / "summary.json", {"run_config_sha256": _sha256(config)})
    report = rights.execute(manifest, tmp_path / "rights", prepared_config=config)
    assert report["prepared_records_without_restored_metadata"] == 0
    assert (
        report["groups"][0]["status"] == "video_license_version_and_attribution_review"
    )
    with sqlite3.connect(tmp_path / "rights/records.sqlite") as con:
        row = con.execute(
            "SELECT original_license,evidence_license,metadata_json FROM records"
        ).fetchone()
        assert row[0] is None and row[1].startswith("Creative Commons")
        assert json.loads(row[2])["channel"] == "Example creator"
    # A different original row cannot supply a more favorable license.
    changed = pq.read_table(source).to_pylist()
    changed[0]["content_hash"] = "different"
    pq.write_table(pa.Table.from_pylist(changed), source)
    value = rights.read(config)
    value["inputs"][0]["sha256"] = _sha256(source)
    _write_json(config, value)
    _write_json(prepared / "summary.json", {"run_config_sha256": _sha256(config)})
    with pytest.raises(ValueError, match="locator/hash mismatch"):
        rights.execute(manifest, tmp_path / "bad-rights", prepared_config=config)
