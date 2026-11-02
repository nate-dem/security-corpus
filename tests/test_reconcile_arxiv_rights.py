import json
import sqlite3

import pytest

from scripts.release.reconcile_arxiv_rights import canonical_license, reconcile, run


def observation(url):
    return {"license_url": url, "canonical_license": canonical_license(url)}


def test_aliases_preserve_license_versions_and_do_not_invent_cc0():
    assert canonical_license("http://creativecommons.org/licenses/by/3.0/") == "CC-BY-3.0"
    assert canonical_license("https://creativecommons.org/licenses/by/4.0/") == "CC-BY-4.0"
    assert canonical_license("Public Domain") == "UNSPECIFIED_PUBLIC_DOMAIN"
    assert canonical_license("http://creativecommons.org/licenses/publicdomain/") != "CC0-1.0"
    assert canonical_license("https://other.example/licenses/by/4.0/").startswith("UNRECOGNIZED:")


def test_no_explicit_license_is_not_a_default_grant():
    assert reconcile("arXiv Perpetual Non-Exclusive License", [observation(None)]) == (
        "missing_explicit_raw_license", None
    )
    assert reconcile("CC-BY-4.0", []) == ("missing_raw_metadata", None)


def test_conflicting_observations_do_not_select_the_favorable_one():
    assert reconcile("CC-BY-4.0", [
        observation("https://creativecommons.org/licenses/by/4.0/"),
        observation("https://arxiv.org/licenses/nonexclusive-distrib/1.0/"),
    ]) == ("multiple_observed_licenses", None)
    assert reconcile("CC-BY-3.0", [
        observation("https://creativecommons.org/licenses/by/4.0/")
    ]) == ("stored_raw_license_conflict", "CC-BY-4.0")


def test_cc0_requires_explicit_raw_evidence():
    assert reconcile("Public Domain", [
        observation("http://creativecommons.org/publicdomain/zero/1.0/")
    ]) == ("public_domain_label_resolved_to_cc0", "CC0-1.0")


def test_end_to_end_keeps_physical_row_identity_and_no_approval(tmp_path):
    ledger = tmp_path / "ledger.sqlite"
    with sqlite3.connect(ledger) as con:
        con.execute("CREATE TABLE records(input_path,row_index,source,record_id,"
                    "content_hash,tokens,original_license,source_url)")
        con.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?,?)", [
            ("a.parquet", n, "arxiv", "arxiv:2401.00001", str(n), 100,
             "Public Domain", "https://arxiv.org/abs/2401.00001") for n in (0, 1)
        ])
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    source = metadata / "2401.jsonl"
    source.write_text(json.dumps({"record": {
        "header": {"datestamp": "2024-01-01"},
        "metadata": {"arXiv": {
            "id": "2401.00001", "license": "http://creativecommons.org/publicdomain/zero/1.0/"
        }},
    }}) + "\n")
    output = tmp_path / "review"
    summary = run(ledger, [metadata], output)
    assert summary["records"] == 2 and summary["tokens"] == 200
    assert not summary["release_ready"]
    rows = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
    assert [r["row_index"] for r in rows] == [0, 1]
    assert all(not r["public_release_approved"] for r in rows)
    assert all(not r["paper_version_binding_established"] for r in rows)
    assert rows[0]["original_license"] == "Public Domain"
    assert rows[0]["observations"][0]["line_number"] == 1
    assert len(summary["metadata_files"][str(source)]["sha256"]) == 64
    with pytest.raises(FileExistsError):
        run(ledger, [metadata], output)
