import hashlib
import json
import shutil
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.release import correct_licenses as fix


@pytest.fixture
def review(tmp_path):
    payload = tmp_path / "payload"
    payload.mkdir()
    rows = []
    files = []
    for source, old, text in (
        ("sigma", "LGPL-2.1", "rule"),
        ("cisa-kev", "CISA Terms of Use", "description"),
    ):
        path = payload / f"{source}.parquet"
        row = {
            "source_id": source,
            "source_record_id": "CVE-2020-1" if source == "cisa-kev" else "rule-1",
            "record_id": f"{source}:1",
            "content": text,
            "content_hash": hashlib.sha256(text.encode()).hexdigest(),
            "content_length": 1,
            "license": old,
            "attribution": "author",
        }
        pq.write_table(pa.Table.from_pylist([row]), path)
        files.append(
            {"path": str(path), "sha256": fix.sha(path), "records": 1, "tokens": 1}
        )
        rows.append((str(path), 0, source, row["record_id"], row["content_hash"], old))
    ledger = tmp_path / "ledger.sqlite"
    with sqlite3.connect(ledger) as con:
        con.execute("CREATE TABLE files(path,sha256,records)")
        con.execute(
            "CREATE TABLE records(input_path,row_index,source,record_id,content_hash,original_license)"
        )
        con.executemany(
            "INSERT INTO files VALUES (?,?,?)",
            [(f["path"], f["sha256"], 1) for f in files],
        )
        con.executemany("INSERT INTO records VALUES (?,?,?,?,?,?)", rows)
    sigma = tmp_path / "sigma"
    sigma.mkdir()
    proof = {
        "input_path": rows[0][0],
        "row_index": 0,
        "record_id": rows[0][3],
        "content_hash": rows[0][4],
        "rule_source_matches": True,
        "content_hash_matches": True,
        "author": "author",
        "observed_source_license": "DRL-1.1",
        "source_revision": "revision",
        "source_url": "source",
        "notice_url": "notice",
        "license_url": "license",
        "raw_file_sha256": "digest",
    }
    (sigma / "records.jsonl").write_text(json.dumps(proof) + "\n")
    (sigma / "source-LICENSE.txt").write_text("DRL-1.1")
    fix.write(
        sigma / "summary.json",
        {
            "ledger_sha256": fix.sha(ledger),
            "records_sha256": fix.sha(sigma / "records.jsonl"),
            "source_notice_sha256": fix.sha(sigma / "source-LICENSE.txt"),
            "source_license": "DRL-1.1",
            "source_revision": "revision",
        },
    )
    cisa = tmp_path / "cisa"
    cisa.mkdir()
    (cisa / "LICENSE").write_text(
        "The KEV database is distributed under the Creative Commons 0 1.0 License."
    )
    fix.write(
        cisa / "known_exploited_vulnerabilities.json",
        {
            "catalogVersion": "2026.04.28",
            "vulnerabilities": [
                {"cveID": "CVE-2020-1", "shortDescription": "description"}
            ],
        },
    )
    fix.write(
        cisa / "sources.json",
        {
            "revision": "revision",
            "files": [
                {
                    "name": name,
                    "sha256": fix.sha(cisa / name),
                    "url": f"https://raw.githubusercontent.com/cisagov/kev-data/revision/{name}",
                }
                for name in ("LICENSE", "known_exploited_vulnerabilities.json")
            ],
        },
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    remote_files = [
        {**f, "path": f"/remote/{source}.parquet"}
        for f, source in zip(files, ("sigma", "cisa-kev"))
    ]
    fix.write(
        snapshot / "corpus-manifest.json",
        {"files": remote_files, "totals": {"records": 2, "tokens": 2}},
    )
    digest = fix.sha(snapshot / "corpus-manifest.json")
    fix.write(
        snapshot / "summary.json", {"complete": True, "corpus_manifest_sha256": digest}
    )
    for name in ("rights", "integrity"):
        (snapshot / name).mkdir()
        fix.write(
            snapshot / name / "summary.json",
            {
                "audit_complete": True,
                "manifest_sha256": digest,
                "ledger_sha256": fix.sha(ledger),
            },
        )
    shutil.copyfile(ledger, snapshot / "rights/records.sqlite")
    fix.write(
        snapshot / "checks-complete.json",
        {
            "cpu_stages_complete": True,
            "summary_sha256": fix.sha(snapshot / "summary.json"),
            "rights_summary_sha256": fix.sha(snapshot / "rights/summary.json"),
            "integrity_summary_sha256": fix.sha(snapshot / "integrity/summary.json"),
        },
    )
    plan = tmp_path / "plan"
    fix.build_plan(ledger, sigma, cisa, plan)
    return ledger, sigma, cisa, plan, snapshot, payload


def test_only_license_changes_and_original_backup_survives(review, tmp_path):
    _, _, _, plan, snapshot, payload = review
    old = {p: fix.sha(p) for p in payload.glob("*.parquet")}
    output = tmp_path / "draft"
    result = fix.apply_plan(
        plan / "plan.json", snapshot, [("/remote", payload)], output
    )
    assert result["complete"] and not result["release_ready"]
    assert result["corrections_by_source"] == {"sigma": 1, "cisa-kev": 1}
    manifest = fix.read(output / "corpus-manifest.json")
    assert manifest["totals"] == {"records": 2, "tokens": 2}
    for f in manifest["files"]:
        new = pq.ParquetFile(f["path"]).read()
        before = pq.ParquetFile(f["original_local_path"]).read()
        assert new.drop(["license"]).equals(before.drop(["license"]))
        assert new["license"][0].as_py() in {"DRL-1.1", "CC0-1.0"}
        assert fix.sha(f["path"]) == f["sha256"]
    assert old == {p: fix.sha(p) for p in old}
    changes = [
        json.loads(line)
        for line in (output / "license-changes.jsonl").read_text().splitlines()
    ]
    assert len(changes) == 2 and all(
        c["old_license"] != c["new_license"] for c in changes
    )
    with pytest.raises(FileExistsError):
        fix.apply_plan(plan / "plan.json", snapshot, [("/remote", payload)], output)


def test_refuses_incomplete_job_before_writing(review, tmp_path):
    _, _, _, plan, snapshot, payload = review
    fix.write(snapshot / "checks-complete.json", {"cpu_stages_complete": False})
    with pytest.raises(ValueError, match="incomplete"):
        fix.apply_plan(
            plan / "plan.json", snapshot, [("/remote", payload)], tmp_path / "draft"
        )
    assert not (tmp_path / "draft").exists()


def test_refuses_damaged_backup(review, tmp_path):
    _, _, _, plan, snapshot, payload = review
    (payload / "sigma.parquet").write_bytes(b"bad")
    with pytest.raises(ValueError, match="checksum mismatch"):
        fix.apply_plan(
            plan / "plan.json", snapshot, [("/remote", payload)], tmp_path / "draft"
        )


def test_rejects_cisa_text_without_exact_source_evidence(review, tmp_path):
    ledger, sigma, cisa, *_ = review
    catalog = fix.read(cisa / "known_exploited_vulnerabilities.json")
    catalog["vulnerabilities"][0]["shortDescription"] = "different"
    fix.write(cisa / "known_exploited_vulnerabilities.json", catalog)
    inventory = fix.read(cisa / "sources.json")
    inventory["files"][1]["sha256"] = fix.sha(
        cisa / "known_exploited_vulnerabilities.json"
    )
    fix.write(cisa / "sources.json", inventory)
    with pytest.raises(ValueError, match="CISA text differs"):
        fix.build_plan(ledger, sigma, cisa, tmp_path / "new-plan")


def test_rejects_changed_plan_and_metadata_identity(review, tmp_path):
    _, _, _, plan, snapshot, payload = review
    path = plan / "plan.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="plan changed"):
        fix.apply_plan(path, snapshot, [("/remote", payload)], tmp_path / "draft")
    patch = fix.read(path)["files"][0]["patches"][0]
    table = pq.ParquetFile(payload / "cisa-kev.parquet").read()
    patch["old_license"] = "incorrect"
    with pytest.raises(ValueError, match="old value differs"):
        fix.patched_table(table, [patch])


def test_unmapped_or_escaping_sources_fail(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        fix.local_path("/other/file", [("/remote", tmp_path)])
    with pytest.raises(ValueError, match="Unsafe"):
        fix.local_path("/remote/../file", [("/remote", tmp_path)])
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        fix.local_path("/remote/escape/file", [("/remote", tmp_path)])
