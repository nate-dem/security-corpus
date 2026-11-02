import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ingest.utils import compute_content_hash, compute_token_count
from scripts.release import audit_snapshot, record_rights
from scripts.youtube.download import _write_json
from scripts.youtube.profile import _sha256


def document(prepared=False):
    text = "A security audit checks authentication and authorization."
    common = {
        "content_hash": compute_content_hash(text),
        "content_length": compute_token_count(text),
    }
    if prepared:
        return {
            **common,
            "text": text,
            "source": "youtube-commons",
            "dataset_revision": "revision",
            "source_shard": "input.parquet",
            "source_row": 0,
        }
    return {
        **common,
        "content": text,
        "source_id": "sigma",
        "source_record_id": "id",
        "record_id": "sigma:id",
        "license": "LGPL-2.1",
    }


def entry(path, records):
    pq.write_table(pa.Table.from_pylist(records), path)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "records": len(records),
        "tokens": sum(r["content_length"] for r in records),
    }


@pytest.mark.parametrize("prepared", [False, True])
def test_recomputes_actual_content_and_tokens(tmp_path, prepared):
    row = document(prepared)
    f = entry(tmp_path / "input.parquet", [row])
    result = audit_snapshot.check_file(f)
    assert not result["issues"]
    assert result["counts"]["tokens"] == result["counts"]["recomputed_tokens"]
    row["content_hash"] = "0" * 64
    row["content_length"] += 1
    f = entry(tmp_path / "input.parquet", [row])
    result = audit_snapshot.check_file(f)
    assert result["issues"] == {
        "content_hash_mismatch": 1,
        "content_length_mismatch": 1,
    }


def test_final_manifest_counts_duplicates_and_resumes(tmp_path, monkeypatch):
    f = entry(tmp_path / "input.parquet", [document(), document()])
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest, {"files": [f], "totals": {k: f[k] for k in ("records", "tokens")}}
    )
    rights, output = tmp_path / "rights", tmp_path / "integrity"
    record_rights.execute(manifest, rights)
    first = audit_snapshot.execute(manifest, output, rights, workers=1)
    assert first["audit_complete"] and not first["integrity_passed"]
    assert first["issues"] == {"duplicate_record_ids": 1}
    assert first["exact_content_groups"] == first["extra_content_copies"] == 1
    assert first["payload_bytes"] == (tmp_path / "input.parquet").stat().st_size
    monkeypatch.setattr(
        audit_snapshot, "check_file", lambda _: pytest.fail("Reuse verified checkpoint")
    )
    assert audit_snapshot.execute(manifest, output, rights, workers=1) == first
