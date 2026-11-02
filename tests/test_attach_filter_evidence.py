from datetime import datetime, timezone
import gzip
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ingest.utils import compute_token_count
from scripts.release import finalize_local as final
from scripts.release.attach_filter_evidence import build, verify_bundle
from scripts.release.correct_licenses import write


def fixture(tmp_path):
    parent, bundle = tmp_path / "parent", tmp_path / "bundle"
    (parent / "records").mkdir(parents=True)
    (parent / "evidence").mkdir()
    bundle.mkdir()
    files, notes, evidence = [], [], []
    for i, source in enumerate(("primus-fineweb", "redsage-cfw", "youtube-commons")):
        text = f"Technical security example from {source}."
        row = dict(source_id=source, source_record_id="1", record_id=source+":1",
                   content=text, content_hash=hashlib.sha256(text.encode()).hexdigest(),
                   content_length=compute_token_count(text), title=source, license="upstream label",
                   ingested_at=datetime(2026,1,1,tzinfo=timezone.utc), published_at=None,
                   source_url=None, raw=None)
        path=parent/f"records/{i:06d}.parquet"
        pq.write_table(pa.Table.from_pylist([row]),path)
        files.append(dict(relative_path=str(path.relative_to(parent)),path=str(path),
                          sha256=final.sha(path),records=1,tokens=row["content_length"],bytes=path.stat().st_size))
        notes.append(dict(file=i,output_row_index=0,row_index=0,source=source,
                          record_id=row["record_id"],content_hash=row["content_hash"],license=row["license"]))
        evidence.append(dict(source=source,record_id=row["record_id"],content_hash=row["content_hash"],
                             content_length=row["content_length"],final_file=i,final_row=0,selection_decision=None))
    with gzip.open(parent / "record-metadata.jsonl.gz", "wt") as f:
        for note in notes:
            f.write(json.dumps(note)+"\n")
    totals={k:sum(x[k] for x in files) for k in ("records","tokens","bytes")}
    manifest=dict(files=files,totals=totals)
    write(parent/"corpus-manifest.json",manifest)
    write(parent/"schema.json",{})
    prepared=dict(prepared=True,version=final.VERSION,manifest_sha256=final.sha(parent/"corpus-manifest.json"),
                  record_metadata_sha256=final.sha(parent/"record-metadata.jsonl.gz"),
                  schema_sha256=final.sha(parent/"schema.json"),evidence_sha256={},totals=totals,
                  retained_text_and_quality_columns_unchanged=True)
    write(parent/"final-prepared.json",prepared)
    assert final.audit(parent,workers=1)["integrity_passed"]
    pq.write_table(pa.Table.from_pylist(evidence[:2]),bundle/"web-records.parquet")
    pq.write_table(pa.Table.from_pylist(evidence[2:]),bundle/"youtube-records.parquet")
    pq.write_table(pa.Table.from_pylist([dict(evidence[2],start=0,end=100)]),bundle/"youtube-spans.parquet")
    write(bundle/"input-plan.json",{"historical":"preserved"})
    report=dict(complete=True,manifest_sha256=prepared["manifest_sha256"],
                retained_by_source={e["source"]:1 for e in evidence},record_coverage=3,
                files={p.name:final.sha(p) for p in bundle.glob("*.parquet")},
                evidence_rows={"web-records":2,"youtube-records":1,"youtube-spans":1})
    write(bundle/"summary.json",report)
    return parent,bundle,manifest,prepared


def test_packages_unchanged_payloads_and_auditable_evidence(tmp_path):
    parent,bundle,manifest,prepared=fixture(tmp_path)
    out=tmp_path/"packaged"
    descriptor=build(parent,bundle,out)
    assert descriptor["record_coverage"] == 3
    for entry in manifest["files"]:
        assert final.sha(out/entry["relative_path"]) == entry["sha256"]
        assert final.sha(parent/entry["relative_path"]) == entry["sha256"]
    assert final.verify_prepared(parent)[0] == prepared
    assert final.audit(out,workers=1)["integrity_passed"]
    assert final.read(out/"filtering-evidence.json")["corpus_manifest_sha256"] == final.sha(out/"corpus-manifest.json")
    # A changed sidecar invalidates the prepared candidate, independently of text.
    target=out/descriptor["tables"]["web-records.parquet"]["relative_path"]
    target.unlink()  # Do not mutate the hard-linked parent bundle in this test.
    target.write_bytes(b"changed")
    with pytest.raises(ValueError,match="evidence changed"):
        final.verify_prepared(out)


def test_rejects_wrong_record_address_even_when_table_hash_was_updated(tmp_path):
    parent,bundle,manifest,prepared=fixture(tmp_path)
    path=bundle/"web-records.parquet"
    table=pq.read_table(path)
    table=table.set_column(table.schema.get_field_index("final_row"),"final_row",pa.array([1,0]))
    pq.write_table(table,path)
    report=final.read(bundle/"summary.json")
    report["files"][path.name]=final.sha(path)
    write(bundle/"summary.json",report)
    with pytest.raises(ValueError,match="outside payload"):
        verify_bundle(parent,manifest,prepared,bundle)
