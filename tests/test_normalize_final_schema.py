from datetime import datetime, timezone
import gzip
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ingest.utils import compute_content_hash, compute_token_count
from scripts.release import finalize_local as final
from scripts.release.correct_licenses import write
from scripts.release.load_final_dataset import load_dataset
from scripts.release.normalize_final_schema import build, normalize


def row(identifier):
    text = "Security test " + identifier
    return dict(source_id="test", source_record_id=identifier, record_id="test:" + identifier,
                content=text, content_hash=compute_content_hash(text),
                content_length=compute_token_count(text), title=None, source_url=None,
                published_at=None, raw=None, ingested_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
                license="CC-BY-4.0")


def test_only_nulls_and_lossless_strings_can_change_types():
    before = pa.Table.from_pylist([row("a")])
    before = before.set_column(before.schema.get_field_index("title"), "title", pa.array([None], pa.int32()))
    after = normalize(before)
    assert before.to_pylist() == after.to_pylist()
    assert after.schema.field("title").type == pa.string()
    assert after.schema.field("content").type == pa.large_string()
    bad = before.set_column(before.schema.get_field_index("title"), "title", pa.array([42], pa.int32()))
    with pytest.raises(ValueError, match="cannot be reinterpreted"):
        normalize(bad)


def test_full_revision_keeps_values_and_loads_all_source_columns(tmp_path):
    parent, output = tmp_path / "parent", tmp_path / "out"
    (parent / "records").mkdir(parents=True)
    (parent / "evidence").mkdir()
    files, notes, source_tables = [], [], []
    for i in range(2):
        record = row(str(i))
        record["has_accepted_answer"] = None if i == 0 else True
        record["custom_" + str(i)] = "preserved"
        table = pa.Table.from_pylist([record])
        if i == 0:
            table = table.set_column(table.schema.get_field_index("has_accepted_answer"),
                                     "has_accepted_answer", pa.array([None], pa.int32()))
        source_tables.append(table)
        relative = f"records/{i:06d}.parquet"
        path = parent / relative
        pq.write_table(table, path)
        files.append(dict(path=str(path), relative_path=relative, sha256=final.sha(path),
                          records=1, tokens=record["content_length"], bytes=path.stat().st_size))
        notes.append(dict(file=i, output_row_index=0, row_index=0, source="test",
                          record_id=record["record_id"], content_hash=record["content_hash"], license=record["license"]))
    with gzip.open(parent / "record-metadata.jsonl.gz", "wt") as f:
        for note in notes:
            f.write(json.dumps(note) + "\n")
    totals = {key: sum(f[key] for f in files) for key in ("records", "tokens", "bytes")}
    write(parent / "schema.json", {})
    write(parent / "corpus-manifest.json", dict(files=files, totals=totals))
    write(parent / "final-prepared.json", dict(
        prepared=True, version=final.VERSION,
        manifest_sha256=final.sha(parent / "corpus-manifest.json"),
        record_metadata_sha256=final.sha(parent / "record-metadata.jsonl.gz"),
        schema_sha256=final.sha(parent / "schema.json"), evidence_sha256={}, totals=totals,
        retained_text_and_quality_columns_unchanged=True))
    assert final.audit(parent, workers=1)["integrity_passed"]
    result = build(parent, output)
    assert result["canonical_schema_identical"] and result["full_union_scan_rows"] == 2
    for i, entry in enumerate(files):
        assert final.sha(parent / entry["relative_path"]) == entry["sha256"]
        assert pq.ParquetFile(output / entry["relative_path"]).read().to_pylist() == source_tables[i].to_pylist()
    loaded = load_dataset(output).to_table().to_pylist()
    assert loaded[0]["custom_0"] == "preserved" and loaded[0]["custom_1"] is None
    assert loaded[1]["custom_1"] == "preserved" and loaded[1]["custom_0"] is None
    assert final.audit(output, workers=1)["integrity_passed"]
    (output / "schema.json").write_text("{}")
    with pytest.raises(ValueError, match="Schema description changed"):
        load_dataset(output)
