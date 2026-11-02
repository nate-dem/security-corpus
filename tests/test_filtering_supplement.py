import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.release.prepare_filtering_supplement import project_table
from scripts.release.upload_huggingface_private import verify_package
from tests.test_huggingface_package import sample_package


def test_projection_preserves_values_but_removes_execution_and_arrow_metadata(tmp_path):
    schema = pa.schema([
        pa.field("record_id", pa.string(), metadata={b"hostname": b"private-node"}),
        pa.field("score", pa.float64()),
        pa.field("attempts_json", pa.string()),
    ], metadata={b"local_path": b"/Users/private/work"})
    source = tmp_path / "source.parquet"
    table = pa.Table.from_pylist([
        {"record_id": "doc:1", "score": 0.80000000001,
         "attempts_json": '{"job_id": "123", "hostname": "private-node"}'},
        {"record_id": "doc:2", "score": None, "attempts_json": None},
    ], schema=schema)
    pq.write_table(table, source)
    out = tmp_path / "out.parquet"
    result = project_table(source, out, ("record_id", "score"))
    actual = pq.read_table(out)
    assert actual.to_pylist() == table.select(["record_id", "score"]).to_pylist()
    assert result["rows"] == 2
    assert actual.schema.metadata is None
    assert all(field.metadata is None for field in actual.schema)


def test_private_values_in_allowed_free_text_block_export(tmp_path):
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"reason": ["See /Users/private/work/output"]}), source)
    with pytest.raises(ValueError, match="Private metadata"):
        project_table(source, tmp_path / "out.parquet", ("reason",))


def test_generic_technical_terms_are_not_removed(tmp_path):
    source = tmp_path / "source.parquet"
    value = "Explains Slurm job scheduling and hostname resolution."
    pq.write_table(pa.table({"reason": [value]}), source)
    project_table(source, tmp_path / "out.parquet", ("reason",))
    assert pq.read_table(tmp_path / "out.parquet").to_pylist() == [{"reason": value}]


def test_upload_permits_only_named_supplement_files(tmp_path):
    sample_package(tmp_path)
    folder = tmp_path / "metadata/filtering"
    folder.mkdir(parents=True)
    for name in ["README.md", "web-records.parquet"]:
        path = folder / name
        path.write_bytes(b"research metadata")
        with (tmp_path / "SHA256SUMS").open("a") as handle:
            handle.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  metadata/filtering/{name}\n")
    verify_package(tmp_path)
    private = folder / "run-config.json"
    private.write_text('{"hostname": "private-node"}')
    with (tmp_path / "SHA256SUMS").open("a") as handle:
        handle.write(f"{hashlib.sha256(private.read_bytes()).hexdigest()}  metadata/filtering/run-config.json\n")
    with pytest.raises(ValueError, match="outside the CyberOpen release"):
        verify_package(tmp_path)
