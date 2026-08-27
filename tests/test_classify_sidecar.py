import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from classify.sidecar import (
    qwen_sidecar_schema,
    rows_to_table,
    validate_sidecar_schema,
    write_sidecar_rows,
)


def test_qwen_sidecar_schema_contains_required_key_columns():
    schema = qwen_sidecar_schema({"arxiv_id": pa.string()})

    assert {"source_id", "record_id", "content_hash"} <= set(schema.names)
    assert "qwen_parse_status" in schema.names
    assert "arxiv_id" in schema.names


def test_rows_to_table_fills_missing_schema_fields_with_nulls():
    schema = qwen_sidecar_schema()
    table = rows_to_table(
        [
            {
                "source_id": "s",
                "record_id": "s:1",
                "content_hash": "h",
                "qwen_should_keep": True,
                "extra": "ignored",
            }
        ],
        schema,
    )

    row = table.to_pylist()[0]
    assert row["source_id"] == "s"
    assert row["qwen_should_keep"] is True
    assert row["qwen_security_relevance"] is None
    assert "extra" not in row


def test_validate_sidecar_schema_rejects_missing_keys():
    with pytest.raises(ValueError):
        validate_sidecar_schema(pa.schema([("record_id", pa.string())]))


def test_write_sidecar_rows_refuses_overwrite(tmp_path):
    path = tmp_path / "sidecar.parquet"
    schema = qwen_sidecar_schema()
    row = {"source_id": "s", "record_id": "s:1", "content_hash": "h"}

    assert write_sidecar_rows(path, [row], schema) == 1
    assert pq.read_table(path).num_rows == 1
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        write_sidecar_rows(path, [row], schema)
