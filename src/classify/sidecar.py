"""Sidecar Parquet schemas and write helpers for filtering decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from classify.io import ensure_parent


ID_COLUMNS = ("source_id", "record_id", "content_hash")


QWEN_SIDECAR_CORE_FIELDS = (
    ("source_id", pa.string()),
    ("record_id", pa.string()),
    ("content_hash", pa.string()),
    ("qwen_security_relevance", pa.int8()),
    ("qwen_quality", pa.int8()),
    ("qwen_should_keep", pa.bool_()),
    ("qwen_reason", pa.string()),
    ("qwen_parse_status", pa.string()),
    ("qwen_model", pa.string()),
    ("qwen_model_revision", pa.string()),
    ("qwen_prompt_version", pa.string()),
    ("qwen_scored_at", pa.string()),
    ("qwen_shard_id", pa.string()),
    ("qwen_task", pa.string()),
    ("qwen_input_kind", pa.string()),
    ("qwen_raw_response", pa.string()),
)

ARTIFACT_QUALITY_CORE_FIELDS = (
    ("source_id", pa.string()),
    ("record_id", pa.string()),
    ("content_hash", pa.string()),
    ("artifact_family", pa.string()),
    ("artifact_quality_model", pa.string()),
    ("artifact_quality_version", pa.string()),
    ("artifact_quality_scored_at", pa.string()),
    ("artifact_duplicate_content_hash_count", pa.int64()),
    ("artifact_structural_should_review", pa.bool_()),
    ("artifact_quality_flags", pa.list_(pa.string())),
)


def qwen_sidecar_schema(
    extra_fields: Mapping[str, pa.DataType] | None = None,
) -> pa.Schema:
    """Return the standard Qwen sidecar schema.

    Extra fields let arXiv stages preserve paper/chunk identifiers without
    changing the common downstream contract.
    """
    return _schema_from_fields(QWEN_SIDECAR_CORE_FIELDS, extra_fields)


def artifact_quality_schema(
    extra_fields: Mapping[str, pa.DataType] | None = None,
) -> pa.Schema:
    """Return the structural artifact-quality sidecar schema."""
    return _schema_from_fields(ARTIFACT_QUALITY_CORE_FIELDS, extra_fields)


def validate_sidecar_schema(schema: pa.Schema, required: Sequence[str] = ID_COLUMNS) -> None:
    """Ensure a sidecar schema has the required key columns."""
    missing = [column for column in required if column not in schema.names]
    if missing:
        raise ValueError(f"Sidecar schema is missing key columns: {', '.join(missing)}")


def rows_to_table(rows: Sequence[Mapping[str, Any]], schema: pa.Schema) -> pa.Table:
    """Convert dictionaries to a table, filling omitted schema fields with nulls."""
    validate_sidecar_schema(schema)
    normalized_rows = [
        {name: row.get(name) for name in schema.names}
        for row in rows
    ]
    return pa.Table.from_pylist(normalized_rows, schema=schema)


def write_sidecar_rows(
    output_path: Path,
    rows: Sequence[Mapping[str, Any]],
    schema: pa.Schema,
    *,
    overwrite: bool = False,
) -> int:
    """Write one sidecar Parquet file and return the row count."""
    if not rows:
        return 0
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing sidecar: {output_path}")
    ensure_parent(output_path)
    table = rows_to_table(rows, schema)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return table.num_rows


def _schema_from_fields(
    fields: Sequence[tuple[str, pa.DataType]],
    extra_fields: Mapping[str, pa.DataType] | None = None,
) -> pa.Schema:
    merged = list(fields)
    if extra_fields:
        existing = {name for name, _ in merged}
        for name, data_type in extra_fields.items():
            if name in existing:
                raise ValueError(f"Duplicate sidecar field: {name}")
            merged.append((name, data_type))
    schema = pa.schema([pa.field(name, data_type) for name, data_type in merged])
    validate_sidecar_schema(schema)
    return schema
