#!/usr/bin/env python3
"""Materialize Qwen filtering JSONL outputs as Parquet datasets.

This script converts the completed Modal Qwen filter outputs into two useful
forms:

1. Decision Parquets: one row per scored record with Qwen labels and reasons.
2. Kept full-record Parquets: original normalized records filtered to
   qwen_should_keep=true, with Qwen decision fields appended.

The QA kept output is joined against local training-clean-v2 source Parquets.
The citation kept output is joined against the full citation paper Parquet,
which may live on an external drive.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
import json
from pathlib import Path
import shutil
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


QA_DECISION_FIELDS = [
    ("source_id", pa.string()),
    ("record_id", pa.string()),
    ("content_hash", pa.string()),
    ("qwen_security_relevance", pa.int16()),
    ("qwen_quality", pa.int16()),
    ("qwen_model_should_keep", pa.bool_()),
    ("qwen_should_keep", pa.bool_()),
    ("qwen_reason", pa.string()),
    ("qwen_parse_status", pa.string()),
    ("qwen_keep_policy", pa.string()),
    ("qwen_keep_policy_passed", pa.bool_()),
    ("qwen_keep_policy_reason", pa.string()),
    ("qwen_model", pa.string()),
    ("qwen_prompt_version", pa.string()),
    ("qwen_scored_at", pa.string()),
    ("qwen_shard_id", pa.string()),
    ("qwen_task", pa.string()),
    ("qwen_input_kind", pa.string()),
    ("qwen_raw_response", pa.string()),
]

CITATION_DECISION_FIELDS = QA_DECISION_FIELDS + [
    ("arxiv_id", pa.string()),
    ("title", pa.string()),
    ("primary_category", pa.string()),
    ("categories", pa.list_(pa.string())),
    ("abstract_preview", pa.string()),
]

QWEN_APPEND_FIELDS = [
    ("qwen_security_relevance", pa.int16()),
    ("qwen_quality", pa.int16()),
    ("qwen_model_should_keep", pa.bool_()),
    ("qwen_should_keep", pa.bool_()),
    ("qwen_reason", pa.string()),
    ("qwen_parse_status", pa.string()),
    ("qwen_keep_policy", pa.string()),
    ("qwen_keep_policy_passed", pa.bool_()),
    ("qwen_keep_policy_reason", pa.string()),
    ("qwen_model", pa.string()),
    ("qwen_prompt_version", pa.string()),
    ("qwen_scored_at", pa.string()),
    ("qwen_shard_id", pa.string()),
    ("qwen_task", pa.string()),
    ("qwen_input_kind", pa.string()),
    ("qwen_raw_response", pa.string()),
]


def main() -> None:
    args = parse_args()
    if args.overwrite:
        _remove_path(args.qa_decisions_out)
        _remove_path(args.citation_decisions_out)
        _remove_path(args.qa_kept_out)
        _remove_path(args.citation_kept_out)

    qa_decisions = _load_qa_decisions(args.qa_jsonl_root)
    citation_decisions = _load_citation_decisions(args.citation_jsonl_root)

    _write_decisions(
        qa_decisions.values(),
        QA_DECISION_FIELDS,
        args.qa_decisions_out,
        batch_size=args.write_batch_size,
    )
    _write_decisions(
        citation_decisions.values(),
        CITATION_DECISION_FIELDS,
        args.citation_decisions_out,
        batch_size=args.write_batch_size,
    )

    qa_kept = {
        record_id: row
        for record_id, row in qa_decisions.items()
        if row.get("qwen_should_keep") is True
    }
    citation_kept = {
        record_id: row
        for record_id, row in citation_decisions.items()
        if row.get("qwen_should_keep") is True
    }

    _write_qa_kept_records(
        decisions=qa_kept,
        source_root=args.qa_source_root,
        output_root=args.qa_kept_out,
        batch_size=args.read_batch_size,
    )
    _write_kept_records_from_parquet(
        decisions=citation_kept,
        input_path=args.citation_full_parquet,
        output_path=args.citation_kept_out,
        batch_size=args.read_batch_size,
        add_nullable_source_format=True,
    )

    print(
        json.dumps(
            {
                "qa_decisions": len(qa_decisions),
                "qa_kept": len(qa_kept),
                "citation_decisions": len(citation_decisions),
                "citation_kept": len(citation_kept),
                "qa_decisions_out": str(args.qa_decisions_out),
                "qa_kept_out": str(args.qa_kept_out),
                "citation_decisions_out": str(args.citation_decisions_out),
                "citation_kept_out": str(args.citation_kept_out),
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qa-jsonl-root",
        type=Path,
        default=Path("data/filtering/v3/qwen_qa_modal"),
    )
    parser.add_argument(
        "--citation-jsonl-root",
        type=Path,
        default=Path("data/filtering/v3/qwen_citation_abstract_modal"),
    )
    parser.add_argument(
        "--qa-source-root",
        type=Path,
        default=Path("data/training-clean-v2/normalized"),
    )
    parser.add_argument(
        "--citation-full-parquet",
        type=Path,
        default=Path(
            "/Volumes/SECURITY/security-corpus/data/arxiv/normalized/"
            "source_id=arxiv/citation_full_raw.parquet"
        ),
    )
    parser.add_argument(
        "--qa-decisions-out",
        type=Path,
        default=Path("data/filtering/v3/qwen_qa_decisions.parquet"),
    )
    parser.add_argument(
        "--citation-decisions-out",
        type=Path,
        default=Path("data/filtering/v3/qwen_citation_abstract_decisions.parquet"),
    )
    parser.add_argument(
        "--qa-kept-out",
        type=Path,
        default=Path("data/filtering/v3/qwen_qa_kept"),
    )
    parser.add_argument(
        "--citation-kept-out",
        type=Path,
        default=Path(
            "/Volumes/SECURITY/security-corpus/data/filtering/v3/"
            "qwen_citation_abstract_kept_full.parquet"
        ),
    )
    parser.add_argument("--read-batch-size", type=int, default=65_536)
    parser.add_argument("--write-batch-size", type=int, default=50_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_qa_decisions(root: Path) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        for row in _iter_jsonl(path):
            record_id = row.get("record_id")
            if not record_id:
                continue
            decisions[str(record_id)] = _project_row(row, QA_DECISION_FIELDS)
    return decisions


def _load_citation_decisions(root: Path) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*.jsonl")):
        for row in _iter_jsonl(path):
            record_id = row.get("record_id")
            if not record_id:
                continue
            decisions[str(record_id)] = _project_row(row, CITATION_DECISION_FIELDS)
    return decisions


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield row


def _project_row(row: Mapping[str, Any], fields: list[tuple[str, pa.DataType]]) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    for name, _type in fields:
        value = row.get(name)
        if name == "qwen_shard_id" and value is not None:
            value = str(value)
        projected[name] = value
    return projected


def _write_decisions(
    rows: Iterator[dict[str, Any]] | Any,
    fields: list[tuple[str, pa.DataType]],
    output_path: Path,
    *,
    batch_size: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([pa.field(name, type_) for name, type_ in fields])
    writer: pq.ParquetWriter | None = None
    batch: list[dict[str, Any]] = []
    try:
        for row in rows:
            batch.append(row)
            if len(batch) >= batch_size:
                writer = _write_row_batch(batch, schema, output_path, writer)
                batch.clear()
        if batch:
            writer = _write_row_batch(batch, schema, output_path, writer)
    finally:
        if writer is not None:
            writer.close()


def _write_row_batch(
    rows: list[dict[str, Any]],
    schema: pa.Schema,
    output_path: Path,
    writer: pq.ParquetWriter | None,
) -> pq.ParquetWriter:
    columns = {
        field.name: pa.array([row.get(field.name) for row in rows], type=field.type)
        for field in schema
    }
    table = pa.Table.from_pydict(columns, schema=schema)
    if writer is None:
        writer = pq.ParquetWriter(output_path, schema, compression="zstd")
    writer.write_table(table)
    return writer


def _write_qa_kept_records(
    *,
    decisions: Mapping[str, Mapping[str, Any]],
    source_root: Path,
    output_root: Path,
    batch_size: int,
) -> None:
    by_source: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record_id, decision in decisions.items():
        source_id = str(decision.get("source_id") or "").strip()
        if not source_id:
            continue
        by_source.setdefault(source_id, {})[record_id] = decision

    for source_id, source_decisions in sorted(by_source.items()):
        source_dir = source_root / f"source_id={source_id}"
        parquet_files = sorted(source_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No source parquet files found for {source_id}: {source_dir}")
        output_path = output_root / f"source_id={source_id}" / "part-00000.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_kept_records_from_parquet_files(
            decisions=source_decisions,
            input_paths=parquet_files,
            output_path=output_path,
            batch_size=batch_size,
            add_nullable_source_format=False,
        )


def _write_kept_records_from_parquet(
    *,
    decisions: Mapping[str, Mapping[str, Any]],
    input_path: Path,
    output_path: Path,
    batch_size: int,
    add_nullable_source_format: bool,
) -> None:
    _write_kept_records_from_parquet_files(
        decisions=decisions,
        input_paths=[input_path],
        output_path=output_path,
        batch_size=batch_size,
        add_nullable_source_format=add_nullable_source_format,
    )


def _write_kept_records_from_parquet_files(
    *,
    decisions: Mapping[str, Mapping[str, Any]],
    input_paths: list[Path],
    output_path: Path,
    batch_size: int,
    add_nullable_source_format: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keep_values = pa.array(list(decisions.keys()), type=pa.string())
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    try:
        for input_path in input_paths:
            parquet = pq.ParquetFile(input_path)
            if "record_id" not in parquet.schema_arrow.names:
                raise ValueError(f"{input_path} has no record_id column")
            for batch in parquet.iter_batches(batch_size=batch_size):
                table = pa.Table.from_batches([batch])
                mask = pc.is_in(table["record_id"], value_set=keep_values)
                kept = table.filter(mask)
                if kept.num_rows == 0:
                    continue
                kept = _append_qwen_columns(
                    kept,
                    decisions,
                    add_nullable_source_format=add_nullable_source_format,
                )
                if writer is None:
                    writer = pq.ParquetWriter(output_path, kept.schema, compression="zstd")
                writer.write_table(kept)
                rows_written += kept.num_rows
    finally:
        if writer is not None:
            writer.close()
    if rows_written != len(decisions):
        raise ValueError(
            f"{output_path}: wrote {rows_written:,} rows but expected {len(decisions):,}"
        )


def _append_qwen_columns(
    table: pa.Table,
    decisions: Mapping[str, Mapping[str, Any]],
    *,
    add_nullable_source_format: bool,
) -> pa.Table:
    record_ids = table["record_id"].to_pylist()
    if add_nullable_source_format and "source_format" not in table.schema.names:
        table = table.append_column(
            "source_format",
            pa.array([None] * table.num_rows, type=pa.string()),
        )
    for name, type_ in QWEN_APPEND_FIELDS:
        values = [decisions[record_id].get(name) for record_id in record_ids]
        if name == "qwen_shard_id":
            values = [None if value is None else str(value) for value in values]
        table = table.append_column(name, pa.array(values, type=type_))
    return table


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


if __name__ == "__main__":
    main()
