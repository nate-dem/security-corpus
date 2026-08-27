#!/usr/bin/env python3
"""Build one prioritized Tier 1 chain Parquet from derived chain files.

Priority order:
1. CVE -> CWE -> CAPEC -> ATT&CK -> Sigma, with CISA KEV
2. CVE -> CWE -> CAPEC -> ATT&CK -> Sigma, without CISA KEV
3. CVE -> CWE -> CAPEC -> ATT&CK, with CISA KEV, no Sigma hop
4. CVE -> CWE -> CAPEC -> ATT&CK, without CISA KEV, no Sigma hop

The output is intentionally non-duplicative: a four-hop base chain is included
only when that base chain has no Sigma rule coverage.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken


ROOT = Path(__file__).resolve().parents[1]
_CLEAN_V2_REASONING_OUTPUT_DIR_NAME = "tier1-reasoning-clean-v2"

_PRIORITY_SCHEMA = pa.schema(
    [
        ("priority", pa.int8()),
        ("priority_label", pa.string()),
        ("chain_type", pa.string()),
        ("has_cisa_kev", pa.bool_()),
        ("has_sigma", pa.bool_()),
        ("chain_record_id", pa.string()),
        ("base_chain_id", pa.string()),
        ("detection_chain_id", pa.string()),
        ("cve_id", pa.string()),
        ("nvd_node_id", pa.string()),
        ("cisa_kev_node_id", pa.string()),
        ("is_known_exploited", pa.bool_()),
        ("cwe_id", pa.string()),
        ("cwe_title", pa.string()),
        ("capec_id", pa.string()),
        ("capec_title", pa.string()),
        ("attack_technique_id", pa.string()),
        ("attack_technique_title", pa.string()),
        ("sigma_rule_id", pa.string()),
        ("sigma_rule_title", pa.string()),
        ("sigma_rule_level", pa.string()),
        ("sigma_node_id", pa.string()),
        ("sigma_source_url", pa.string()),
        ("path_node_ids", pa.list_(pa.string())),
        ("path_relationships", pa.list_(pa.string())),
        ("evidence_edge_ids", pa.list_(pa.string())),
        ("content", pa.string()),
        ("content_length", pa.int64()),
        ("chain_text", pa.string()),
        ("chain_token_count", pa.int64()),
    ]
)

_DETECTION_COLUMNS = [
    "detection_chain_id",
    "base_chain_id",
    "cve_id",
    "nvd_node_id",
    "cisa_kev_node_id",
    "is_known_exploited",
    "cwe_id",
    "cwe_title",
    "capec_id",
    "capec_title",
    "attack_technique_id",
    "attack_technique_title",
    "sigma_rule_id",
    "sigma_rule_title",
    "sigma_rule_level",
    "sigma_node_id",
    "sigma_source_url",
    "path_node_ids",
    "path_relationships",
    "evidence_edge_ids",
    "chain_text",
]

_CHAIN_COLUMNS = [
    "chain_id",
    "cve_id",
    "nvd_node_id",
    "cisa_kev_node_id",
    "is_known_exploited",
    "cwe_id",
    "cwe_title",
    "capec_id",
    "capec_title",
    "attack_technique_id",
    "attack_technique_title",
    "sigma_rule_count",
    "path_node_ids",
    "path_relationships",
    "evidence_edge_ids",
    "chain_text",
]

_PRIORITY_LABELS = {
    1: "CVE-CWE-CAPEC-ATTACK-Sigma with CISA KEV",
    2: "CVE-CWE-CAPEC-ATTACK-Sigma without CISA KEV",
    3: "CVE-CWE-CAPEC-ATTACK with CISA KEV, no Sigma",
    4: "CVE-CWE-CAPEC-ATTACK without CISA KEV, no Sigma",
}


def build_priority_chains(
    input_dir: Path,
    output_path: Path,
    *,
    summary_path: Path,
    batch_size: int,
) -> dict[int, dict[str, Any]]:
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    summary_path = Path(summary_path)
    _validate_clean_v2_reasoning_dir(input_dir)
    _validate_clean_v2_priority_path(output_path, "output")
    _validate_clean_v2_priority_path(summary_path, "summary")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = tiktoken.get_encoding("cl100k_base")
    summary = _empty_summary()
    writer: pq.ParquetWriter | None = None

    try:
        writer = _write_detection_rows(
            input_dir / "detection_chains.parquet",
            output_path=output_path,
            writer=writer,
            summary=summary,
            encoder=encoder,
            batch_size=batch_size,
        )
        writer = _write_four_hop_fallback_rows(
            input_dir / "chains.parquet",
            output_path=output_path,
            writer=writer,
            summary=summary,
            encoder=encoder,
            batch_size=batch_size,
        )
    finally:
        if writer is not None:
            writer.close()

    _write_summary(summary, summary_path)
    return summary


def _validate_clean_v2_reasoning_dir(input_dir: Path) -> None:
    if input_dir.name != _CLEAN_V2_REASONING_OUTPUT_DIR_NAME:
        raise ValueError(
            "Priority chains must be built from the clean-v2 Tier 1 reasoning directory "
            f"named {_CLEAN_V2_REASONING_OUTPUT_DIR_NAME}, got {input_dir}."
        )


def _validate_clean_v2_priority_path(path: Path, label: str) -> None:
    if path.parent.name != _CLEAN_V2_REASONING_OUTPUT_DIR_NAME:
        raise ValueError(
            f"Priority chain {label} path must be inside {_CLEAN_V2_REASONING_OUTPUT_DIR_NAME}, "
            f"got {path}."
        )


def _write_detection_rows(
    path: Path,
    *,
    output_path: Path,
    writer: pq.ParquetWriter | None,
    summary: dict[int, dict[str, Any]],
    encoder,
    batch_size: int,
) -> pq.ParquetWriter:
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(columns=_DETECTION_COLUMNS, batch_size=batch_size):
        rows = []
        for row in batch.to_pylist():
            priority = 1 if row["cisa_kev_node_id"] else 2
            rows.append(
                {
                    "priority": priority,
                    "priority_label": _PRIORITY_LABELS[priority],
                    "chain_type": "cve-cwe-capec-attack-sigma",
                    "has_cisa_kev": bool(row["cisa_kev_node_id"]),
                    "has_sigma": True,
                    "chain_record_id": row["detection_chain_id"],
                    "base_chain_id": row["base_chain_id"],
                    "detection_chain_id": row["detection_chain_id"],
                    "cve_id": row["cve_id"],
                    "nvd_node_id": row["nvd_node_id"],
                    "cisa_kev_node_id": row["cisa_kev_node_id"],
                    "is_known_exploited": row["is_known_exploited"],
                    "cwe_id": row["cwe_id"],
                    "cwe_title": row["cwe_title"],
                    "capec_id": row["capec_id"],
                    "capec_title": row["capec_title"],
                    "attack_technique_id": row["attack_technique_id"],
                    "attack_technique_title": row["attack_technique_title"],
                    "sigma_rule_id": row["sigma_rule_id"],
                    "sigma_rule_title": row["sigma_rule_title"],
                    "sigma_rule_level": row["sigma_rule_level"],
                    "sigma_node_id": row["sigma_node_id"],
                    "sigma_source_url": row["sigma_source_url"],
                    "path_node_ids": row["path_node_ids"],
                    "path_relationships": row["path_relationships"],
                    "evidence_edge_ids": row["evidence_edge_ids"],
                    "chain_text": row["chain_text"],
                }
            )
        writer = _write_rows(output_path, rows, writer, summary, encoder)
    return writer


def _write_four_hop_fallback_rows(
    path: Path,
    *,
    output_path: Path,
    writer: pq.ParquetWriter | None,
    summary: dict[int, dict[str, Any]],
    encoder,
    batch_size: int,
) -> pq.ParquetWriter:
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(columns=_CHAIN_COLUMNS, batch_size=batch_size):
        rows = []
        for row in batch.to_pylist():
            if row["sigma_rule_count"] > 0:
                continue
            priority = 3 if row["cisa_kev_node_id"] else 4
            rows.append(
                {
                    "priority": priority,
                    "priority_label": _PRIORITY_LABELS[priority],
                    "chain_type": "cve-cwe-capec-attack",
                    "has_cisa_kev": bool(row["cisa_kev_node_id"]),
                    "has_sigma": False,
                    "chain_record_id": row["chain_id"],
                    "base_chain_id": row["chain_id"],
                    "detection_chain_id": None,
                    "cve_id": row["cve_id"],
                    "nvd_node_id": row["nvd_node_id"],
                    "cisa_kev_node_id": row["cisa_kev_node_id"],
                    "is_known_exploited": row["is_known_exploited"],
                    "cwe_id": row["cwe_id"],
                    "cwe_title": row["cwe_title"],
                    "capec_id": row["capec_id"],
                    "capec_title": row["capec_title"],
                    "attack_technique_id": row["attack_technique_id"],
                    "attack_technique_title": row["attack_technique_title"],
                    "sigma_rule_id": None,
                    "sigma_rule_title": None,
                    "sigma_rule_level": None,
                    "sigma_node_id": None,
                    "sigma_source_url": None,
                    "path_node_ids": row["path_node_ids"],
                    "path_relationships": row["path_relationships"],
                    "evidence_edge_ids": row["evidence_edge_ids"],
                    "chain_text": row["chain_text"],
                }
            )
        if rows:
            writer = _write_rows(output_path, rows, writer, summary, encoder)
    return writer


def _write_rows(
    output_path: Path,
    rows: list[dict[str, Any]],
    writer: pq.ParquetWriter | None,
    summary: dict[int, dict[str, Any]],
    encoder,
) -> pq.ParquetWriter:
    token_counts = _token_counts(encoder, [row["chain_text"] for row in rows])
    for row, token_count in zip(rows, token_counts):
        row["content"] = row["chain_text"]
        row["content_length"] = token_count
        row["chain_token_count"] = token_count
        priority_summary = summary[row["priority"]]
        priority_summary["records"] += 1
        priority_summary["total_chain_tokens"] += token_count

    table = pa.Table.from_pylist(rows, schema=_PRIORITY_SCHEMA)
    if writer is None:
        writer = pq.ParquetWriter(output_path, _PRIORITY_SCHEMA, compression="snappy")
    writer.write_table(table)
    return writer


def _token_counts(encoder, texts: list[str]) -> list[int]:
    try:
        encoded = encoder.encode_batch(texts, num_threads=4, disallowed_special=())
    except TypeError:
        encoded = encoder.encode_batch(texts, disallowed_special=())
    return [len(tokens) for tokens in encoded]


def _empty_summary() -> dict[int, dict[str, Any]]:
    return {
        priority: {
            "priority": priority,
            "priority_label": label,
            "records": 0,
            "total_chain_tokens": 0,
        }
        for priority, label in _PRIORITY_LABELS.items()
    }


def _write_summary(summary: dict[int, dict[str, Any]], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for priority in sorted(summary):
        row = dict(summary[priority])
        records = row["records"]
        row["avg_chain_tokens"] = row["total_chain_tokens"] / records if records else 0
        rows.append(row)

    if summary_path.suffix == ".json":
        summary_path.write_text(json.dumps(rows, indent=2) + "\n")
        return

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "priority",
                "priority_label",
                "records",
                "total_chain_tokens",
                "avg_chain_tokens",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _path_arg(value: str) -> Path:
    return Path(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build one prioritized Tier 1 chain Parquet.")
    parser.add_argument(
        "--input-dir",
        type=_path_arg,
        default=Path("data/tier1-reasoning-clean-v2"),
        help="Directory containing chains.parquet and detection_chains.parquet.",
    )
    parser.add_argument(
        "--output-path",
        type=_path_arg,
        default=Path("data/tier1-reasoning-clean-v2/priority_chains.parquet"),
        help="Combined prioritized Parquet output.",
    )
    parser.add_argument(
        "--summary-path",
        type=_path_arg,
        default=Path("data/tier1-reasoning-clean-v2/priority_summary.csv"),
        help="CSV or JSON summary output.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Rows per read/write/tokenization batch.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_priority_chains(
        args.input_dir,
        args.output_path,
        summary_path=args.summary_path,
        batch_size=args.batch_size,
    )
    total_records = sum(item["records"] for item in summary.values())
    total_tokens = sum(item["total_chain_tokens"] for item in summary.values())
    print(f"output: {args.output_path}")
    print(f"summary: {args.summary_path}")
    print(f"records: {total_records:,}")
    print(f"chain_text tokens: {total_tokens:,}")
    for priority in sorted(summary):
        item = summary[priority]
        print(
            f"P{priority}: {item['records']:,} records, "
            f"{item['total_chain_tokens']:,} tokens"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
