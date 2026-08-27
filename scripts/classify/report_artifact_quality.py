#!/usr/bin/env python3
"""Write structural sidecar quality reports for Sigma and CloudTrail."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.dataset as ds


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))

from classify.artifact_quality import score_cloudtrail_row, score_sigma_row  # noqa: E402
from classify.sidecar import artifact_quality_schema, write_sidecar_rows  # noqa: E402


SIGMA_EXTRA_FIELDS = {
    "sigma_yaml_parse_status": pa.string(),
    "sigma_missing_id": pa.bool_(),
    "sigma_missing_title": pa.bool_(),
    "sigma_missing_logsource": pa.bool_(),
    "sigma_missing_detection": pa.bool_(),
    "sigma_empty_or_trivial_detection": pa.bool_(),
    "sigma_incomplete_rule_source": pa.bool_(),
    "sigma_malformed_yaml": pa.bool_(),
    "sigma_content_length_outlier": pa.bool_(),
    "sigma_exact_duplicate_rule": pa.bool_(),
}

CLOUDTRAIL_EXTRA_FIELDS = {
    "cloudtrail_event_count": pa.int64(),
    "cloudtrail_session_duration_seconds": pa.int64(),
    "cloudtrail_action_count": pa.int64(),
    "cloudtrail_service_count": pa.int64(),
    "cloudtrail_principal_count": pa.int64(),
    "cloudtrail_action_repetition_ratio": pa.float64(),
    "cloudtrail_missing_event_count": pa.bool_(),
    "cloudtrail_missing_duration": pa.bool_(),
    "cloudtrail_no_services": pa.bool_(),
    "cloudtrail_no_actions": pa.bool_(),
    "cloudtrail_no_principals": pa.bool_(),
    "cloudtrail_insufficient_context": pa.bool_(),
    "cloudtrail_action_repetition_outlier": pa.bool_(),
    "cloudtrail_content_length_outlier": pa.bool_(),
    "cloudtrail_exact_duplicate_session": pa.bool_(),
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.source == "sigma":
        schema = artifact_quality_schema(SIGMA_EXTRA_FIELDS)
        scorer = _score_sigma
    elif args.source == "cloudtrail-flaws":
        schema = artifact_quality_schema(CLOUDTRAIL_EXTRA_FIELDS)
        scorer = _score_cloudtrail
    else:
        raise ValueError(f"Unsupported artifact source: {args.source}")

    dataset = ds.dataset(str(args.input), format="parquet", partitioning="hive")
    columns = _available_columns(dataset, args.source)
    duplicate_counts = _content_hash_counts(dataset)

    rows = []
    for batch in dataset.scanner(columns=columns, batch_size=args.batch_size).to_batches():
        for row in batch.to_pylist():
            if row.get("source_id") != args.source:
                continue
            rows.append(scorer(args, row, duplicate_counts))

    count = write_sidecar_rows(args.output, rows, schema, overwrite=args.overwrite)
    print(f"Wrote {count} {args.source} artifact-quality rows to {args.output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Normalized Parquet file/dir.")
    parser.add_argument("--output", type=Path, required=True, help="Output sidecar Parquet.")
    parser.add_argument("--source", choices=["sigma", "cloudtrail-flaws"], required=True)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--min-content-length",
        type=int,
        default=None,
        help="RESEARCHER: structural review flag for shorter records.",
    )
    parser.add_argument(
        "--max-content-length",
        type=int,
        default=None,
        help="RESEARCHER: structural review flag for longer records.",
    )
    parser.add_argument(
        "--min-event-count",
        type=int,
        default=None,
        help="RESEARCHER: CloudTrail insufficient-context flag.",
    )
    parser.add_argument(
        "--max-action-repetition-ratio",
        type=float,
        default=None,
        help="RESEARCHER: CloudTrail repetition outlier flag.",
    )
    return parser


def _score_sigma(
    args: argparse.Namespace,
    row: Mapping[str, Any],
    duplicate_counts: Counter[str],
) -> dict[str, Any]:
    return score_sigma_row(
        row,
        duplicate_count=duplicate_counts[str(row.get("content_hash") or "")],
        min_content_length=args.min_content_length,
        max_content_length=args.max_content_length,
    )


def _score_cloudtrail(
    args: argparse.Namespace,
    row: Mapping[str, Any],
    duplicate_counts: Counter[str],
) -> dict[str, Any]:
    return score_cloudtrail_row(
        row,
        duplicate_count=duplicate_counts[str(row.get("content_hash") or "")],
        min_content_length=args.min_content_length,
        max_content_length=args.max_content_length,
        min_event_count=args.min_event_count,
        max_action_repetition_ratio=args.max_action_repetition_ratio,
    )


def _available_columns(dataset: ds.Dataset, source: str) -> list[str]:
    available = set(dataset.schema.names)
    if source == "sigma":
        wanted = [
            "source_id",
            "record_id",
            "content_hash",
            "content_length",
            "title",
            "rule_id",
            "rule_source",
        ]
    else:
        wanted = [
            "source_id",
            "record_id",
            "content_hash",
            "content_length",
            "event_count",
            "session_duration_seconds",
            "principals",
            "actions",
            "aws_services",
        ]
    columns = [column for column in wanted if column in available]
    missing_keys = [column for column in ("source_id", "record_id", "content_hash") if column not in columns]
    if missing_keys:
        raise ValueError(f"Input is missing required key columns: {', '.join(missing_keys)}")
    return columns


def _content_hash_counts(dataset: ds.Dataset) -> Counter[str]:
    counts: Counter[str] = Counter()
    for batch in dataset.scanner(columns=["content_hash"]).to_batches():
        counts.update(str(value or "") for value in batch.column("content_hash").to_pylist())
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
