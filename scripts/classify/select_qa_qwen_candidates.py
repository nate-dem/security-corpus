#!/usr/bin/env python3
"""Select QA/social records for Qwen review from QA quality sidecars.

This writes a non-destructive candidate sidecar. By default it keeps every QA
record as a Qwen candidate; researcher-chosen thresholds can be passed to drop
only the clearly bad bottom slice.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Sequence

import duckdb


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))

from classify.io import ensure_parent  # noqa: E402


SELECTION_VERSION = "qa-qwen-candidates-v1"


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {args.output}")

    ensure_parent(args.output)
    con = duckdb.connect()
    corpus_expr = _read_parquet_expr(args.corpus)
    quality_expr = _read_parquet_expr(args.quality_sidecar)
    corpus_columns = _available_columns(con, corpus_expr)
    quality_columns = _available_columns(con, quality_expr)
    confidence_expr = _confidence_expr(quality_columns, args.probability_prefix)
    candidate_condition, reason_expr = _candidate_sql(args, confidence_expr)
    selected_at = datetime.now(timezone.utc).isoformat()

    con.sql(f"""
        COPY (
            WITH qa AS (
                SELECT
                    source_id,
                    record_id,
                    content_hash,
                    coalesce({_optional_column("score", corpus_columns, "NULL::BIGINT")}, 0)::BIGINT AS source_score,
                    coalesce({_optional_column("answer_count", corpus_columns, "NULL::BIGINT")}, 0)::BIGINT AS answer_count,
                    coalesce({_optional_column("has_accepted_answer", corpus_columns, "NULL::BOOLEAN")}, false)::BOOLEAN AS has_accepted_answer,
                    coalesce({_optional_column("closed", corpus_columns, "NULL::BOOLEAN")}, false)::BOOLEAN AS closed
                FROM {corpus_expr}
                WHERE (
                    source_id = 'stackoverflow'
                    OR source_id LIKE 'stackexchange-%'
                    OR source_id LIKE 'reddit-%'
                )
            ),
            quality AS (
                SELECT
                    source_id,
                    record_id,
                    content_hash,
                    {args.score_column}::DOUBLE AS qa_quality_score,
                    {_optional_column(args.label_column, quality_columns, "NULL::VARCHAR")} AS qa_quality_predicted_label,
                    {confidence_expr} AS qa_quality_classifier_confidence,
                    classifier_model AS qa_quality_classifier_model,
                    classifier_version AS qa_quality_classifier_version,
                    scored_at AS qa_quality_scored_at
                FROM {quality_expr}
            )
            SELECT
                qa.source_id,
                qa.record_id,
                qa.content_hash,
                ({candidate_condition})::BOOLEAN AS qa_candidate_for_qwen,
                {reason_expr} AS qa_candidate_reason,
                quality.qa_quality_score,
                quality.qa_quality_predicted_label,
                quality.qa_quality_classifier_confidence,
                CASE
                    WHEN quality.qa_quality_classifier_confidence IS NULL THEN NULL
                    ELSE 1.0 - quality.qa_quality_classifier_confidence
                END AS qa_quality_classifier_uncertainty,
                quality.qa_quality_classifier_model,
                quality.qa_quality_classifier_version,
                quality.qa_quality_scored_at,
                {_sql_string(selected_at)} AS qa_candidate_selected_at,
                {_sql_string(args.selection_version)} AS qa_candidate_selection_version
            FROM qa
            LEFT JOIN quality USING (source_id, record_id, content_hash)
        )
        TO {_sql_string(args.output.as_posix())}
        (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)

    summary = con.sql(f"""
        SELECT
            count(*) AS rows,
            sum(qa_candidate_for_qwen::INT) AS candidates
        FROM read_parquet({_sql_string(args.output.as_posix())})
    """).fetchone()
    print(
        f"Wrote {summary[0]} QA candidate rows to {args.output} "
        f"({summary[1]} selected for Qwen)."
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="Normalized corpus Parquet file/dir.")
    parser.add_argument(
        "--quality-sidecar",
        type=Path,
        required=True,
        help="QA quality classifier sidecar Parquet file/dir.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output candidate sidecar Parquet.")
    parser.add_argument("--score-column", default="qa_quality_binary_score")
    parser.add_argument("--label-column", default="qa_quality_binary_predicted_label")
    parser.add_argument("--probability-prefix", default="qa_quality_binary_prob")
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=None,
        help="RESEARCHER: keep for Qwen when QA quality score is at least this value.",
    )
    parser.add_argument(
        "--min-uncertainty",
        type=float,
        default=None,
        help="RESEARCHER: keep for Qwen when classifier uncertainty is at least this value.",
    )
    parser.add_argument(
        "--metadata-min-score",
        type=int,
        default=None,
        help="RESEARCHER: keep for Qwen when source score is at least this value.",
    )
    parser.add_argument(
        "--metadata-min-answer-count",
        type=int,
        default=None,
        help="RESEARCHER: keep for Qwen when answer_count is at least this value.",
    )
    parser.add_argument(
        "--include-accepted-answer",
        action="store_true",
        help="Keep for Qwen when the source record has an accepted answer.",
    )
    parser.add_argument("--selection-version", default=SELECTION_VERSION)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _candidate_sql(args: argparse.Namespace, confidence_expr: str) -> tuple[str, str]:
    terms = ["quality.qa_quality_score IS NULL"]
    reasons = ["WHEN quality.qa_quality_score IS NULL THEN 'missing_quality_score'"]

    if args.min_quality_score is not None:
        terms.append(f"quality.qa_quality_score >= {args.min_quality_score}")
        reasons.append(
            "WHEN quality.qa_quality_score >= "
            f"{args.min_quality_score} THEN 'quality_score_threshold'"
        )
    if args.min_uncertainty is not None and confidence_expr != "NULL::DOUBLE":
        terms.append(
            f"(1.0 - quality.qa_quality_classifier_confidence) >= {args.min_uncertainty}"
        )
        reasons.append(
            "WHEN (1.0 - quality.qa_quality_classifier_confidence) >= "
            f"{args.min_uncertainty} THEN 'classifier_uncertainty'"
        )
    if args.metadata_min_score is not None:
        terms.append(f"qa.source_score >= {args.metadata_min_score}")
        reasons.append(
            f"WHEN qa.source_score >= {args.metadata_min_score} THEN 'source_score_signal'"
        )
    if args.metadata_min_answer_count is not None:
        terms.append(f"qa.answer_count >= {args.metadata_min_answer_count}")
        reasons.append(
            "WHEN qa.answer_count >= "
            f"{args.metadata_min_answer_count} THEN 'answer_count_signal'"
        )
    if args.include_accepted_answer:
        terms.append("qa.has_accepted_answer")
        reasons.append("WHEN qa.has_accepted_answer THEN 'accepted_answer_signal'")

    if len(terms) == 1:
        terms.append("true")
        reasons.append("WHEN true THEN 'all_records_no_thresholds'")

    condition = "\n                    OR ".join(terms)
    reason_expr = "CASE\n                    " + "\n                    ".join(reasons)
    reason_expr += "\n                    ELSE 'below_researcher_thresholds'\n                END"
    return condition, reason_expr


def _read_parquet_expr(path: Path) -> str:
    if path.is_dir():
        parquet_path = (path / "**" / "*.parquet").as_posix()
    else:
        parquet_path = path.as_posix()
    return (
        "read_parquet("
        f"{_sql_string(parquet_path)}, "
        "union_by_name = true, hive_partitioning = true"
        ")"
    )


def _available_columns(con: duckdb.DuckDBPyConnection, parquet_expr: str) -> set[str]:
    rows = con.sql(f"DESCRIBE SELECT * FROM {parquet_expr}").fetchall()
    return {row[0] for row in rows}


def _confidence_expr(columns: set[str], probability_prefix: str) -> str:
    probability_columns = sorted(
        column for column in columns if column.startswith(f"{probability_prefix}_")
    )
    if not probability_columns:
        return "NULL::DOUBLE"
    if len(probability_columns) == 1:
        return f"{probability_columns[0]}::DOUBLE"
    joined = ", ".join(f"{column}::DOUBLE" for column in probability_columns)
    return f"greatest({joined})"


def _optional_column(column: str, columns: set[str], fallback: str) -> str:
    if column in columns:
        return column
    return fallback


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
