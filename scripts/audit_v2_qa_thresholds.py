#!/usr/bin/env python3
"""Audit Q&A threshold candidates for second-pass filtering.

This script does not apply any filters or write cleaned data. It produces
source-specific score/length distributions plus candidate rule impact tables
so the V2 policy can be chosen from evidence.

Preferred use after building training-clean-v1:

    python scripts/audit_v2_qa_thresholds.py \
      --data-dir data/training-clean-v1 \
      --output-dir reports/training-clean-v1/v2_qa_thresholds

For exploratory local runs against the original normalized corpus, add
``--apply-v1-qa-filters``. That mirrors V1's invalid-content and zero-engagement
Q&A drops, but it does not perform global exact-deduplication.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import duckdb


DEFAULT_DATA_DIR = Path("data/training-clean-v1")
DEFAULT_OUTPUT_DIR = Path("reports/training-clean-v1/v2_qa_thresholds")
LENGTH_THRESHOLDS = (50, 100)
REQUESTED_V2_QA_THRESHOLDS = {
    "stackoverflow": (0, 150),
    "stackexchange-tor": (0, 100),
    "stackexchange-infosec": (1, 200),
    "stackexchange-reverseengineering": (1, 150),
    "stackexchange-crypto": (1, 150),
    "reddit-cybersecurity": (1, 100),
    "reddit-asknetsec": (1, 150),
    "reddit-hacking": (1, 100),
    "reddit-cybersecurity_help": (1, 200),
    "reddit-antivirus": (1, 150),
    "reddit-tor": (1, 100),
    "reddit-vpn": (1, 100),
    "reddit-netsec": (1, 100),
    "reddit-cryptography": (1, 150),
    "reddit-netsecstudents": (1, 150),
    "reddit-cloudflare": (1, 100),
    "reddit-security": (1, 100),
    "reddit-bugbounty": (1, 150),
    "reddit-computerviruses": (1, 150),
    "reddit-cybersecurityadvice": (1, 150),
    "reddit-phishing": (1, 100),
    "reddit-malware": (1, 100),
    "reddit-reverseengineering": (2, 200),
    "reddit-computersecurity": (1, 150),
    "reddit-blueteamsec": (1, 100),
    "reddit-cyberlaws": (1, 100),
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    parquet_files = _find_parquet_files(args.data_dir)
    if not parquet_files:
        print(f"No Parquet files found under {args.data_dir}")
        print("Expected layout like: data/training-clean-v1/normalized/source_id=*/part-*.parquet")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    corpus_expr = _read_parquet_expr(parquet_files)
    columns = _available_columns(con, corpus_expr)
    _create_qa_view(
        con,
        corpus_expr,
        columns,
        apply_v1_qa_filters=args.apply_v1_qa_filters,
    )
    _create_cutoff_view(con)
    _create_requested_threshold_view(con)

    reports = [
        (
            "qa_source_thresholds.csv",
            _source_thresholds_sql(),
        ),
        (
            "qa_score_threshold_impact.csv",
            _score_threshold_impact_sql(),
        ),
        (
            "qa_candidate_rule_counts.csv",
            _candidate_rule_counts_sql(),
        ),
        (
            "qa_short_no_answer_low_score_samples.csv",
            _short_no_answer_low_score_samples_sql(args.sample_limit_per_source),
        ),
        (
            "qa_closed_low_score_no_answer_samples.csv",
            _closed_low_score_no_answer_samples_sql(args.sample_limit_per_source),
        ),
        (
            "qa_score_zero_answered_samples.csv",
            _score_zero_answered_samples_sql(args.sample_limit_per_source),
        ),
        (
            "qa_requested_v2_filter_impact.csv",
            _requested_filter_impact_sql(),
        ),
        (
            "qa_requested_v2_filter_samples.csv",
            _requested_filter_samples_sql(args.sample_limit_per_source),
        ),
        (
            "qa_requested_v2_unconfigured_sources.csv",
            _requested_unconfigured_sources_sql(),
        ),
        (
            "qa_p25_content_lengths.csv",
            _p25_content_lengths_sql(),
        ),
        (
            "qa_v2_aggressive_filter_impact.csv",
            _v2_aggressive_filter_impact_sql(),
        ),
        (
            "qa_v2_aggressive_filter_samples.csv",
            _v2_aggressive_filter_samples_sql(args.sample_limit_per_source),
        ),
    ]

    for filename, sql in reports:
        rows = _fetch_all_dicts(con, sql)
        _write_csv(args.output_dir / filename, rows)

    _write_readme(args, parquet_files)
    print(f"Wrote Q&A V2 threshold reports to {args.output_dir}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Q&A/social threshold candidates for V2 filtering.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Input data directory (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output report directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--apply-v1-qa-filters",
        action="store_true",
        help=(
            "Apply V1 invalid-content and zero-engagement Q&A filters before "
            "analysis. Useful for exploratory runs against original normalized "
            "data; exact content-hash deduplication is not reproduced."
        ),
    )
    parser.add_argument(
        "--sample-limit-per-source",
        type=int,
        default=10,
        help="Sample rows per source for review CSVs (default: 10).",
    )
    return parser


def _find_parquet_files(data_dir: Path) -> list[Path]:
    return [
        path for path in sorted(data_dir.glob("**/normalized/source_id=*/*.parquet"))
        if not path.name.startswith("._")
    ]


def _read_parquet_expr(parquet_files: Sequence[Path]) -> str:
    file_list = ", ".join(_sql_string(path.as_posix()) for path in parquet_files)
    return f"""
        read_parquet(
            [{file_list}],
            union_by_name = true,
            hive_partitioning = true
        )
    """


def _available_columns(con: duckdb.DuckDBPyConnection, corpus_expr: str) -> set[str]:
    rows = con.sql(f"DESCRIBE SELECT * FROM {corpus_expr}").fetchall()
    return {row[0] for row in rows}


def _create_qa_view(
    con: duckdb.DuckDBPyConnection,
    corpus_expr: str,
    columns: set[str],
    *,
    apply_v1_qa_filters: bool,
) -> None:
    v1_filters = ""
    if apply_v1_qa_filters:
        content_expr = _raw_column_expr("content", columns, "NULL::VARCHAR")
        length_expr = _raw_column_expr("content_length", columns, "NULL::BIGINT")
        hash_expr = _raw_column_expr("content_hash", columns, "NULL::VARCHAR")
        answer_expr = _raw_column_expr("answer_count", columns, "NULL::BIGINT")
        score_expr = _raw_column_expr("score", columns, "NULL::BIGINT")
        v1_filters = """
          AND {content_expr} IS NOT NULL
          AND trim({content_expr}) <> ''
          AND {length_expr} IS NOT NULL
          AND {length_expr} > 0
          AND {hash_expr} IS NOT NULL
          AND {hash_expr} <> ''
          AND NOT (coalesce({answer_expr}, 0) = 0 AND coalesce({score_expr}, 0) <= 0)
        """.format(
            content_expr=content_expr,
            length_expr=length_expr,
            hash_expr=hash_expr,
            answer_expr=answer_expr,
            score_expr=score_expr,
        )

    con.sql(f"""
        CREATE VIEW qa_base AS
        SELECT
            source_id,
            {_source_family_case()} AS source_family,
            record_id,
            {_column_expr("source_record_id", columns, "NULL::VARCHAR")},
            {_column_expr("content_hash", columns, "NULL::VARCHAR")},
            {_column_expr("content_length", columns, "NULL::BIGINT")},
            {_column_expr("title", columns, "NULL::VARCHAR")},
            {_column_expr("source_url", columns, "NULL::VARCHAR")},
            {_column_expr("published_at", columns, "NULL::TIMESTAMP")},
            {_column_expr("content", columns, "NULL::VARCHAR")},
            coalesce({_raw_column_expr("score", columns, "NULL::BIGINT")}, 0)::BIGINT AS score,
            coalesce({_raw_column_expr("answer_count", columns, "NULL::BIGINT")}, 0)::BIGINT AS answer_count,
            coalesce({_raw_column_expr("has_accepted_answer", columns, "NULL::BOOLEAN")}, false)::BOOLEAN AS has_accepted_answer,
            coalesce({_raw_column_expr("closed", columns, "NULL::BOOLEAN")}, false)::BOOLEAN AS closed,
            {_column_expr("tags", columns, "NULL::VARCHAR[]")}
        FROM {corpus_expr}
        WHERE (
            source_id = 'stackoverflow'
            OR source_id LIKE 'stackexchange-%'
            OR source_id LIKE 'reddit-%'
        )
        {v1_filters}
    """)


def _column_expr(column: str, columns: set[str], fallback: str) -> str:
    if column in columns:
        return column
    return f"{fallback} AS {column}"


def _raw_column_expr(column: str, columns: set[str], fallback: str) -> str:
    if column in columns:
        return column
    return fallback


def _source_family_case() -> str:
    return """
        CASE
            WHEN source_id = 'stackoverflow' THEN 'qa-stackoverflow'
            WHEN source_id LIKE 'stackexchange-%' THEN 'qa-stackexchange'
            WHEN source_id LIKE 'reddit-%' THEN 'qa-reddit'
            ELSE 'other'
        END
    """


def _create_cutoff_view(con: duckdb.DuckDBPyConnection) -> None:
    con.sql("""
        CREATE VIEW qa_cutoffs AS
        SELECT
            source_id,
            source_family,
            count(*)::BIGINT AS records,
            coalesce(sum(content_length), 0)::BIGINT AS tokens,
            quantile_disc(score, 0.05)::BIGINT AS p05_score,
            quantile_disc(score, 0.10)::BIGINT AS p10_score,
            quantile_disc(score, 0.25)::BIGINT AS p25_score,
            quantile_disc(score, 0.50)::BIGINT AS p50_score,
            quantile_disc(score, 0.75)::BIGINT AS p75_score,
            quantile_disc(score, 0.90)::BIGINT AS p90_score,
            quantile_disc(content_length, 0.05)::BIGINT AS p05_tokens,
            quantile_disc(content_length, 0.10)::BIGINT AS p10_tokens,
            quantile_disc(content_length, 0.25)::BIGINT AS p25_tokens,
            quantile_disc(content_length, 0.50)::BIGINT AS p50_tokens,
            quantile_disc(content_length, 0.75)::BIGINT AS p75_tokens,
            quantile_disc(content_length, 0.90)::BIGINT AS p90_tokens
        FROM qa_base
        GROUP BY source_id, source_family
    """)


def _create_requested_threshold_view(con: duckdb.DuckDBPyConnection) -> None:
    rows = [
        f"SELECT {_sql_string(source_id)} AS source_id, {score_threshold}::BIGINT AS score_threshold, "
        f"{length_threshold}::BIGINT AS length_threshold"
        for source_id, (score_threshold, length_threshold) in sorted(REQUESTED_V2_QA_THRESHOLDS.items())
    ]
    con.sql(f"""
        CREATE VIEW requested_v2_qa_thresholds AS
        {" UNION ALL ".join(rows)}
    """)


def _source_thresholds_sql() -> str:
    return f"""
        SELECT
            c.source_id,
            c.source_family,
            c.records,
            c.tokens,
            c.p05_score,
            c.p10_score,
            c.p25_score,
            c.p50_score,
            c.p75_score,
            c.p90_score,
            c.p05_tokens,
            c.p10_tokens,
            c.p25_tokens,
            c.p50_tokens,
            c.p75_tokens,
            c.p90_tokens,
            avg(q.answer_count)::DOUBLE AS avg_answers,
            round(100.0 * count(*) FILTER (WHERE q.answer_count = 0) / count(*), 2) AS pct_no_answers,
            round(100.0 * count(*) FILTER (WHERE q.closed) / count(*), 2) AS pct_closed,
            round(100.0 * count(*) FILTER (WHERE q.content_length < {LENGTH_THRESHOLDS[0]}) / count(*), 2) AS pct_under_50_tokens,
            round(100.0 * count(*) FILTER (WHERE q.content_length < {LENGTH_THRESHOLDS[1]}) / count(*), 2) AS pct_under_100_tokens
        FROM qa_base q
        JOIN qa_cutoffs c USING (source_id, source_family)
        GROUP BY
            c.source_id,
            c.source_family,
            c.records,
            c.tokens,
            c.p05_score,
            c.p10_score,
            c.p25_score,
            c.p50_score,
            c.p75_score,
            c.p90_score,
            c.p05_tokens,
            c.p10_tokens,
            c.p25_tokens,
            c.p50_tokens,
            c.p75_tokens,
            c.p90_tokens
        ORDER BY c.tokens DESC
    """


def _requested_filter_impact_sql() -> str:
    return """
        WITH source_totals AS (
            SELECT
                source_id,
                source_family,
                count(*)::BIGINT AS source_records,
                coalesce(sum(content_length), 0)::BIGINT AS source_tokens
            FROM qa_base
            GROUP BY source_id, source_family
        ),
        configured AS (
            SELECT
                t.source_id,
                coalesce(s.source_family, 'missing') AS source_family,
                t.score_threshold,
                t.length_threshold,
                coalesce(s.source_records, 0)::BIGINT AS source_records,
                coalesce(s.source_tokens, 0)::BIGINT AS source_tokens
            FROM requested_v2_qa_thresholds t
            LEFT JOIN source_totals s USING (source_id)
        ),
        source_rows AS (
            SELECT
                c.source_id,
                c.source_family,
                c.score_threshold,
                c.length_threshold,
                c.source_records,
                c.source_tokens,
                count(q.record_id) FILTER (
                    WHERE q.score <= c.score_threshold
                      AND q.content_length < c.length_threshold
                )::BIGINT AS drop_records,
                coalesce(sum(q.content_length) FILTER (
                    WHERE q.score <= c.score_threshold
                      AND q.content_length < c.length_threshold
                ), 0)::BIGINT AS drop_tokens,
                count(q.record_id) FILTER (
                    WHERE q.score <= c.score_threshold
                      AND q.content_length < c.length_threshold
                      AND q.answer_count = 0
                )::BIGINT AS drop_no_answer_records,
                count(q.record_id) FILTER (
                    WHERE q.score <= c.score_threshold
                      AND q.content_length < c.length_threshold
                      AND q.closed
                )::BIGINT AS drop_closed_records
            FROM configured c
            LEFT JOIN qa_base q USING (source_id)
            GROUP BY
                c.source_id,
                c.source_family,
                c.score_threshold,
                c.length_threshold,
                c.source_records,
                c.source_tokens
        ),
        with_pct AS (
            SELECT
                source_id,
                source_family,
                score_threshold,
                length_threshold,
                source_records,
                source_tokens,
                drop_records,
                CASE
                    WHEN source_records = 0 THEN 0
                    ELSE round(100.0 * drop_records / source_records, 2)
                END AS pct_records,
                drop_tokens,
                CASE
                    WHEN source_tokens = 0 THEN 0
                    ELSE round(100.0 * drop_tokens / source_tokens, 2)
                END AS pct_tokens,
                drop_no_answer_records,
                drop_closed_records
            FROM source_rows
        )
        SELECT * FROM with_pct
        UNION ALL
        SELECT
            'TOTAL_CONFIGURED' AS source_id,
            'all-configured-qa' AS source_family,
            NULL::BIGINT AS score_threshold,
            NULL::BIGINT AS length_threshold,
            sum(source_records)::BIGINT AS source_records,
            sum(source_tokens)::BIGINT AS source_tokens,
            sum(drop_records)::BIGINT AS drop_records,
            CASE
                WHEN sum(source_records) = 0 THEN 0
                ELSE round(100.0 * sum(drop_records) / sum(source_records), 2)
            END AS pct_records,
            sum(drop_tokens)::BIGINT AS drop_tokens,
            CASE
                WHEN sum(source_tokens) = 0 THEN 0
                ELSE round(100.0 * sum(drop_tokens) / sum(source_tokens), 2)
            END AS pct_tokens,
            sum(drop_no_answer_records)::BIGINT AS drop_no_answer_records,
            sum(drop_closed_records)::BIGINT AS drop_closed_records
        FROM source_rows
        ORDER BY source_id
    """


def _requested_filter_samples_sql(limit_per_source: int) -> str:
    return f"""
        WITH candidates AS (
            SELECT
                q.source_id,
                q.source_family,
                t.score_threshold,
                t.length_threshold,
                q.record_id,
                q.content_length,
                q.score,
                q.answer_count,
                q.has_accepted_answer,
                q.closed,
                q.title,
                q.source_url,
                q.published_at,
                left(coalesce(q.content, ''), 1200) AS preview,
                row_number() OVER (
                    PARTITION BY q.source_id
                    ORDER BY q.content_length ASC NULLS FIRST, q.score ASC, q.record_id
                ) AS rn
            FROM qa_base q
            JOIN requested_v2_qa_thresholds t USING (source_id)
            WHERE q.score <= t.score_threshold
              AND q.content_length < t.length_threshold
        )
        SELECT *
        FROM candidates
        WHERE rn <= {limit_per_source}
        ORDER BY source_id, rn
    """


def _requested_unconfigured_sources_sql() -> str:
    return """
        SELECT
            q.source_id,
            q.source_family,
            count(*)::BIGINT AS records,
            coalesce(sum(q.content_length), 0)::BIGINT AS tokens
        FROM qa_base q
        LEFT JOIN requested_v2_qa_thresholds t USING (source_id)
        WHERE t.source_id IS NULL
        GROUP BY q.source_id, q.source_family
        ORDER BY tokens DESC
    """


def _p25_content_lengths_sql() -> str:
    return """
        SELECT
            source_id,
            source_family,
            records,
            tokens,
            p25_score,
            p50_score,
            p25_tokens AS p25_content_length,
            p50_tokens AS p50_content_length
        FROM qa_cutoffs
        ORDER BY tokens DESC
    """


def _v2_aggressive_filter_impact_sql() -> str:
    return """
        WITH source_rows AS (
            SELECT
                c.source_id,
                c.source_family,
                c.p50_score AS score_threshold,
                c.p50_tokens AS length_threshold,
                c.records AS source_records,
                c.tokens AS source_tokens,
                count(q.record_id) FILTER (
                    WHERE (q.score <= c.p50_score AND q.content_length <= c.p50_tokens)
                       OR (
                           q.source_family IN ('qa-stackoverflow', 'qa-stackexchange')
                           AND q.score <= c.p50_score
                           AND q.answer_count = 0
                       )
                )::BIGINT AS drop_records,
                coalesce(sum(q.content_length) FILTER (
                    WHERE (q.score <= c.p50_score AND q.content_length <= c.p50_tokens)
                       OR (
                           q.source_family IN ('qa-stackoverflow', 'qa-stackexchange')
                           AND q.score <= c.p50_score
                           AND q.answer_count = 0
                       )
                ), 0)::BIGINT AS drop_tokens,
                count(q.record_id) FILTER (
                    WHERE q.score <= c.p50_score
                      AND q.content_length <= c.p50_tokens
                )::BIGINT AS drop_length_low_score_records,
                count(q.record_id) FILTER (
                    WHERE q.source_family IN ('qa-stackoverflow', 'qa-stackexchange')
                      AND q.score <= c.p50_score
                      AND q.answer_count = 0
                )::BIGINT AS drop_so_se_unanswered_low_score_records,
                count(q.record_id) FILTER (
                    WHERE q.score <= c.p50_score
                      AND q.content_length <= c.p50_tokens
                      AND q.source_family IN ('qa-stackoverflow', 'qa-stackexchange')
                      AND q.answer_count = 0
                )::BIGINT AS drop_both_rule_records,
                count(q.record_id) FILTER (
                    WHERE (
                        (q.score <= c.p50_score AND q.content_length <= c.p50_tokens)
                        OR (
                            q.source_family IN ('qa-stackoverflow', 'qa-stackexchange')
                            AND q.score <= c.p50_score
                            AND q.answer_count = 0
                        )
                    )
                      AND q.answer_count = 0
                )::BIGINT AS drop_no_answer_records,
                count(q.record_id) FILTER (
                    WHERE (
                        (q.score <= c.p50_score AND q.content_length <= c.p50_tokens)
                        OR (
                            q.source_family IN ('qa-stackoverflow', 'qa-stackexchange')
                            AND q.score <= c.p50_score
                            AND q.answer_count = 0
                        )
                    )
                      AND q.answer_count > 0
                )::BIGINT AS drop_answered_records,
                count(q.record_id) FILTER (
                    WHERE (
                        (q.score <= c.p50_score AND q.content_length <= c.p50_tokens)
                        OR (
                            q.source_family IN ('qa-stackoverflow', 'qa-stackexchange')
                            AND q.score <= c.p50_score
                            AND q.answer_count = 0
                        )
                    )
                      AND q.has_accepted_answer
                )::BIGINT AS drop_accepted_answer_records,
                count(q.record_id) FILTER (
                    WHERE (
                        (q.score <= c.p50_score AND q.content_length <= c.p50_tokens)
                        OR (
                            q.source_family IN ('qa-stackoverflow', 'qa-stackexchange')
                            AND q.score <= c.p50_score
                            AND q.answer_count = 0
                        )
                    )
                      AND q.closed
                )::BIGINT AS drop_closed_records
            FROM qa_cutoffs c
            JOIN qa_base q USING (source_id, source_family)
            GROUP BY
                c.source_id,
                c.source_family,
                c.p50_score,
                c.p50_tokens,
                c.records,
                c.tokens
        ),
        with_pct AS (
            SELECT
                source_id,
                source_family,
                score_threshold,
                length_threshold,
                source_records,
                source_tokens,
                drop_records,
                round(100.0 * drop_records / source_records, 2) AS pct_records,
                drop_tokens,
                round(100.0 * drop_tokens / source_tokens, 2) AS pct_tokens,
                drop_length_low_score_records,
                drop_so_se_unanswered_low_score_records,
                drop_both_rule_records,
                drop_no_answer_records,
                drop_answered_records,
                drop_accepted_answer_records,
                drop_closed_records
            FROM source_rows
        )
        SELECT * FROM with_pct
        UNION ALL
        SELECT
            'TOTAL_QA' AS source_id,
            'all-qa' AS source_family,
            NULL::BIGINT AS score_threshold,
            NULL::BIGINT AS length_threshold,
            sum(source_records)::BIGINT AS source_records,
            sum(source_tokens)::BIGINT AS source_tokens,
            sum(drop_records)::BIGINT AS drop_records,
            round(100.0 * sum(drop_records) / sum(source_records), 2) AS pct_records,
            sum(drop_tokens)::BIGINT AS drop_tokens,
            round(100.0 * sum(drop_tokens) / sum(source_tokens), 2) AS pct_tokens,
            sum(drop_length_low_score_records)::BIGINT AS drop_length_low_score_records,
            sum(drop_so_se_unanswered_low_score_records)::BIGINT AS drop_so_se_unanswered_low_score_records,
            sum(drop_both_rule_records)::BIGINT AS drop_both_rule_records,
            sum(drop_no_answer_records)::BIGINT AS drop_no_answer_records,
            sum(drop_answered_records)::BIGINT AS drop_answered_records,
            sum(drop_accepted_answer_records)::BIGINT AS drop_accepted_answer_records,
            sum(drop_closed_records)::BIGINT AS drop_closed_records
        FROM source_rows
        ORDER BY source_id
    """


def _v2_aggressive_filter_samples_sql(limit_per_source: int) -> str:
    return f"""
        WITH candidates AS (
            SELECT
                q.source_id,
                q.source_family,
                c.p50_score AS score_threshold,
                c.p50_tokens AS length_threshold,
                CASE
                    WHEN q.score <= c.p50_score
                      AND q.content_length <= c.p50_tokens
                      AND q.source_family IN ('qa-stackoverflow', 'qa-stackexchange')
                      AND q.answer_count = 0
                        THEN 'both'
                    WHEN q.score <= c.p50_score
                      AND q.content_length <= c.p50_tokens
                        THEN 'length_low_score'
                    ELSE 'so_se_unanswered_low_score'
                END AS drop_rule,
                q.record_id,
                q.content_length,
                q.score,
                q.answer_count,
                q.has_accepted_answer,
                q.closed,
                q.title,
                q.source_url,
                q.published_at,
                left(coalesce(q.content, ''), 1200) AS preview,
                row_number() OVER (
                    PARTITION BY q.source_id
                    ORDER BY
                        q.answer_count DESC,
                        q.content_length DESC NULLS LAST,
                        q.score DESC,
                        q.record_id
                ) AS rn
            FROM qa_base q
            JOIN qa_cutoffs c USING (source_id, source_family)
            WHERE (q.score <= c.p50_score AND q.content_length <= c.p50_tokens)
               OR (
                   q.source_family IN ('qa-stackoverflow', 'qa-stackexchange')
                   AND q.score <= c.p50_score
                   AND q.answer_count = 0
               )
        )
        SELECT *
        FROM candidates
        WHERE rn <= {limit_per_source}
        ORDER BY source_id, rn
    """


def _score_threshold_impact_sql() -> str:
    threshold_rows = """
        SELECT 'raw_score_le_0' AS threshold_label, 0::BIGINT AS score_threshold, 0 AS sort_order
        UNION ALL SELECT 'raw_score_le_1', 1::BIGINT, 1
        UNION ALL SELECT 'source_p10', p10_score, 2
        UNION ALL SELECT 'source_p25', p25_score, 3
    """
    return f"""
        WITH expanded AS (
            SELECT
                q.*,
                c.records AS source_records,
                c.tokens AS source_tokens,
                t.threshold_label,
                t.score_threshold,
                t.sort_order
            FROM qa_base q
            JOIN qa_cutoffs c USING (source_id, source_family)
            JOIN LATERAL ({threshold_rows}) t ON true
        )
        SELECT
            source_id,
            source_family,
            threshold_label,
            score_threshold,
            count(*) FILTER (WHERE score <= score_threshold)::BIGINT AS records_score_le,
            round(100.0 * count(*) FILTER (WHERE score <= score_threshold) / max(source_records), 2) AS pct_records_score_le,
            coalesce(sum(content_length) FILTER (WHERE score <= score_threshold), 0)::BIGINT AS tokens_score_le,
            round(100.0 * coalesce(sum(content_length) FILTER (WHERE score <= score_threshold), 0) / max(source_tokens), 2) AS pct_tokens_score_le,
            count(*) FILTER (WHERE answer_count = 0 AND score <= score_threshold)::BIGINT AS records_no_answer_score_le,
            round(100.0 * count(*) FILTER (WHERE answer_count = 0 AND score <= score_threshold) / max(source_records), 2) AS pct_no_answer_score_le,
            count(*) FILTER (WHERE content_length < {LENGTH_THRESHOLDS[0]} AND score <= score_threshold)::BIGINT AS records_under_50_score_le,
            round(100.0 * count(*) FILTER (WHERE content_length < {LENGTH_THRESHOLDS[0]} AND score <= score_threshold) / max(source_records), 2) AS pct_under_50_score_le,
            count(*) FILTER (WHERE content_length < {LENGTH_THRESHOLDS[1]} AND score <= score_threshold)::BIGINT AS records_under_100_score_le,
            round(100.0 * count(*) FILTER (WHERE content_length < {LENGTH_THRESHOLDS[1]} AND score <= score_threshold) / max(source_records), 2) AS pct_under_100_score_le,
            count(*) FILTER (WHERE closed AND score <= score_threshold)::BIGINT AS records_closed_score_le,
            round(100.0 * count(*) FILTER (WHERE closed AND score <= score_threshold) / max(source_records), 2) AS pct_closed_score_le
        FROM expanded
        GROUP BY source_id, source_family, threshold_label, score_threshold, sort_order
        ORDER BY source_id, sort_order
    """


def _candidate_rule_counts_sql() -> str:
    common = """
        SELECT
            source_id,
            source_family,
            rule,
            count(*)::BIGINT AS records,
            round(100.0 * count(*) / max(source_records), 2) AS pct_records,
            coalesce(sum(content_length), 0)::BIGINT AS tokens,
            round(100.0 * coalesce(sum(content_length), 0) / max(source_tokens), 2) AS pct_tokens,
            quantile_disc(score, 0.50)::BIGINT AS p50_score,
            quantile_disc(content_length, 0.50)::BIGINT AS p50_tokens,
            max(score_cutoff)::BIGINT AS score_cutoff
        FROM rule_rows
        GROUP BY source_id, source_family, rule
    """
    return f"""
        WITH joined AS (
            SELECT
                q.*,
                c.records AS source_records,
                c.tokens AS source_tokens,
                c.p10_score,
                c.p25_score
            FROM qa_base q
            JOIN qa_cutoffs c USING (source_id, source_family)
        ),
        rule_rows AS (
            SELECT *, 'score_le_0' AS rule, 0::BIGINT AS score_cutoff
            FROM joined
            WHERE score <= 0
            UNION ALL
            SELECT *, 'score_le_source_p10', p10_score
            FROM joined
            WHERE score <= p10_score
            UNION ALL
            SELECT *, 'score_le_source_p25', p25_score
            FROM joined
            WHERE score <= p25_score
            UNION ALL
            SELECT *, 'no_answer_score_le_source_p10', p10_score
            FROM joined
            WHERE answer_count = 0 AND score <= p10_score
            UNION ALL
            SELECT *, 'no_answer_score_le_source_p25', p25_score
            FROM joined
            WHERE answer_count = 0 AND score <= p25_score
            UNION ALL
            SELECT *, 'under_50_tokens', NULL::BIGINT
            FROM joined
            WHERE content_length < {LENGTH_THRESHOLDS[0]}
            UNION ALL
            SELECT *, 'under_100_tokens', NULL::BIGINT
            FROM joined
            WHERE content_length < {LENGTH_THRESHOLDS[1]}
            UNION ALL
            SELECT *, 'under_50_tokens_score_le_source_p25', p25_score
            FROM joined
            WHERE content_length < {LENGTH_THRESHOLDS[0]} AND score <= p25_score
            UNION ALL
            SELECT *, 'under_100_tokens_score_le_source_p25', p25_score
            FROM joined
            WHERE content_length < {LENGTH_THRESHOLDS[1]} AND score <= p25_score
            UNION ALL
            SELECT *, 'under_50_tokens_no_answer_score_le_source_p25', p25_score
            FROM joined
            WHERE content_length < {LENGTH_THRESHOLDS[0]} AND answer_count = 0 AND score <= p25_score
            UNION ALL
            SELECT *, 'under_100_tokens_no_answer_score_le_source_p25', p25_score
            FROM joined
            WHERE content_length < {LENGTH_THRESHOLDS[1]} AND answer_count = 0 AND score <= p25_score
            UNION ALL
            SELECT *, 'closed_score_le_0', 0::BIGINT
            FROM joined
            WHERE closed AND score <= 0
            UNION ALL
            SELECT *, 'closed_score_le_source_p25', p25_score
            FROM joined
            WHERE closed AND score <= p25_score
            UNION ALL
            SELECT *, 'closed_no_answer_score_le_source_p25', p25_score
            FROM joined
            WHERE closed AND answer_count = 0 AND score <= p25_score
            UNION ALL
            SELECT *, 'closed_under_100_tokens_score_le_source_p25', p25_score
            FROM joined
            WHERE closed AND content_length < {LENGTH_THRESHOLDS[1]} AND score <= p25_score
        )
        {common}
        ORDER BY source_id, records DESC, rule
    """


def _short_no_answer_low_score_samples_sql(limit_per_source: int) -> str:
    return f"""
        WITH candidates AS (
            SELECT
                q.source_id,
                q.source_family,
                q.record_id,
                q.content_length,
                q.score,
                q.answer_count,
                c.p25_score AS source_p25_score,
                q.title,
                q.source_url,
                q.published_at,
                left(coalesce(q.content, ''), 1200) AS preview,
                row_number() OVER (
                    PARTITION BY q.source_id
                    ORDER BY q.content_length ASC NULLS FIRST, q.score ASC, q.record_id
                ) AS rn
            FROM qa_base q
            JOIN qa_cutoffs c USING (source_id, source_family)
            WHERE q.content_length < {LENGTH_THRESHOLDS[1]}
              AND q.answer_count = 0
              AND q.score <= c.p25_score
        )
        SELECT *
        FROM candidates
        WHERE rn <= {limit_per_source}
        ORDER BY source_id, rn
    """


def _closed_low_score_no_answer_samples_sql(limit_per_source: int) -> str:
    return f"""
        WITH candidates AS (
            SELECT
                q.source_id,
                q.source_family,
                q.record_id,
                q.content_length,
                q.score,
                q.answer_count,
                c.p25_score AS source_p25_score,
                q.has_accepted_answer,
                q.title,
                q.source_url,
                q.published_at,
                left(coalesce(q.content, ''), 1200) AS preview,
                row_number() OVER (
                    PARTITION BY q.source_id
                    ORDER BY q.score ASC, q.content_length ASC NULLS FIRST, q.record_id
                ) AS rn
            FROM qa_base q
            JOIN qa_cutoffs c USING (source_id, source_family)
            WHERE q.closed
              AND q.answer_count = 0
              AND q.score <= c.p25_score
        )
        SELECT *
        FROM candidates
        WHERE rn <= {limit_per_source}
        ORDER BY source_id, rn
    """


def _score_zero_answered_samples_sql(limit_per_source: int) -> str:
    return f"""
        WITH candidates AS (
            SELECT
                source_id,
                source_family,
                record_id,
                content_length,
                score,
                answer_count,
                has_accepted_answer,
                closed,
                title,
                source_url,
                published_at,
                left(coalesce(content, ''), 1200) AS preview,
                row_number() OVER (
                    PARTITION BY source_id
                    ORDER BY answer_count DESC, content_length DESC NULLS LAST, record_id
                ) AS rn
            FROM qa_base
            WHERE score <= 0
              AND answer_count > 0
        )
        SELECT *
        FROM candidates
        WHERE rn <= {limit_per_source}
        ORDER BY source_id, rn
    """


def _fetch_all_dicts(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    result = con.execute(sql)
    columns = [desc[0] for desc in result.description]
    return [dict(zip(columns, row)) for row in result.fetchall()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(args: argparse.Namespace, parquet_files: Sequence[Path]) -> None:
    caveat = ""
    if args.apply_v1_qa_filters:
        caveat = (
            "\n\n`--apply-v1-qa-filters` was used. The invalid-content and "
            "zero-engagement Q&A drops are mirrored, but global exact "
            "content-hash deduplication is not reproduced in this exploratory run."
        )

    readme = f"""# V2 Q&A Threshold Analysis

Data directory: `{args.data_dir}`

Parquet files: `{len(parquet_files)}`

This report is for choosing second-pass Q&A/social filtering thresholds. It
does not modify data.
{caveat}

## Files

- `qa_source_thresholds.csv`: per-source score and token percentiles.
- `qa_score_threshold_impact.csv`: impact of raw score cutoffs and source p10/p25 cutoffs.
- `qa_candidate_rule_counts.csv`: estimated record/token impact for compound rule shapes.
- `qa_short_no_answer_low_score_samples.csv`: examples for the short + no-answer + low-score rule.
- `qa_closed_low_score_no_answer_samples.csv`: examples for closed + no-answer + low-score Stack Exchange-style records.
- `qa_score_zero_answered_samples.csv`: examples showing why score-zero answered threads may or may not be worth keeping.
- `qa_requested_v2_filter_impact.csv`: impact of the explicit per-source score/length thresholds requested for V2.
- `qa_requested_v2_filter_samples.csv`: examples that would be dropped by the requested V2 thresholds.
- `qa_requested_v2_unconfigured_sources.csv`: QA sources present in data with no requested V2 threshold.
- `qa_p25_content_lengths.csv`: compact table of p25 content lengths and score cutoffs by source.
- `qa_v2_aggressive_filter_impact.csv`: impact of dropping score <= source p50 and content_length <= source p50, plus SO/SE score <= source p50 with no answers.
- `qa_v2_aggressive_filter_samples.csv`: examples dropped by the aggressive V2 QA variant.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
