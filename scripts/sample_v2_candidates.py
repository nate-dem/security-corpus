#!/usr/bin/env python3
"""Sample candidate records for second-pass filtering decisions.

Run this against training-clean-v1 when possible:

    python scripts/sample_v2_candidates.py \
      --data-dir data/training-clean-v1 \
      --output-dir reports/training-clean-v1/v2_samples

The output is intentionally review-oriented: compact CSV files with metadata
and content previews for the records most likely to need V2 policy decisions.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb


DEFAULT_DATA_DIR = Path("data/training-clean-v1")
DEFAULT_OUTPUT_DIR = Path("reports/training-clean-v1/v2_samples")


@dataclass(frozen=True)
class SampleReport:
    name: str
    title: str
    sql: str
    description: str


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
    _create_sample_view(con, corpus_expr, columns)

    reports = _build_reports(args.limit, args.preview_chars)
    index_lines = [
        "# V2 Candidate Samples",
        "",
        f"Data directory: `{args.data_dir}`",
        f"Parquet files: `{len(parquet_files)}`",
        "",
    ]

    for report in reports:
        rows = _fetch_all_dicts(con, report.sql)
        output_path = args.output_dir / f"{report.name}.csv"
        _write_csv(output_path, rows)
        index_lines.extend([
            f"## {report.title}",
            "",
            report.description,
            "",
            f"Rows: `{len(rows)}`",
            f"CSV: `{output_path.name}`",
            "",
        ])

    (args.output_dir / "README.md").write_text("\n".join(index_lines), encoding="utf-8")
    print(f"Wrote V2 samples to {args.output_dir}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate review samples for second-pass filtering decisions.",
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
        help=f"Sample report directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum rows per sample CSV (default: 50).",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=1200,
        help="Content preview characters per sampled row (default: 1200).",
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


def _create_sample_view(
    con: duckdb.DuckDBPyConnection,
    corpus_expr: str,
    columns: set[str],
) -> None:
    con.sql(f"""
        CREATE VIEW sample_base AS
        SELECT
            source_id,
            record_id,
            {_column_expr("source_record_id", columns, "NULL::VARCHAR")},
            {_column_expr("content_hash", columns, "NULL::VARCHAR")},
            {_column_expr("content_length", columns, "NULL::BIGINT")},
            {_column_expr("title", columns, "NULL::VARCHAR")},
            {_column_expr("source_url", columns, "NULL::VARCHAR")},
            {_column_expr("published_at", columns, "NULL::TIMESTAMP")},
            {_column_expr("content", columns, "NULL::VARCHAR")},
            {_column_expr("score", columns, "NULL::BIGINT")},
            {_column_expr("answer_count", columns, "NULL::BIGINT")},
            {_column_expr("has_accepted_answer", columns, "NULL::BOOLEAN")},
            {_column_expr("closed", columns, "NULL::BOOLEAN")},
            {_column_expr("tags", columns, "NULL::VARCHAR[]")},
            {_column_expr("event_count", columns, "NULL::BIGINT")},
            {_column_expr("session_duration_seconds", columns, "NULL::BIGINT")},
            {_column_expr("source_ip", columns, "NULL::VARCHAR")},
            {_column_expr("rule_id", columns, "NULL::VARCHAR")},
            {_column_expr("rule_level", columns, "NULL::VARCHAR")},
            {_column_expr("cve_id", columns, "NULL::VARCHAR")},
            {_column_expr("severity", columns, "NULL::VARCHAR")},
            {_column_expr("cvss_score", columns, "NULL::DOUBLE")},
            {_source_family_case()} AS source_family
        FROM {corpus_expr}
    """)


def _column_expr(column: str, columns: set[str], fallback: str) -> str:
    if column in columns:
        return column
    return f"{fallback} AS {column}"


def _source_family_case() -> str:
    return """
        CASE
            WHEN source_id IN ('nvd', 'cisa-kev', 'github-advisory') THEN 'vulnerability'
            WHEN source_id IN ('mitre-attack', 'mitre-cwe', 'capec', 'bron') THEN 'knowledge-base'
            WHEN source_id = 'sigma' THEN 'detection-rules'
            WHEN source_id = 'cloudtrail-flaws' THEN 'logs'
            WHEN source_id = 'arxiv' THEN 'academic-papers'
            WHEN source_id = 'youtube-transcripts' THEN 'transcripts'
            WHEN source_id = 'stackoverflow' THEN 'qa-stackoverflow'
            WHEN source_id LIKE 'stackexchange-%' THEN 'qa-stackexchange'
            WHEN source_id LIKE 'reddit-%' THEN 'qa-reddit'
            ELSE 'other'
        END
    """


def _build_reports(limit: int, preview_chars: int) -> list[SampleReport]:
    preview = f"left(coalesce(content, ''), {preview_chars}) AS preview"
    common = f"""
        source_id,
        record_id,
        content_length,
        title,
        source_url,
        published_at,
        {preview}
    """
    qa_cols = f"""
        source_id,
        record_id,
        content_length,
        score,
        answer_count,
        has_accepted_answer,
        closed,
        tags,
        title,
        source_url,
        published_at,
        {preview}
    """

    return [
        SampleReport(
            name="candidate_counts",
            title="Candidate Counts",
            description="How many records fall into each V2 review bucket.",
            sql="""
                SELECT * FROM (
                    SELECT 'arxiv_over_32000_tokens' AS bucket, count(*) AS records, coalesce(sum(content_length), 0)::BIGINT AS tokens
                    FROM sample_base WHERE source_id = 'arxiv' AND content_length > 32000
                    UNION ALL
                    SELECT 'arxiv_over_64000_tokens', count(*), coalesce(sum(content_length), 0)::BIGINT
                    FROM sample_base WHERE source_id = 'arxiv' AND content_length > 64000
                    UNION ALL
                    SELECT 'cloudtrail_over_16000_tokens', count(*), coalesce(sum(content_length), 0)::BIGINT
                    FROM sample_base WHERE source_id = 'cloudtrail-flaws' AND content_length > 16000
                    UNION ALL
                    SELECT 'cloudtrail_over_100000_tokens', count(*), coalesce(sum(content_length), 0)::BIGINT
                    FROM sample_base WHERE source_id = 'cloudtrail-flaws' AND content_length > 100000
                    UNION ALL
                    SELECT 'sigma_over_10000_tokens', count(*), coalesce(sum(content_length), 0)::BIGINT
                    FROM sample_base WHERE source_id = 'sigma' AND content_length > 10000
                    UNION ALL
                    SELECT 'qa_over_16000_tokens', count(*), coalesce(sum(content_length), 0)::BIGINT
                    FROM sample_base WHERE source_family LIKE 'qa-%' AND content_length > 16000
                    UNION ALL
                    SELECT 'qa_under_50_tokens', count(*), coalesce(sum(content_length), 0)::BIGINT
                    FROM sample_base WHERE source_family LIKE 'qa-%' AND content_length < 50
                    UNION ALL
                    SELECT 'closed_qa', count(*), coalesce(sum(content_length), 0)::BIGINT
                    FROM sample_base WHERE source_family IN ('qa-stackoverflow', 'qa-stackexchange') AND closed = true
                    UNION ALL
                    SELECT 'nvd_rejected', count(*), coalesce(sum(content_length), 0)::BIGINT
                    FROM sample_base WHERE source_id = 'nvd' AND content ILIKE 'Rejected reason:%'
                    UNION ALL
                    SELECT 'nvd_tiny_nonrejected_under_16_tokens', count(*), coalesce(sum(content_length), 0)::BIGINT
                    FROM sample_base
                    WHERE source_id = 'nvd'
                      AND content_length < 16
                      AND content NOT ILIKE 'Rejected reason:%'
                )
                ORDER BY records DESC
            """,
        ),
        SampleReport(
            name="arxiv_longest",
            title="arXiv Longest Records",
            description="Full-paper rows most likely to need section or token-window chunking.",
            sql=f"""
                SELECT {common}
                FROM sample_base
                WHERE source_id = 'arxiv'
                ORDER BY content_length DESC NULLS LAST
                LIMIT {limit}
            """,
        ),
        SampleReport(
            name="cloudtrail_longest",
            title="CloudTrail Longest Sessions",
            description="Already-sessionized records whose size may still require event-boundary chunking or capping.",
            sql=f"""
                SELECT
                    source_id,
                    record_id,
                    content_length,
                    event_count,
                    session_duration_seconds,
                    source_ip,
                    published_at,
                    {preview}
                FROM sample_base
                WHERE source_id = 'cloudtrail-flaws'
                ORDER BY content_length DESC NULLS LAST
                LIMIT {limit}
            """,
        ),
        SampleReport(
            name="sigma_longest",
            title="Sigma Longest Records",
            description="Detection-rule outliers; most Sigma records are short, so these should be inspected directly.",
            sql=f"""
                SELECT
                    source_id,
                    record_id,
                    content_length,
                    rule_id,
                    rule_level,
                    title,
                    source_url,
                    {preview}
                FROM sample_base
                WHERE source_id = 'sigma'
                ORDER BY content_length DESC NULLS LAST
                LIMIT {limit}
            """,
        ),
        SampleReport(
            name="qa_longest",
            title="Q&A Longest Records",
            description="Long Q&A/social threads that may need chunking or special handling.",
            sql=f"""
                SELECT {qa_cols}
                FROM sample_base
                WHERE source_family LIKE 'qa-%'
                ORDER BY content_length DESC NULLS LAST
                LIMIT {limit}
            """,
        ),
        SampleReport(
            name="qa_shortest",
            title="Q&A Shortest Records",
            description="Very short Q&A/social records for deciding source-family minimum-length rules.",
            sql=f"""
                SELECT {qa_cols}
                FROM sample_base
                WHERE source_family LIKE 'qa-%'
                ORDER BY content_length ASC NULLS FIRST, source_id, record_id
                LIMIT {limit}
            """,
        ),
        SampleReport(
            name="qa_closed_samples",
            title="Closed Q&A Samples",
            description="Closed Stack Overflow / Stack Exchange questions, split into low-score and high-score examples.",
            sql=f"""
                WITH low_score AS (
                    SELECT
                        'low_score_closed' AS bucket,
                        {qa_cols}
                    FROM sample_base
                    WHERE source_family IN ('qa-stackoverflow', 'qa-stackexchange')
                      AND closed = true
                    ORDER BY score ASC NULLS FIRST, content_length ASC NULLS FIRST, record_id
                    LIMIT {max(1, limit // 2)}
                ),
                high_score AS (
                    SELECT
                        'high_score_closed' AS bucket,
                        {qa_cols}
                    FROM sample_base
                    WHERE source_family IN ('qa-stackoverflow', 'qa-stackexchange')
                      AND closed = true
                    ORDER BY score DESC NULLS LAST, content_length DESC NULLS LAST, record_id
                    LIMIT {max(1, limit // 2)}
                )
                SELECT * FROM low_score
                UNION ALL
                SELECT * FROM high_score
            """,
        ),
        SampleReport(
            name="nvd_rejected_samples",
            title="NVD Rejected CVE Samples",
            description="Rejected/withdrawn CVE boilerplate that may warrant a source-specific V2 policy.",
            sql=f"""
                SELECT
                    source_id,
                    record_id,
                    cve_id,
                    content_length,
                    severity,
                    cvss_score,
                    source_url,
                    {preview}
                FROM sample_base
                WHERE source_id = 'nvd'
                  AND content ILIKE 'Rejected reason:%'
                ORDER BY content_length DESC NULLS LAST, record_id
                LIMIT {limit}
            """,
        ),
        SampleReport(
            name="nvd_tiny_nonrejected",
            title="NVD Tiny Non-Rejected Samples",
            description="Very short non-rejected NVD records; inspect before deciding if generic title-only records should remain.",
            sql=f"""
                SELECT
                    source_id,
                    record_id,
                    cve_id,
                    content_length,
                    severity,
                    cvss_score,
                    source_url,
                    {preview}
                FROM sample_base
                WHERE source_id = 'nvd'
                  AND content_length < 16
                  AND content NOT ILIKE 'Rejected reason:%'
                ORDER BY content_length ASC NULLS FIRST, record_id
                LIMIT {limit}
            """,
        ),
        SampleReport(
            name="youtube_longest",
            title="YouTube Transcript Longest Records",
            description="Transcript outliers if youtube-transcripts is present in the sampled dataset.",
            sql=f"""
                SELECT {common}
                FROM sample_base
                WHERE source_id = 'youtube-transcripts'
                ORDER BY content_length DESC NULLS LAST
                LIMIT {limit}
            """,
        ),
    ]


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


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
