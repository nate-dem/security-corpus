#!/usr/bin/env python3
"""Audit normalized corpus Parquet output.

This script is intentionally downstream of ingestion. It does not decide which
records are good enough for a training mixture; it measures the normalized
corpus so those decisions can be made from source-specific evidence.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb


DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("reports/normalized_audit")

BASE_REQUIRED_COLUMNS = (
    "record_id",
    "source_id",
    "source_record_id",
    "content",
    "content_length",
    "content_hash",
    "license",
    "ingested_at",
)


@dataclass(frozen=True)
class QueryReport:
    name: str
    title: str
    sql: str
    description: str | None = None
    write_csv: bool = True


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    parquet_files = _find_parquet_files(args.data_dir)
    if not parquet_files:
        print(f"No normalized Parquet files found under {args.data_dir}")
        print("Expected layout: data/{source}/normalized/source_id={source}/*.parquet")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    _create_corpus_view(con, parquet_files)

    available_columns = _get_columns(con)
    reports = _build_reports(args.example_limit, available_columns)
    report_lines: list[str] = []

    header = f"Normalized Corpus Audit ({len(parquet_files)} Parquet files)"
    _emit_heading(header, report_lines)
    _emit_line(f"Data directory: {args.data_dir}", report_lines)
    _emit_line(f"Output directory: {args.output_dir}", report_lines)

    source_counts = _source_file_counts(parquet_files)
    _emit_table(
        "Parquet Files By Source",
        ("source_id", "files"),
        sorted(source_counts.items()),
        report_lines,
    )
    _write_csv(args.output_dir / "parquet_files_by_source.csv", ("source_id", "files"), sorted(source_counts.items()))

    for report in reports:
        columns, rows = _run_query(con, report.sql)
        _emit_table(report.title, columns, rows, report_lines, report.description)
        if report.write_csv:
            _write_csv(args.output_dir / f"{report.name}.csv", columns, rows)

    markdown_path = args.output_dir / "audit_report.md"
    markdown_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nWrote audit report: {markdown_path}")
    print(f"Wrote CSV tables to: {args.output_dir}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit normalized corpus Parquet files for cleaning/filtering decisions.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Corpus data directory (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for audit CSV/Markdown output (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--example-limit",
        type=int,
        default=20,
        help="Maximum rows for example/outlier tables (default: 20).",
    )
    return parser


def _find_parquet_files(data_dir: Path) -> list[Path]:
    return [
        path for path in sorted(data_dir.glob("**/normalized/source_id=*/*.parquet"))
        if not path.name.startswith("._")
    ]


def _create_corpus_view(con: duckdb.DuckDBPyConnection, parquet_files: Sequence[Path]) -> None:
    file_list = ", ".join(_sql_string(path.as_posix()) for path in parquet_files)
    con.sql(f"""
        CREATE VIEW corpus AS
        SELECT *
        FROM read_parquet(
            [{file_list}],
            union_by_name = true,
            hive_partitioning = true
        )
    """)


def _get_columns(con: duckdb.DuckDBPyConnection) -> set[str]:
    rows = con.sql("DESCRIBE corpus").fetchall()
    return {row[0] for row in rows}


def _build_reports(example_limit: int, available_columns: set[str]) -> list[QueryReport]:
    required_presence_exprs = []
    for column in BASE_REQUIRED_COLUMNS:
        if column in available_columns:
            required_presence_exprs.append(
                f"count(*) FILTER (WHERE {column} IS NULL) AS null_{column}"
            )

    content_expr = (
        "count(*) FILTER (WHERE content IS NULL OR trim(content) = '') AS blank_content"
        if "content" in available_columns
        else "0 AS blank_content"
    )
    zero_length_expr = (
        "count(*) FILTER (WHERE content_length IS NULL OR content_length <= 0) AS non_positive_length"
        if "content_length" in available_columns
        else "0 AS non_positive_length"
    )
    missing_hash_expr = (
        "count(*) FILTER (WHERE content_hash IS NULL OR content_hash = '') AS missing_hash"
        if "content_hash" in available_columns
        else "0 AS missing_hash"
    )
    missing_license_expr = (
        "count(*) FILTER (WHERE license IS NULL OR license = '') AS missing_license"
        if "license" in available_columns
        else "0 AS missing_license"
    )
    title_expr = "left(coalesce(title, ''), 120) AS title" if "title" in available_columns else "'' AS title"
    published_year_expr = (
        "min(year(published_at)) AS first_year, max(year(published_at)) AS last_year"
        if "published_at" in available_columns
        else "NULL AS first_year, NULL AS last_year"
    )

    reports = [
        QueryReport(
            name="family_summary",
            title="Source Family Summary",
            description="Token totals here are measured from the normalized content_length column.",
            sql=f"""
                SELECT
                    {_source_family_case()} AS source_family,
                    count(DISTINCT source_id) AS sources,
                    count(*) AS records,
                    sum(content_length)::BIGINT AS tokens,
                    percentile_cont(0.50) WITHIN GROUP (ORDER BY content_length)::INT AS p50_tokens,
                    percentile_cont(0.95) WITHIN GROUP (ORDER BY content_length)::INT AS p95_tokens,
                    max(content_length)::INT AS max_tokens
                FROM corpus
                GROUP BY source_family
                ORDER BY tokens DESC NULLS LAST
            """,
        ),
        QueryReport(
            name="source_summary",
            title="Per-Source Summary",
            sql=f"""
                SELECT
                    source_id,
                    {_source_family_case()} AS source_family,
                    count(*) AS records,
                    sum(content_length)::BIGINT AS tokens,
                    percentile_cont(0.05) WITHIN GROUP (ORDER BY content_length)::INT AS p05_tokens,
                    percentile_cont(0.50) WITHIN GROUP (ORDER BY content_length)::INT AS p50_tokens,
                    percentile_cont(0.95) WITHIN GROUP (ORDER BY content_length)::INT AS p95_tokens,
                    percentile_cont(0.99) WITHIN GROUP (ORDER BY content_length)::INT AS p99_tokens,
                    max(content_length)::INT AS max_tokens,
                    {missing_hash_expr},
                    {missing_license_expr},
                    {content_expr},
                    {zero_length_expr},
                    {published_year_expr}
                FROM corpus
                GROUP BY source_id, source_family
                ORDER BY tokens DESC NULLS LAST
            """,
        ),
        QueryReport(
            name="required_field_nulls",
            title="Required Field Nulls",
            description="These are corpus-integrity checks, not quality thresholds.",
            sql=f"""
                SELECT
                    source_id,
                    count(*) AS records,
                    {", ".join(required_presence_exprs) if required_presence_exprs else "0 AS no_required_columns_found"}
                FROM corpus
                GROUP BY source_id
                ORDER BY source_id
            """,
        ),
        QueryReport(
            name="duplicate_record_ids",
            title="Duplicate Record IDs",
            description="Duplicate record_id values are usually ingestion bugs because IDs are namespaced by source.",
            sql="""
                WITH duplicated_ids AS (
                    SELECT
                        record_id,
                        min(source_id) AS example_source_id,
                        count(*) AS copies
                    FROM corpus
                    WHERE record_id IS NOT NULL
                    GROUP BY record_id
                    HAVING count(*) > 1
                )
                SELECT
                    example_source_id AS source_id,
                    count(*) AS duplicate_record_ids,
                    sum(copies - 1)::BIGINT AS extra_records
                FROM duplicated_ids
                GROUP BY example_source_id
                ORDER BY extra_records DESC, duplicate_record_ids DESC
            """,
        ),
        QueryReport(
            name="duplicate_content_hashes_by_source",
            title="Exact Duplicate Content Hashes By Source",
            description="Some duplicate hashes are valid repeated text for different entities; review examples before dropping.",
            sql="""
                SELECT
                    source_id,
                    count(*) AS records,
                    count(DISTINCT content_hash) AS unique_hashes,
                    (count(*) - count(DISTINCT content_hash))::BIGINT AS duplicate_extra_records,
                    round(100.0 * (count(*) - count(DISTINCT content_hash)) / count(*), 3) AS pct_duplicate
                FROM corpus
                WHERE content_hash IS NOT NULL
                GROUP BY source_id
                HAVING count(*) - count(DISTINCT content_hash) > 0
                ORDER BY duplicate_extra_records DESC, pct_duplicate DESC
            """,
        ),
        QueryReport(
            name="cross_source_duplicate_hashes",
            title="Cross-Source Exact Duplicate Hashes",
            sql="""
                WITH hashes AS (
                    SELECT
                        content_hash,
                        count(*) AS records,
                        count(DISTINCT source_id) AS sources
                    FROM corpus
                    WHERE content_hash IS NOT NULL
                    GROUP BY content_hash
                )
                SELECT
                    count(*) FILTER (WHERE records > 1) AS duplicate_hashes,
                    sum(records - 1) FILTER (WHERE records > 1)::BIGINT AS duplicate_extra_records,
                    count(*) FILTER (WHERE sources > 1) AS cross_source_hashes
                FROM hashes
            """,
        ),
        QueryReport(
            name="duplicate_content_examples",
            title="Duplicate Content Examples",
            write_csv=True,
            sql=f"""
                WITH duplicate_groups AS (
                    SELECT
                        content_hash,
                        count(*) AS copies,
                        count(DISTINCT source_id) AS sources,
                        min(content_length) AS min_tokens,
                        max(content_length) AS max_tokens
                    FROM corpus
                    WHERE content_hash IS NOT NULL
                    GROUP BY content_hash
                    HAVING count(*) > 1
                    ORDER BY copies DESC, sources DESC, max_tokens DESC
                    LIMIT {example_limit}
                ),
                examples AS (
                    SELECT
                        g.content_hash,
                        g.copies,
                        g.sources,
                        g.min_tokens,
                        g.max_tokens,
                        c.record_id,
                        row_number() OVER (
                            PARTITION BY g.content_hash
                            ORDER BY c.source_id, c.record_id
                        ) AS rn
                    FROM duplicate_groups g
                    JOIN corpus c USING (content_hash)
                )
                SELECT
                    content_hash,
                    copies,
                    sources,
                    min_tokens,
                    max_tokens,
                    string_agg(record_id, ' | ' ORDER BY record_id) FILTER (WHERE rn <= 5) AS example_record_ids
                FROM examples
                GROUP BY content_hash, copies, sources, min_tokens, max_tokens
                ORDER BY copies DESC, sources DESC, max_tokens DESC
            """,
        ),
        QueryReport(
            name="longest_records",
            title="Longest Records",
            description="These are the first candidates for chunking or capping policy review.",
            sql=f"""
                SELECT
                    source_id,
                    record_id,
                    content_length AS tokens,
                    {title_expr}
                FROM corpus
                ORDER BY content_length DESC NULLS LAST
                LIMIT {example_limit}
            """,
        ),
        QueryReport(
            name="shortest_records",
            title="Shortest Records",
            description="Short records are not automatically bad; structured sources often contain valuable compact records.",
            sql=f"""
                SELECT
                    source_id,
                    record_id,
                    content_length AS tokens,
                    {title_expr}
                FROM corpus
                ORDER BY content_length ASC NULLS FIRST
                LIMIT {example_limit}
            """,
        ),
    ]

    if "score" in available_columns:
        reports.append(_qa_report())
    if "cvss_score" in available_columns or "severity" in available_columns:
        reports.append(_vulnerability_report(available_columns))
    if "event_count" in available_columns:
        reports.append(_cloudtrail_report())
    if "license" in available_columns:
        reports.append(_license_report())

    return reports


def _qa_report() -> QueryReport:
    return QueryReport(
        name="qa_quality_signals",
        title="Q&A Quality Signals",
        description="Use these as per-source evidence. Do not reuse one global score threshold across platforms.",
        sql=f"""
            SELECT
                source_id,
                {_source_family_case()} AS source_family,
                count(*) AS records,
                sum(content_length)::BIGINT AS tokens,
                percentile_cont(0.10) WITHIN GROUP (ORDER BY score)::INT AS p10_score,
                percentile_cont(0.50) WITHIN GROUP (ORDER BY score)::INT AS p50_score,
                percentile_cont(0.90) WITHIN GROUP (ORDER BY score)::INT AS p90_score,
                avg(answer_count)::DOUBLE AS avg_answers,
                round(100.0 * count(*) FILTER (
                    WHERE score <= 0 AND coalesce(answer_count, 0) = 0
                ) / count(*), 2) AS pct_zero_engagement,
                round(100.0 * count(*) FILTER (
                    WHERE coalesce(answer_count, 0) = 0
                ) / count(*), 2) AS pct_no_answers,
                round(100.0 * count(*) FILTER (
                    WHERE closed = true
                ) / count(*), 2) AS pct_closed,
                round(100.0 * count(*) FILTER (
                    WHERE content_length < 50
                ) / count(*), 2) AS pct_under_50_tokens
            FROM corpus
            WHERE score IS NOT NULL
            GROUP BY source_id, source_family
            ORDER BY tokens DESC NULLS LAST
        """,
    )


def _vulnerability_report(available_columns: set[str]) -> QueryReport:
    cvss_exprs = (
        """
        count(cvss_score) AS records_with_cvss,
        percentile_cont(0.50) WITHIN GROUP (ORDER BY cvss_score)::DOUBLE AS p50_cvss,
        percentile_cont(0.90) WITHIN GROUP (ORDER BY cvss_score)::DOUBLE AS p90_cvss,
        """
        if "cvss_score" in available_columns
        else "0 AS records_with_cvss, NULL AS p50_cvss, NULL AS p90_cvss,"
    )
    severity_expr = (
        "count(*) FILTER (WHERE severity IS NULL OR severity = '') AS missing_severity,"
        if "severity" in available_columns
        else "0 AS missing_severity,"
    )
    cve_expr = (
        "count(*) FILTER (WHERE cve_id IS NOT NULL AND cve_id <> '') AS records_with_cve,"
        if "cve_id" in available_columns
        else "0 AS records_with_cve,"
    )
    exploited_expr = (
        "count(*) FILTER (WHERE exploited_in_wild = true) AS exploited_in_wild_records"
        if "exploited_in_wild" in available_columns
        else "0 AS exploited_in_wild_records"
    )
    return QueryReport(
        name="vulnerability_signals",
        title="Vulnerability Source Signals",
        sql=f"""
            SELECT
                source_id,
                count(*) AS records,
                sum(content_length)::BIGINT AS tokens,
                {cve_expr}
                {severity_expr}
                {cvss_exprs}
                {exploited_expr}
            FROM corpus
            WHERE source_id IN ('nvd', 'cisa-kev', 'github-advisory')
            GROUP BY source_id
            ORDER BY records DESC
        """,
    )


def _cloudtrail_report() -> QueryReport:
    return QueryReport(
        name="cloudtrail_session_signals",
        title="CloudTrail Session Signals",
        description="Large sessions may need source-specific chunking before training.",
        sql="""
            SELECT
                source_id,
                count(*) AS sessions,
                sum(content_length)::BIGINT AS tokens,
                percentile_cont(0.50) WITHIN GROUP (ORDER BY content_length)::INT AS p50_tokens,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY content_length)::INT AS p95_tokens,
                percentile_cont(0.99) WITHIN GROUP (ORDER BY content_length)::INT AS p99_tokens,
                max(content_length)::INT AS max_tokens,
                percentile_cont(0.50) WITHIN GROUP (ORDER BY event_count)::INT AS p50_events,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY event_count)::INT AS p95_events,
                max(event_count)::INT AS max_events
            FROM corpus
            WHERE event_count IS NOT NULL
            GROUP BY source_id
            ORDER BY tokens DESC
        """,
    )


def _license_report() -> QueryReport:
    return QueryReport(
        name="license_summary",
        title="License Summary",
        sql="""
            SELECT
                license,
                count(DISTINCT source_id) AS sources,
                count(*) AS records,
                sum(content_length)::BIGINT AS tokens
            FROM corpus
            GROUP BY license
            ORDER BY tokens DESC NULLS LAST
        """,
    )


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


def _source_file_counts(parquet_files: Sequence[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in parquet_files:
        source_id = _source_id_from_path(path)
        counts[source_id] = counts.get(source_id, 0) + 1
    return counts


def _source_id_from_path(path: Path) -> str:
    for parent in path.parents:
        if parent.name.startswith("source_id="):
            return parent.name.removeprefix("source_id=")
    return "<unknown>"


def _run_query(con: duckdb.DuckDBPyConnection, sql: str) -> tuple[tuple[str, ...], list[tuple[Any, ...]]]:
    result = con.execute(sql)
    columns = tuple(desc[0] for desc in result.description)
    rows = result.fetchall()
    return columns, rows


def _emit_heading(title: str, report_lines: list[str]) -> None:
    text = f"# {title}"
    print(text)
    report_lines.append(text)
    report_lines.append("")


def _emit_line(line: str, report_lines: list[str]) -> None:
    print(line)
    report_lines.append(line)
    report_lines.append("")


def _emit_table(
    title: str,
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
    report_lines: list[str],
    description: str | None = None,
) -> None:
    rows = list(rows)
    print(f"\n## {title}")
    report_lines.append(f"## {title}")
    if description:
        print(description)
        report_lines.append("")
        report_lines.append(description)

    if not rows:
        print("(no rows)")
        report_lines.append("")
        report_lines.append("(no rows)")
        report_lines.append("")
        return

    table = _format_markdown_table(columns, rows)
    print(table)
    report_lines.append("")
    report_lines.append(table)
    report_lines.append("")


def _format_markdown_table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    formatted_rows = [[_format_cell(value) for value in row] for row in rows]
    formatted_columns = [str(column) for column in columns]
    widths = [
        max(len(formatted_columns[idx]), *(len(row[idx]) for row in formatted_rows))
        for idx in range(len(formatted_columns))
    ]
    lines = [
        "| " + " | ".join(column.ljust(widths[idx]) for idx, column in enumerate(formatted_columns)) + " |",
        "| " + " | ".join("-" * widths[idx] for idx in range(len(formatted_columns))) + " |",
    ]
    for row in formatted_rows:
        lines.append("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(widths))) + " |")
    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    text = str(value).replace("\n", " ").replace("|", "\\|")
    if len(text) > 140:
        return text[:137] + "..."
    return text


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
