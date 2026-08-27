#!/usr/bin/env python3
"""Build the first downstream training-clean corpus export.

This leaves normalized source Parquet untouched and writes a derived dataset for
training preparation. The v1 policy is intentionally small:

1. Drop invalid/empty text records.
2. Drop Q&A records with no answers/comments and non-positive score.
3. Drop exact duplicate content by keeping one row per content_hash.

Threshold-like quality decisions remain downstream of this export.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb


DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("data/training-clean-v1/normalized")
DEFAULT_REPORT_DIR = Path("reports/training-clean-v1")


@dataclass(frozen=True)
class BuildPaths:
    data_dir: Path
    output_dir: Path
    report_dir: Path
    temp_dir: Path


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = BuildPaths(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        temp_dir=args.temp_dir,
    )

    parquet_files = _find_source_parquet(paths.data_dir, paths.output_dir)
    if args.sources:
        selected_sources = set(args.sources)
        parquet_files = [
            path for path in parquet_files
            if _source_id_from_path(path) in selected_sources
        ]
    if not parquet_files:
        print(f"No normalized source Parquet files found under {paths.data_dir}")
        print("Expected layout: data/{source}/normalized/source_id={source}/*.parquet")
        return 1

    source_files = _source_file_map(parquet_files)

    if not args.dry_run:
        _prepare_output_dir(paths.output_dir, overwrite=args.overwrite)
    paths.temp_dir.mkdir(parents=True, exist_ok=True)
    paths.report_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    _configure_duckdb(
        con,
        temp_dir=paths.temp_dir,
    )
    metadata_expr = _read_parquet_expr(parquet_files, include_filename=True)
    available_columns = _available_columns(con, metadata_expr)
    metadata_sql = _metadata_select_sql(metadata_expr, available_columns)
    classified_sql = _classified_select_sql(metadata_sql)
    ranked_sql = _ranked_select_sql(classified_sql)

    input_summary = _fetch_one_dict(con, _input_summary_sql(classified_sql))
    drop_summary = _fetch_all_dicts(con, _drop_reason_summary_sql(classified_sql, ranked_sql))
    per_source_summary = _fetch_all_dicts(con, _per_source_summary_sql(classified_sql, ranked_sql))
    duplicate_examples = _fetch_all_dicts(con, _duplicate_examples_sql(ranked_sql, args.example_limit))
    zero_engagement_examples = _fetch_all_dicts(
        con,
        _zero_engagement_examples_sql(classified_sql, args.example_limit),
    )

    if args.dry_run:
        output_summary = {
            "output_records": input_summary["input_records"]
            - sum(row["dropped_records"] for row in drop_summary),
            "output_tokens": input_summary["input_tokens"]
            - sum(row["dropped_tokens"] for row in drop_summary),
        }
    else:
        _create_survivors_table(con, ranked_sql)
        _copy_clean_dataset_by_source(con, source_files, paths.output_dir)
        output_summary = _fetch_one_dict(con, _output_summary_sql(paths.output_dir))
        if not args.keep_temp:
            shutil.rmtree(paths.temp_dir, ignore_errors=True)

    manifest = _manifest(
        paths=paths,
        source_file_count=len(parquet_files),
        input_summary=input_summary,
        output_summary=output_summary,
        drop_summary=drop_summary,
        dry_run=args.dry_run,
    )

    _write_json(paths.report_dir / "manifest.json", manifest)
    _write_csv(paths.report_dir / "drop_reason_summary.csv", drop_summary)
    _write_csv(paths.report_dir / "per_source_summary.csv", per_source_summary)
    _write_csv(paths.report_dir / "duplicate_examples.csv", duplicate_examples)
    _write_csv(
        paths.report_dir / "invalid_content_examples.csv",
        _fetch_all_dicts(con, _invalid_content_examples_sql(classified_sql, args.example_limit)),
    )
    _write_csv(paths.report_dir / "zero_engagement_examples.csv", zero_engagement_examples)
    _write_markdown_report(
        paths.report_dir / "summary.md",
        manifest=manifest,
        drop_summary=drop_summary,
        per_source_summary=per_source_summary,
    )

    _print_summary(paths, manifest, dry_run=args.dry_run)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build downstream training-clean-v1 Parquet from normalized source Parquet.",
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
        help=f"Training-clean output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=f"Summary report directory (default: {DEFAULT_REPORT_DIR}).",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("data/training-clean-v1/tmp"),
        help="DuckDB spill/temp directory (default: data/training-clean-v1/tmp).",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        help="Optional source_id list for a smaller build/test run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing non-empty output directory before writing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute summaries without writing Parquet output.",
    )
    parser.add_argument(
        "--example-limit",
        type=int,
        default=50,
        help="Rows to write in example CSVs (default: 50).",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep DuckDB temp files after a successful write.",
    )
    return parser


def _find_source_parquet(data_dir: Path, output_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("**/normalized/source_id=*/*.parquet"))
    output_dir = output_dir.resolve()
    return [
        path for path in files
        if output_dir not in path.resolve().parents
        and not _is_appledouble_file(path)
    ]


def _is_appledouble_file(path: Path) -> bool:
    return path.name.startswith("._")


def _source_file_map(parquet_files: Sequence[Path]) -> dict[str, list[Path]]:
    files_by_source: dict[str, list[Path]] = {}
    for path in parquet_files:
        source_id = _source_id_from_path(path)
        files_by_source.setdefault(source_id, []).append(path)
    return dict(sorted(files_by_source.items()))


def _source_id_from_path(path: Path) -> str:
    for parent in path.parents:
        if parent.name.startswith("source_id="):
            return parent.name.removeprefix("source_id=")
    raise ValueError(f"Could not infer source_id from path: {path}")


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise SystemExit(
                f"Output directory is not empty: {output_dir}\n"
                "Use --overwrite to rebuild it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _configure_duckdb(
    con: duckdb.DuckDBPyConnection,
    temp_dir: Path,
) -> None:
    con.sql(f"SET temp_directory = {_sql_string(temp_dir.as_posix())}")
    con.sql("SET preserve_insertion_order = false")


def _create_survivors_table(
    con: duckdb.DuckDBPyConnection,
    ranked_sql: str,
) -> None:
    con.sql(f"""
        CREATE TEMP TABLE v1_survivors AS
        SELECT
            v1_source_file,
            source_id,
            record_id,
            content_hash
        FROM ({ranked_sql})
        WHERE v1_dedup_rank = 1
    """)


def _copy_clean_dataset_by_source(
    con: duckdb.DuckDBPyConnection,
    source_files: dict[str, list[Path]],
    output_dir: Path,
) -> None:
    for source_id, files in source_files.items():
        source_survivor_count = con.sql(f"""
            SELECT count(*)
            FROM v1_survivors
            WHERE source_id = {_sql_string(source_id)}
        """).fetchone()[0]
        if source_survivor_count == 0:
            print(f"Skipping {source_id}: no surviving records")
            continue

        source_output_dir = output_dir / f"source_id={source_id}"
        source_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Writing {source_id}: {source_survivor_count:,} records")

        for idx, path in enumerate(files):
            source_file = path.as_posix()
            file_survivor_count = con.sql(f"""
                SELECT count(*)
                FROM v1_survivors
                WHERE source_id = {_sql_string(source_id)}
                  AND v1_source_file = {_sql_string(source_file)}
            """).fetchone()[0]
            if file_survivor_count == 0:
                continue

            source_output_file = source_output_dir / f"part-{idx:05d}.parquet"
            source_expr = _read_parquet_expr([path])
            con.sql(f"""
                COPY (
                    SELECT c.*
                    FROM {source_expr} AS c
                    SEMI JOIN (
                        SELECT record_id, content_hash
                        FROM v1_survivors
                        WHERE source_id = {_sql_string(source_id)}
                          AND v1_source_file = {_sql_string(source_file)}
                    ) AS s
                    ON c.record_id = s.record_id
                       AND c.content_hash = s.content_hash
                )
                TO {_sql_string(source_output_file.as_posix())}
                (
                    FORMAT PARQUET,
                    COMPRESSION SNAPPY
                )
            """)


def _read_parquet_expr(parquet_files: Sequence[Path], include_filename: bool = False) -> str:
    file_list = ", ".join(_sql_string(path.as_posix()) for path in parquet_files)
    filename_arg = ", filename = true" if include_filename else ""
    return f"""
        read_parquet(
            [{file_list}],
            union_by_name = true,
            hive_partitioning = true
            {filename_arg}
        )
    """


def _available_columns(con: duckdb.DuckDBPyConnection, corpus_expr: str) -> set[str]:
    rows = con.sql(f"DESCRIBE SELECT * FROM {corpus_expr}").fetchall()
    return {row[0] for row in rows}


def _metadata_select_sql(corpus_expr: str, available_columns: set[str]) -> str:
    return f"""
        SELECT
            {_column_expr("source_id", available_columns, "'unknown'::VARCHAR")},
            {_column_expr("filename", available_columns, "NULL::VARCHAR", alias="v1_source_file")},
            {_column_expr("record_id", available_columns, "NULL::VARCHAR")},
            {_column_expr("content_hash", available_columns, "NULL::VARCHAR")},
            {_column_expr("content_length", available_columns, "0::BIGINT")},
            {_column_expr("score", available_columns, "NULL::BIGINT")},
            {_column_expr("answer_count", available_columns, "NULL::BIGINT")},
            {_column_expr("title", available_columns, "NULL::VARCHAR")},
            {_content_is_blank_expr(available_columns)}
        FROM {corpus_expr}
    """


def _column_expr(
    column: str,
    available_columns: set[str],
    fallback: str,
    alias: str | None = None,
) -> str:
    output_name = alias or column
    if column in available_columns:
        if output_name == column:
            return column
        return f"{column} AS {output_name}"
    return f"{fallback} AS {output_name}"


def _content_is_blank_expr(available_columns: set[str]) -> str:
    if "content" not in available_columns:
        return "true AS content_is_blank"
    return "(content IS NULL OR trim(content) = '') AS content_is_blank"


def _classified_select_sql(metadata_sql: str) -> str:
    return f"""
        SELECT
            *,
            CASE
                WHEN content_is_blank
                     OR content_length IS NULL
                     OR content_length <= 0
                     OR content_hash IS NULL
                     OR content_hash = ''
                    THEN 'invalid_empty_content'
                WHEN source_id = 'stackoverflow'
                     AND coalesce(answer_count, 0) = 0
                     AND coalesce(score, 0) <= 0
                    THEN 'qa_zero_engagement'
                WHEN source_id LIKE 'stackexchange-%'
                     AND coalesce(answer_count, 0) = 0
                     AND coalesce(score, 0) <= 0
                    THEN 'qa_zero_engagement'
                WHEN source_id LIKE 'reddit-%'
                     AND coalesce(answer_count, 0) = 0
                     AND coalesce(score, 0) <= 0
                    THEN 'qa_zero_engagement'
                ELSE NULL
            END AS v1_scope_drop_reason
        FROM ({metadata_sql})
    """


def _ranked_select_sql(classified_sql: str) -> str:
    return f"""
        WITH eligible AS (
            SELECT *
            FROM ({classified_sql})
            WHERE v1_scope_drop_reason IS NULL
        ),
        ranked_inputs AS (
            SELECT
                *,
                max(CASE WHEN source_id = 'cisa-kev' THEN 1 ELSE 0 END)
                    OVER (PARTITION BY content_hash) AS v1_group_has_cisa,
                max(CASE WHEN source_id = 'nvd' THEN 1 ELSE 0 END)
                    OVER (PARTITION BY content_hash) AS v1_group_has_nvd
            FROM eligible
        )
        SELECT
            *,
            row_number() OVER (
                PARTITION BY content_hash
                ORDER BY
                    CASE
                        WHEN v1_group_has_cisa = 1
                             AND v1_group_has_nvd = 1
                             AND source_id = 'cisa-kev'
                            THEN 0
                        ELSE 1
                    END ASC,
                    source_id ASC,
                    record_id ASC
            ) AS v1_dedup_rank
        FROM ranked_inputs
    """


def _input_summary_sql(classified_sql: str) -> str:
    return f"""
        SELECT
            count(*) AS input_records,
            coalesce(sum(content_length), 0)::BIGINT AS input_tokens,
            count(DISTINCT source_id) AS input_sources
        FROM ({classified_sql})
    """


def _drop_reason_summary_sql(classified_sql: str, ranked_sql: str) -> str:
    return f"""
        WITH drops AS (
            SELECT
                v1_scope_drop_reason AS drop_reason,
                count(*) AS dropped_records,
                coalesce(sum(content_length), 0)::BIGINT AS dropped_tokens
            FROM ({classified_sql})
            WHERE v1_scope_drop_reason IS NOT NULL
            GROUP BY v1_scope_drop_reason
            UNION ALL
            SELECT
                'exact_duplicate' AS drop_reason,
                count(*) AS dropped_records,
                coalesce(sum(content_length), 0)::BIGINT AS dropped_tokens
            FROM ({ranked_sql})
            WHERE v1_dedup_rank > 1
        )
        SELECT *
        FROM drops
        WHERE dropped_records > 0
        ORDER BY dropped_records DESC
    """


def _per_source_summary_sql(classified_sql: str, ranked_sql: str) -> str:
    return f"""
        WITH input AS (
            SELECT
                source_id,
                count(*) AS input_records,
                coalesce(sum(content_length), 0)::BIGINT AS input_tokens
            FROM ({classified_sql})
            GROUP BY source_id
        ),
        scope_drops AS (
            SELECT
                source_id,
                count(*) AS invalid_dropped_records,
                coalesce(sum(content_length), 0)::BIGINT AS invalid_dropped_tokens
            FROM ({classified_sql})
            WHERE v1_scope_drop_reason = 'invalid_empty_content'
            GROUP BY source_id
        ),
        zero_engagement_drops AS (
            SELECT
                source_id,
                count(*) AS zero_engagement_dropped_records,
                coalesce(sum(content_length), 0)::BIGINT AS zero_engagement_dropped_tokens
            FROM ({classified_sql})
            WHERE v1_scope_drop_reason = 'qa_zero_engagement'
            GROUP BY source_id
        ),
        ranked AS (
            SELECT *
            FROM ({ranked_sql})
        ),
        dedup_drops AS (
            SELECT
                source_id,
                count(*) AS exact_duplicate_dropped_records,
                coalesce(sum(content_length), 0)::BIGINT AS exact_duplicate_dropped_tokens
            FROM ranked
            WHERE v1_dedup_rank > 1
            GROUP BY source_id
        ),
        output AS (
            SELECT
                source_id,
                count(*) AS output_records,
                coalesce(sum(content_length), 0)::BIGINT AS output_tokens
            FROM ranked
            WHERE v1_dedup_rank = 1
            GROUP BY source_id
        )
        SELECT
            input.source_id,
            input.input_records,
            input.input_tokens,
            coalesce(scope_drops.invalid_dropped_records, 0) AS invalid_dropped_records,
            coalesce(scope_drops.invalid_dropped_tokens, 0) AS invalid_dropped_tokens,
            coalesce(zero_engagement_drops.zero_engagement_dropped_records, 0) AS zero_engagement_dropped_records,
            coalesce(zero_engagement_drops.zero_engagement_dropped_tokens, 0) AS zero_engagement_dropped_tokens,
            coalesce(dedup_drops.exact_duplicate_dropped_records, 0) AS exact_duplicate_dropped_records,
            coalesce(dedup_drops.exact_duplicate_dropped_tokens, 0) AS exact_duplicate_dropped_tokens,
            coalesce(output.output_records, 0) AS output_records,
            coalesce(output.output_tokens, 0) AS output_tokens
        FROM input
        LEFT JOIN scope_drops USING (source_id)
        LEFT JOIN zero_engagement_drops USING (source_id)
        LEFT JOIN dedup_drops USING (source_id)
        LEFT JOIN output USING (source_id)
        ORDER BY input_tokens DESC
    """


def _duplicate_examples_sql(ranked_sql: str, limit: int) -> str:
    return f"""
        SELECT
            content_hash,
            source_id,
            record_id,
            content_length,
            v1_dedup_rank
        FROM ({ranked_sql})
        WHERE v1_dedup_rank > 1
        ORDER BY v1_dedup_rank DESC, content_hash, source_id, record_id
        LIMIT {limit}
    """


def _zero_engagement_examples_sql(classified_sql: str, limit: int) -> str:
    return f"""
        SELECT
            source_id,
            record_id,
            score,
            answer_count,
            content_length,
            left(coalesce(title, ''), 160) AS title
        FROM ({classified_sql})
        WHERE v1_scope_drop_reason = 'qa_zero_engagement'
        ORDER BY source_id, content_length DESC NULLS LAST, record_id
        LIMIT {limit}
    """


def _invalid_content_examples_sql(classified_sql: str, limit: int) -> str:
    return f"""
        SELECT
            source_id,
            record_id,
            content_length,
            content_hash,
            content_is_blank,
            left(coalesce(title, ''), 160) AS title
        FROM ({classified_sql})
        WHERE v1_scope_drop_reason = 'invalid_empty_content'
        ORDER BY source_id, record_id
        LIMIT {limit}
    """


def _output_summary_sql(output_dir: Path) -> str:
    return f"""
        SELECT
            count(*) AS output_records,
            coalesce(sum(content_length), 0)::BIGINT AS output_tokens,
            count(DISTINCT source_id) AS output_sources
        FROM read_parquet(
            {_sql_string((output_dir / 'source_id=*/*.parquet').as_posix())},
            union_by_name = true,
            hive_partitioning = true
        )
    """


def _manifest(
    paths: BuildPaths,
    source_file_count: int,
    input_summary: dict[str, Any],
    output_summary: dict[str, Any],
    drop_summary: list[dict[str, Any]],
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "name": "training-clean-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "source_file_count": source_file_count,
        "data_dir": paths.data_dir.as_posix(),
        "output_dir": paths.output_dir.as_posix(),
        "report_dir": paths.report_dir.as_posix(),
        "rules": [
            {
                "name": "invalid_empty_content",
                "description": (
                    "Drop records with null/blank content, null/non-positive "
                    "content_length, or missing content_hash."
                ),
            },
            {
                "name": "qa_zero_engagement",
                "description": (
                    "Drop Stack Overflow, Stack Exchange, and Reddit records "
                    "where coalesce(answer_count, 0) = 0 and coalesce(score, 0) <= 0."
                ),
            },
            {
                "name": "exact_duplicate",
                "description": (
                    "Keep one record per content_hash. When a duplicate group contains "
                    "both cisa-kev and nvd, keep cisa-kev; otherwise order by source_id "
                    "then record_id."
                ),
            },
        ],
        "input_summary": input_summary,
        "drop_summary": drop_summary,
        "output_summary": output_summary,
    }


def _write_markdown_report(
    path: Path,
    manifest: dict[str, Any],
    drop_summary: list[dict[str, Any]],
    per_source_summary: list[dict[str, Any]],
) -> None:
    lines = [
        "# training-clean-v1",
        "",
        f"Created at: `{manifest['created_at']}`",
        f"Output: `{manifest['output_dir']}`",
        "",
        "## Totals",
        "",
        _markdown_table(
            ("metric", "records", "tokens"),
            (
                ("input", manifest["input_summary"]["input_records"], manifest["input_summary"]["input_tokens"]),
                ("output", manifest["output_summary"]["output_records"], manifest["output_summary"]["output_tokens"]),
            ),
        ),
        "",
        "## Drop Reasons",
        "",
        _markdown_table_from_dicts(drop_summary),
        "",
        "## Per Source",
        "",
        _markdown_table_from_dicts(per_source_summary),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _print_summary(paths: BuildPaths, manifest: dict[str, Any], dry_run: bool) -> None:
    prefix = "Dry run complete" if dry_run else "training-clean-v1 written"
    print(prefix)
    print(f"Input records:  {manifest['input_summary']['input_records']:,}")
    print(f"Output records: {manifest['output_summary']['output_records']:,}")
    print(f"Input tokens:   {manifest['input_summary']['input_tokens']:,}")
    print(f"Output tokens:  {manifest['output_summary']['output_tokens']:,}")
    print(f"Output dir:     {paths.output_dir}")
    print(f"Report dir:     {paths.report_dir}")


def _fetch_one_dict(con: duckdb.DuckDBPyConnection, sql: str) -> dict[str, Any]:
    rows = _fetch_all_dicts(con, sql)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row, got {len(rows)}")
    return rows[0]


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


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def _markdown_table_from_dicts(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(no rows)"
    columns = tuple(rows[0])
    values = [tuple(row[column] for column in columns) for row in rows]
    return _markdown_table(columns, values)


def _markdown_table(columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    rows = list(rows)
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
    text = str(value).replace("\n", " ").replace("|", "\\|")
    if len(text) > 160:
        return text[:157] + "..."
    return text


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
