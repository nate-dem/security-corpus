#!/usr/bin/env python3
"""Refresh only the arXiv partition in training-clean-v1.

This is an incremental helper for the common case where the only newly added
normalized data is arXiv. It leaves every non-arXiv v1 partition untouched and
rewrites only:

    data/training-clean-v1/normalized/source_id=arxiv/

The script still uses the existing v1 corpus as the dedup universe:

1. Drop invalid/empty arXiv records.
2. Drop arXiv records whose content_hash already exists in non-arXiv v1.
3. Deduplicate within the refreshed arXiv normalized input by content_hash.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import duckdb


DEFAULT_ARXIV_NORMALIZED_DIR = Path("data/arxiv/normalized")
DEFAULT_V1_NORMALIZED_DIR = Path("data/training-clean-v1/normalized")
DEFAULT_REPORT_DIR = Path("reports/training-clean-v1-arxiv-incremental")
DEFAULT_TEMP_DIR = Path("data/training-clean-v1/tmp/arxiv-incremental")


@dataclass(frozen=True)
class BuildPaths:
    arxiv_normalized_dir: Path
    v1_normalized_dir: Path
    report_dir: Path
    temp_dir: Path


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = BuildPaths(
        arxiv_normalized_dir=args.arxiv_normalized_dir,
        v1_normalized_dir=args.v1_normalized_dir,
        report_dir=args.report_dir,
        temp_dir=args.temp_dir,
    )

    arxiv_files = _find_parquet(paths.arxiv_normalized_dir, source_id="arxiv")
    if not arxiv_files:
        print(f"No normalized arXiv Parquet files found under {paths.arxiv_normalized_dir}")
        print("Expected layout: data/arxiv/normalized/source_id=arxiv/*.parquet")
        return 1

    non_arxiv_v1_files = [
        path for path in _find_parquet(paths.v1_normalized_dir)
        if _source_id_from_path(path) != "arxiv"
    ]
    if not non_arxiv_v1_files:
        print(f"No non-arXiv v1 Parquet files found under {paths.v1_normalized_dir}")
        print("This incremental updater needs an existing training-clean-v1 corpus.")
        return 1

    output_partition = paths.v1_normalized_dir / "source_id=arxiv"
    if output_partition.exists() and any(output_partition.iterdir()) and not args.overwrite and not args.dry_run:
        raise SystemExit(
            f"Output arXiv v1 partition is not empty: {output_partition}\n"
            "Use --overwrite to replace it."
        )

    paths.temp_dir.mkdir(parents=True, exist_ok=True)
    paths.report_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    _configure_duckdb(con, paths.temp_dir)

    arxiv_expr = _read_parquet_expr(arxiv_files, include_filename=True)
    non_arxiv_expr = _read_parquet_expr(non_arxiv_v1_files)
    available_columns = _available_columns(con, arxiv_expr)
    arxiv_sql = _arxiv_metadata_sql(arxiv_expr, available_columns)

    con.sql(f"""
        CREATE TEMP TABLE non_arxiv_v1_hashes AS
        SELECT DISTINCT content_hash
        FROM {non_arxiv_expr}
        WHERE content_hash IS NOT NULL
          AND content_hash != ''
    """)

    classified_sql = _classified_sql(arxiv_sql)
    ranked_sql = _ranked_sql(classified_sql)

    input_summary = _fetch_one_dict(con, _input_summary_sql(classified_sql))
    drop_summary = _fetch_all_dicts(con, _drop_summary_sql(classified_sql, ranked_sql))
    output_summary = _fetch_one_dict(con, _output_summary_sql(ranked_sql))

    if not args.dry_run:
        temp_output_root = paths.temp_dir / "output"
        temp_partition = temp_output_root / "source_id=arxiv"
        shutil.rmtree(temp_output_root, ignore_errors=True)
        temp_partition.mkdir(parents=True, exist_ok=True)
        _create_survivors_table(con, ranked_sql)
        _write_arxiv_partition(con, arxiv_files, temp_partition)

        if output_partition.exists():
            shutil.rmtree(output_partition)
        output_partition.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(temp_partition.as_posix(), output_partition.as_posix())
        shutil.rmtree(temp_output_root, ignore_errors=True)

        if not args.keep_temp:
            shutil.rmtree(paths.temp_dir, ignore_errors=True)

    manifest = {
        "name": "training-clean-v1-arxiv-incremental",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "arxiv_file_count": len(arxiv_files),
        "non_arxiv_v1_file_count": len(non_arxiv_v1_files),
        "arxiv_normalized_dir": paths.arxiv_normalized_dir.as_posix(),
        "v1_normalized_dir": paths.v1_normalized_dir.as_posix(),
        "output_partition": output_partition.as_posix(),
        "rules": [
            {
                "name": "invalid_empty_content",
                "description": "Drop arXiv records with null/blank content, non-positive content_length, or missing content_hash.",
            },
            {
                "name": "cross_source_duplicate",
                "description": "Drop arXiv records whose content_hash already exists in non-arXiv training-clean-v1.",
            },
            {
                "name": "exact_duplicate",
                "description": "Keep one arXiv record per content_hash, ordered by record_id.",
            },
        ],
        "input_summary": input_summary,
        "drop_summary": drop_summary,
        "output_summary": output_summary,
    }

    _write_json(paths.report_dir / "manifest.json", manifest)
    _write_csv(paths.report_dir / "drop_reason_summary.csv", drop_summary)
    _write_markdown_report(paths.report_dir / "summary.md", manifest, drop_summary)
    _print_summary(manifest, dry_run=args.dry_run)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Incrementally refresh the arXiv partition in training-clean-v1.",
    )
    parser.add_argument(
        "--arxiv-normalized-dir",
        type=Path,
        default=DEFAULT_ARXIV_NORMALIZED_DIR,
        help=f"Normalized arXiv source directory (default: {DEFAULT_ARXIV_NORMALIZED_DIR}).",
    )
    parser.add_argument(
        "--v1-normalized-dir",
        type=Path,
        default=DEFAULT_V1_NORMALIZED_DIR,
        help=f"Existing training-clean-v1 normalized directory (default: {DEFAULT_V1_NORMALIZED_DIR}).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=f"Report directory (default: {DEFAULT_REPORT_DIR}).",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=DEFAULT_TEMP_DIR,
        help=f"DuckDB/temp output directory (default: {DEFAULT_TEMP_DIR}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the existing training-clean-v1 arXiv partition.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute summaries without replacing the arXiv partition.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep DuckDB temp files after a successful write.",
    )
    return parser


def _find_parquet(root: Path, source_id: str | None = None) -> list[Path]:
    pattern = f"source_id={source_id}/*.parquet" if source_id else "source_id=*/*.parquet"
    return [
        path for path in sorted(root.glob(pattern))
        if not path.name.startswith("._")
    ]


def _source_id_from_path(path: Path) -> str:
    for parent in path.parents:
        if parent.name.startswith("source_id="):
            return parent.name.removeprefix("source_id=")
    raise ValueError(f"Could not infer source_id from path: {path}")


def _configure_duckdb(con: duckdb.DuckDBPyConnection, temp_dir: Path) -> None:
    con.sql(f"SET temp_directory = {_sql_string(temp_dir.as_posix())}")
    con.sql("SET preserve_insertion_order = false")


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


def _arxiv_metadata_sql(corpus_expr: str, available_columns: set[str]) -> str:
    return f"""
        SELECT
            {_column_expr("filename", available_columns, "NULL::VARCHAR", alias="v1_source_file")},
            {_column_expr("source_id", available_columns, "'arxiv'::VARCHAR")},
            {_column_expr("record_id", available_columns, "NULL::VARCHAR")},
            {_column_expr("content_hash", available_columns, "NULL::VARCHAR")},
            {_column_expr("content_length", available_columns, "0::BIGINT")},
            {_column_expr("title", available_columns, "NULL::VARCHAR")},
            {_content_is_blank_expr(available_columns)}
        FROM {corpus_expr}
        WHERE source_id = 'arxiv'
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


def _classified_sql(arxiv_sql: str) -> str:
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
                WHEN content_hash IN (SELECT content_hash FROM non_arxiv_v1_hashes)
                    THEN 'cross_source_duplicate'
                ELSE NULL
            END AS v1_arxiv_drop_reason
        FROM ({arxiv_sql})
    """


def _ranked_sql(classified_sql: str) -> str:
    return f"""
        SELECT
            *,
            row_number() OVER (
                PARTITION BY content_hash
                ORDER BY record_id ASC NULLS LAST
            ) AS v1_arxiv_dedup_rank
        FROM ({classified_sql})
        WHERE v1_arxiv_drop_reason IS NULL
    """


def _create_survivors_table(con: duckdb.DuckDBPyConnection, ranked_sql: str) -> None:
    con.sql(f"""
        CREATE TEMP TABLE arxiv_v1_survivors AS
        SELECT
            v1_source_file,
            record_id,
            content_hash
        FROM ({ranked_sql})
        WHERE v1_arxiv_dedup_rank = 1
    """)


def _write_arxiv_partition(
    con: duckdb.DuckDBPyConnection,
    arxiv_files: Sequence[Path],
    output_partition: Path,
) -> None:
    total = con.sql("SELECT count(*) FROM arxiv_v1_survivors").fetchone()[0]
    print(f"Writing arxiv: {total:,} records")
    for idx, path in enumerate(arxiv_files):
        source_file = path.as_posix()
        count = con.sql(f"""
            SELECT count(*)
            FROM arxiv_v1_survivors
            WHERE v1_source_file = {_sql_string(source_file)}
        """).fetchone()[0]
        if count == 0:
            continue

        source_expr = _read_parquet_expr([path])
        output_file = output_partition / f"part-{idx:05d}.parquet"
        con.sql(f"""
            COPY (
                SELECT c.*
                FROM {source_expr} AS c
                SEMI JOIN (
                    SELECT record_id, content_hash
                    FROM arxiv_v1_survivors
                    WHERE v1_source_file = {_sql_string(source_file)}
                ) AS s
                ON c.record_id = s.record_id
                   AND c.content_hash = s.content_hash
            )
            TO {_sql_string(output_file.as_posix())}
            (
                FORMAT PARQUET,
                COMPRESSION SNAPPY
            )
        """)


def _input_summary_sql(classified_sql: str) -> str:
    return f"""
        SELECT
            count(*) AS input_records,
            coalesce(sum(content_length), 0)::BIGINT AS input_tokens
        FROM ({classified_sql})
    """


def _drop_summary_sql(classified_sql: str, ranked_sql: str) -> str:
    return f"""
        WITH drops AS (
            SELECT
                v1_arxiv_drop_reason AS drop_reason,
                count(*) AS dropped_records,
                coalesce(sum(content_length), 0)::BIGINT AS dropped_tokens
            FROM ({classified_sql})
            WHERE v1_arxiv_drop_reason IS NOT NULL
            GROUP BY v1_arxiv_drop_reason
            UNION ALL
            SELECT
                'exact_duplicate' AS drop_reason,
                count(*) AS dropped_records,
                coalesce(sum(content_length), 0)::BIGINT AS dropped_tokens
            FROM ({ranked_sql})
            WHERE v1_arxiv_dedup_rank > 1
        )
        SELECT *
        FROM drops
        WHERE dropped_records > 0
        ORDER BY dropped_records DESC
    """


def _output_summary_sql(ranked_sql: str) -> str:
    return f"""
        SELECT
            count(*) AS output_records,
            coalesce(sum(content_length), 0)::BIGINT AS output_tokens
        FROM ({ranked_sql})
        WHERE v1_arxiv_dedup_rank = 1
    """


def _fetch_one_dict(con: duckdb.DuckDBPyConnection, sql: str) -> dict[str, Any]:
    result = con.sql(sql)
    row = result.fetchone()
    if row is None:
        return {}
    columns = [desc[0] for desc in result.description]
    return dict(zip(columns, row))


def _fetch_all_dicts(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    result = con.sql(sql)
    columns = [desc[0] for desc in result.description]
    return [dict(zip(columns, row)) for row in result.fetchall()]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_report(
    path: Path,
    manifest: dict[str, Any],
    drop_summary: list[dict[str, Any]],
) -> None:
    lines = [
        "# training-clean-v1 arXiv incremental refresh",
        "",
        f"Created at: `{manifest['created_at']}`",
        f"arXiv normalized input: `{manifest['arxiv_normalized_dir']}`",
        f"v1 output partition: `{manifest['output_partition']}`",
        "",
        "## Totals",
        "",
        _markdown_table(
            ("metric", "records", "tokens"),
            (
                (
                    "input_arxiv_normalized",
                    manifest["input_summary"]["input_records"],
                    manifest["input_summary"]["input_tokens"],
                ),
                (
                    "output_arxiv_v1",
                    manifest["output_summary"]["output_records"],
                    manifest["output_summary"]["output_tokens"],
                ),
            ),
        ),
        "",
        "## Drop Reasons",
        "",
        _markdown_table_from_dicts(drop_summary),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(_format_markdown_value(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def _markdown_table_from_dicts(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    headers = list(rows[0].keys())
    return _markdown_table(headers, [[row.get(header, "") for header in headers] for row in rows])


def _format_markdown_value(value: Any) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _print_summary(manifest: dict[str, Any], *, dry_run: bool) -> None:
    action = "training-clean-v1 arXiv dry run complete" if dry_run else "training-clean-v1 arXiv partition refreshed"
    print(action)
    print(f"Input records:  {manifest['input_summary']['input_records']:,}")
    print(f"Output records: {manifest['output_summary']['output_records']:,}")
    print(f"Input tokens:   {manifest['input_summary']['input_tokens']:,}")
    print(f"Output tokens:  {manifest['output_summary']['output_tokens']:,}")
    print(f"Output dir:     {manifest['output_partition']}")


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
