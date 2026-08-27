#!/usr/bin/env python3
"""Build the second downstream training-clean corpus export.

This script reads training-clean-v1 Parquet and writes a derived
training-clean-v2 dataset. It leaves both normalized source Parquet and v1
Parquet untouched.

V2 policy implemented here:

1. Drop low-signal QA rows using source-specific medians:
   score <= source p50 score AND content_length <= source p50 token length.
2. For Stack Overflow / Stack Exchange, also drop unanswered rows with
   score <= source p50 score.
3. Drop rejected NVD CVE records.
4. Convert arXiv full-paper rows into cleaned semantic chunks using adaptive
   4k/8k/16k/32k/64k token buckets.
5. Drop extreme CloudTrail outliers and chunk oversized sessions by
   chronological event boundary.
6. Drop extreme Sigma outliers while keeping retained rules unchunked.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))

from ingest.utils import compute_content_hash, compute_token_count as _base_compute_token_count  # noqa: E402


def compute_token_count(content: str) -> int:
    """Return token count while treating tokenizer sentinel strings as normal text."""
    try:
        return _base_compute_token_count(content)
    except ValueError as exc:
        if "disallowed special token" not in str(exc):
            raise
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(content, disallowed_special=()))


DEFAULT_DATA_DIR = Path("data/training-clean-v1")
DEFAULT_OUTPUT_DIR = Path("data/training-clean-v2/normalized")
DEFAULT_REPORT_DIR = Path("reports/training-clean-v2")
DEFAULT_TEMP_DIR = Path("data/training-clean-v2/tmp")

QA_SOURCE_PREDICATE = """
    source_id = 'stackoverflow'
    OR source_id LIKE 'stackexchange-%'
    OR source_id LIKE 'reddit-%'
"""
SO_SE_SOURCE_PREDICATE = """
    source_id = 'stackoverflow'
    OR source_id LIKE 'stackexchange-%'
"""

SECTION_RE = re.compile(
    r"\\(?P<cmd>part|chapter|section|subsection|subsubsection|paragraph|subparagraph)"
    r"\*?(?:\[[^\]]*\])?\{(?P<title>[^{}]{1,300})\}",
    re.IGNORECASE,
)
ABSTRACT_RE = re.compile(r"\\begin\{abstract\}(?P<body>.*?)\\end\{abstract\}", re.IGNORECASE | re.DOTALL)
BEGIN_DOCUMENT_RE = re.compile(r"\\begin\{document\}", re.IGNORECASE)
END_DOCUMENT_RE = re.compile(r"\\end\{document\}", re.IGNORECASE)
THE_BIB_RE = re.compile(r"\\begin\{thebibliography\}.*", re.IGNORECASE | re.DOTALL)
BIBLIOGRAPHY_CMD_RE = re.compile(r"\\bibliography\{[^}]*\}.*", re.IGNORECASE | re.DOTALL)
TOKENIZER_SENTINEL_RE = re.compile(
    r"<\|(?:endoftext|endofprompt|fim_prefix|fim_middle|fim_suffix)\|>"
)
FIGURE_ENV_RE = re.compile(
    r"\\begin\{(?P<env>figure\*?|wrapfigure|sidewaysfigure)\}.*?\\end\{(?P=env)\}",
    re.IGNORECASE | re.DOTALL,
)
GRAPHICS_ENV_RE = re.compile(
    r"\\begin\{(?P<env>tikzpicture|axis|pgfpicture|pspicture|picture)\*?\}.*?"
    r"\\end\{(?P=env)\*?\}",
    re.IGNORECASE | re.DOTALL,
)
CAPTION_RE = re.compile(
    r"\\(?:caption|subcaption)(?:\[[^\]]*\])?\{(?P<body>(?:[^{}]|\{[^{}]*\})*)\}",
    re.IGNORECASE | re.DOTALL,
)
ADDPLOT_COORDINATES_RE = re.compile(
    r"\\addplot(?:\[[^\]]*\])?\s*coordinates\s*\{.*?\}\s*;",
    re.IGNORECASE | re.DOTALL,
)
COORDINATE_PAIR_RE = re.compile(r"\([-+]?\d+(?:\.\d+)?,\s*[-+]?\d+(?:\.\d+)?\)")
LOW_VALUE_SECTION_TITLES = {
    "acknowledgement",
    "acknowledgements",
    "acknowledgment",
    "acknowledgments",
    "bibliography",
    "references",
}


@dataclass(frozen=True)
class BuildPaths:
    data_dir: Path
    output_dir: Path
    report_dir: Path
    temp_dir: Path


@dataclass(frozen=True)
class ArxivPolicy:
    chunk_sizes: tuple[int, ...]


@dataclass(frozen=True)
class ArxivChunkPolicy:
    target_tokens: int
    min_tokens: int
    max_tokens: int
    hard_max_tokens: int


@dataclass(frozen=True)
class CloudTrailPolicy:
    drop_above_tokens: int
    trigger_tokens: int
    target_tokens: int
    max_tokens: int
    max_chunks_per_session: int


@dataclass(frozen=True)
class SigmaPolicy:
    drop_above_tokens: int


@dataclass
class Chunk:
    title: str
    text: str


@dataclass
class ArxivSummary:
    input_records: int = 0
    input_tokens: int = 0
    output_records: int = 0
    output_tokens: int = 0
    skipped_empty_records: int = 0


@dataclass
class CloudTrailSummary:
    input_records: int = 0
    input_tokens: int = 0
    output_records: int = 0
    output_tokens: int = 0
    dropped_extreme_records: int = 0
    dropped_extreme_tokens: int = 0
    chunked_sessions: int = 0
    capped_sessions: int = 0
    dropped_chunks_from_cap: int = 0
    oversized_events: int = 0
    unchunkable_sessions: int = 0


@dataclass
class SigmaSummary:
    input_records: int = 0
    input_tokens: int = 0
    output_records: int = 0
    output_tokens: int = 0
    chunked_rules: int = 0
    dropped_extreme_records: int = 0
    dropped_extreme_tokens: int = 0
    capped_rules: int = 0
    dropped_chunks_from_cap: int = 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = BuildPaths(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        temp_dir=args.temp_dir,
    )
    arxiv_chunk_sizes = _parse_token_sizes(args.arxiv_chunk_sizes)
    arxiv_policy = ArxivPolicy(
        chunk_sizes=arxiv_chunk_sizes,
    )
    cloudtrail_policy = CloudTrailPolicy(
        drop_above_tokens=args.cloudtrail_drop_above_tokens,
        trigger_tokens=args.cloudtrail_trigger_tokens,
        target_tokens=args.cloudtrail_target_tokens,
        max_tokens=args.cloudtrail_max_tokens,
        max_chunks_per_session=args.cloudtrail_max_chunks_per_session,
    )
    sigma_policy = SigmaPolicy(
        drop_above_tokens=args.sigma_drop_above_tokens,
    )
    if not arxiv_policy.chunk_sizes:
        raise SystemExit("--arxiv-chunk-sizes must include at least one positive integer")
    if tuple(sorted(arxiv_policy.chunk_sizes)) != arxiv_policy.chunk_sizes:
        raise SystemExit("--arxiv-chunk-sizes must be sorted ascending")
    if cloudtrail_policy.target_tokens > cloudtrail_policy.max_tokens:
        raise SystemExit("--cloudtrail-target-tokens must be <= --cloudtrail-max-tokens")
    if cloudtrail_policy.max_tokens > cloudtrail_policy.drop_above_tokens:
        raise SystemExit("--cloudtrail-max-tokens must be <= --cloudtrail-drop-above-tokens")
    if cloudtrail_policy.max_chunks_per_session < 0:
        raise SystemExit("--cloudtrail-max-chunks-per-session must be >= 0")

    parquet_files = _find_input_parquet(paths.data_dir, paths.output_dir)
    if args.sources:
        selected_sources = set(args.sources)
        parquet_files = [
            path for path in parquet_files
            if _source_id_from_path(path) in selected_sources
        ]
    if not parquet_files:
        print(f"No input Parquet files found under {paths.data_dir}")
        print("Expected layout like: data/training-clean-v1/normalized/source_id=*/part-*.parquet")
        return 1

    source_files = _source_file_map(parquet_files)

    if not args.dry_run:
        _prepare_output_dir(paths.output_dir, overwrite=args.overwrite)
    paths.temp_dir.mkdir(parents=True, exist_ok=True)
    paths.report_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    _configure_duckdb(con, paths.temp_dir)
    corpus_expr = _read_parquet_expr(parquet_files, include_filename=True)
    available_columns = _available_columns(con, corpus_expr)
    metadata_sql = _metadata_select_sql(corpus_expr, available_columns)
    qa_thresholds_sql = _qa_thresholds_sql(metadata_sql)
    classified_sql = _classified_select_sql(metadata_sql, qa_thresholds_sql)

    input_summary = _fetch_one_dict(con, _input_summary_sql(metadata_sql))
    qa_thresholds = _fetch_all_dicts(con, _qa_thresholds_output_sql(qa_thresholds_sql))
    drop_summary = _fetch_all_dicts(con, _drop_summary_sql(classified_sql))
    drop_examples = _fetch_all_dicts(con, _drop_examples_sql(classified_sql, args.example_limit))
    per_source_filter_summary = _fetch_all_dicts(con, _per_source_filter_summary_sql(classified_sql))

    arxiv_summary = ArxivSummary()
    cloudtrail_summary = CloudTrailSummary()
    sigma_summary = SigmaSummary()
    if not args.dry_run:
        _create_survivors_table(con, classified_sql)
        _copy_non_arxiv_survivors(con, source_files, paths.output_dir)

    arxiv_files = source_files.get("arxiv", [])
    if arxiv_files:
        arxiv_summary = _process_arxiv(
            arxiv_files,
            output_dir=paths.output_dir,
            policy=arxiv_policy,
            dry_run=args.dry_run,
            write_batch_size=args.write_batch_size,
        )

    cloudtrail_files = source_files.get("cloudtrail-flaws", [])
    if cloudtrail_files:
        cloudtrail_summary = _process_cloudtrail(
            cloudtrail_files,
            output_dir=paths.output_dir,
            policy=cloudtrail_policy,
            dry_run=args.dry_run,
            write_batch_size=args.write_batch_size,
        )

    if cloudtrail_summary.dropped_extreme_records:
        drop_summary.append({
            "drop_reason": "cloudtrail_extreme_length",
            "dropped_records": cloudtrail_summary.dropped_extreme_records,
            "dropped_tokens": cloudtrail_summary.dropped_extreme_tokens,
        })

    sigma_files = source_files.get("sigma", [])
    if sigma_files:
        sigma_summary = _process_sigma(
            sigma_files,
            output_dir=paths.output_dir,
            policy=sigma_policy,
            dry_run=args.dry_run,
            write_batch_size=args.write_batch_size,
        )

    if sigma_summary.dropped_extreme_records:
        drop_summary.append({
            "drop_reason": "sigma_extreme_length",
            "dropped_records": sigma_summary.dropped_extreme_records,
            "dropped_tokens": sigma_summary.dropped_extreme_tokens,
        })

    if args.dry_run:
        non_arxiv_output_summary = _fetch_one_dict(con, _non_arxiv_output_summary_sql(classified_sql))
        output_summary = {
            "output_records": (
                non_arxiv_output_summary["output_records"]
                + arxiv_summary.output_records
                + cloudtrail_summary.output_records
                + sigma_summary.output_records
            ),
            "output_tokens": (
                non_arxiv_output_summary["output_tokens"]
                + arxiv_summary.output_tokens
                + cloudtrail_summary.output_tokens
                + sigma_summary.output_tokens
            ),
            "output_sources": (
                non_arxiv_output_summary["output_sources"]
                + (1 if arxiv_summary.output_records else 0)
                + (1 if cloudtrail_summary.output_records else 0)
                + (1 if sigma_summary.output_records else 0)
            ),
        }
    else:
        output_summary = _fetch_one_dict(con, _output_summary_sql(paths.output_dir))
        if not args.keep_temp:
            shutil.rmtree(paths.temp_dir, ignore_errors=True)

    manifest = _manifest(
        paths=paths,
        input_summary=input_summary,
        output_summary=output_summary,
        drop_summary=drop_summary,
        source_file_count=len(parquet_files),
        dry_run=args.dry_run,
        arxiv_policy=arxiv_policy,
        arxiv_summary=arxiv_summary,
        cloudtrail_policy=cloudtrail_policy,
        cloudtrail_summary=cloudtrail_summary,
        sigma_policy=sigma_policy,
        sigma_summary=sigma_summary,
    )

    _write_json(paths.report_dir / "manifest.json", manifest)
    _write_csv(paths.report_dir / "qa_thresholds.csv", qa_thresholds)
    _write_csv(paths.report_dir / "drop_reason_summary.csv", drop_summary)
    _write_csv(paths.report_dir / "drop_examples.csv", drop_examples)
    _write_csv(paths.report_dir / "per_source_filter_summary.csv", per_source_filter_summary)
    _write_csv(paths.report_dir / "arxiv_chunk_summary.csv", [arxiv_summary.__dict__])
    _write_csv(paths.report_dir / "cloudtrail_chunk_summary.csv", [cloudtrail_summary.__dict__])
    _write_csv(paths.report_dir / "sigma_chunk_summary.csv", [sigma_summary.__dict__])
    _write_markdown_report(
        paths.report_dir / "summary.md",
        manifest=manifest,
        drop_summary=drop_summary,
        per_source_filter_summary=per_source_filter_summary,
    )
    _print_summary(paths, manifest, dry_run=args.dry_run)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build downstream training-clean-v2 Parquet from training-clean-v1 Parquet.",
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
        help=f"Training-clean-v2 output directory (default: {DEFAULT_OUTPUT_DIR}).",
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
        default=DEFAULT_TEMP_DIR,
        help=f"DuckDB spill/temp directory (default: {DEFAULT_TEMP_DIR}).",
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
        help="Rows to write in drop_examples.csv (default: 50).",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep DuckDB temp files after a successful write.",
    )
    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=1000,
        help="Rows per PyArrow write batch for transformed sources (default: 1000).",
    )
    parser.add_argument(
        "--arxiv-chunk-sizes",
        default="4000,8000,16000,32000,64000",
        help=(
            "Comma-separated adaptive arXiv chunk buckets in tokens. Each paper "
            "uses the smallest bucket that fits after cleaning; papers above the "
            "largest bucket are chunked at that maximum (default: "
            "4000,8000,16000,32000,64000)."
        ),
    )
    parser.add_argument(
        "--cloudtrail-drop-above-tokens",
        type=int,
        default=1_024_000,
        help="Drop CloudTrail sessions above this token count (default: 1024000).",
    )
    parser.add_argument(
        "--cloudtrail-trigger-tokens",
        type=int,
        default=64000,
        help="Chunk CloudTrail sessions above this token count (default: 64000).",
    )
    parser.add_argument(
        "--cloudtrail-target-tokens",
        type=int,
        default=64000,
        help="Target CloudTrail chunk size in tokens (default: 64000).",
    )
    parser.add_argument(
        "--cloudtrail-max-tokens",
        type=int,
        default=64000,
        help="Soft maximum CloudTrail chunk size in tokens (default: 64000).",
    )
    parser.add_argument(
        "--cloudtrail-max-chunks-per-session",
        type=int,
        default=0,
        help="Maximum chunks to keep per original CloudTrail session; 0 disables capping (default: 0).",
    )
    parser.add_argument(
        "--sigma-drop-above-tokens",
        type=int,
        default=16000,
        help="Drop Sigma records above this token count; retained records are not chunked (default: 16000).",
    )
    return parser


def _parse_token_sizes(value: str) -> tuple[int, ...]:
    sizes: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            size = int(part)
        except ValueError as exc:
            raise SystemExit(f"Invalid token size in --arxiv-chunk-sizes: {part!r}") from exc
        if size <= 0:
            raise SystemExit("--arxiv-chunk-sizes values must be positive integers")
        sizes.append(size)
    return tuple(sizes)


def _find_input_parquet(data_dir: Path, output_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("**/normalized/source_id=*/*.parquet"))
    output_dir = output_dir.resolve()
    return [
        path for path in files
        if output_dir not in path.resolve().parents
        and not path.name.startswith("._")
    ]


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


def _metadata_select_sql(corpus_expr: str, available_columns: set[str]) -> str:
    return f"""
        SELECT
            {_column_expr("filename", available_columns, "NULL::VARCHAR", alias="v2_source_file")},
            {_column_expr("source_id", available_columns, "'unknown'::VARCHAR")},
            {_column_expr("record_id", available_columns, "NULL::VARCHAR")},
            {_column_expr("content_hash", available_columns, "NULL::VARCHAR")},
            {_column_expr("content_length", available_columns, "0::BIGINT")},
            {_column_expr("score", available_columns, "NULL::BIGINT")},
            {_column_expr("answer_count", available_columns, "NULL::BIGINT")},
            {_column_expr("title", available_columns, "NULL::VARCHAR")},
            {_column_expr("content", available_columns, "NULL::VARCHAR")}
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


def _qa_thresholds_sql(metadata_sql: str) -> str:
    return f"""
        SELECT
            source_id,
            quantile_disc(coalesce(score, 0), 0.50)::BIGINT AS p50_score,
            quantile_disc(content_length, 0.50)::BIGINT AS p50_content_length
        FROM ({metadata_sql})
        WHERE ({QA_SOURCE_PREDICATE})
          AND content_length IS NOT NULL
          AND content_length > 0
        GROUP BY source_id
    """


def _classified_select_sql(metadata_sql: str, qa_thresholds_sql: str) -> str:
    return f"""
        WITH metadata AS (
            SELECT *
            FROM ({metadata_sql})
        ),
        qa_thresholds AS (
            SELECT *
            FROM ({qa_thresholds_sql})
        )
        SELECT
            metadata.*,
            qa_thresholds.p50_score,
            qa_thresholds.p50_content_length,
            CASE
                WHEN source_id = 'arxiv'
                    THEN NULL
                WHEN source_id = 'nvd'
                     AND content ILIKE 'Rejected reason:%'
                    THEN 'nvd_rejected'
                WHEN ({QA_SOURCE_PREDICATE})
                     AND coalesce(score, 0) <= qa_thresholds.p50_score
                     AND content_length <= qa_thresholds.p50_content_length
                    THEN 'qa_low_score_low_length'
                WHEN ({SO_SE_SOURCE_PREDICATE})
                     AND coalesce(score, 0) <= qa_thresholds.p50_score
                     AND coalesce(answer_count, 0) = 0
                    THEN 'qa_so_se_unanswered_low_score'
                ELSE NULL
            END AS v2_drop_reason
        FROM metadata
        LEFT JOIN qa_thresholds USING (source_id)
    """


def _create_survivors_table(con: duckdb.DuckDBPyConnection, classified_sql: str) -> None:
    con.sql(f"""
        CREATE TEMP TABLE v2_non_arxiv_survivors AS
        SELECT
            v2_source_file,
            source_id,
            record_id,
            content_hash
        FROM ({classified_sql})
        WHERE source_id NOT IN ('arxiv', 'cloudtrail-flaws', 'sigma')
          AND v2_drop_reason IS NULL
    """)


def _copy_non_arxiv_survivors(
    con: duckdb.DuckDBPyConnection,
    source_files: dict[str, list[Path]],
    output_dir: Path,
) -> None:
    for source_id, files in source_files.items():
        if source_id in {"arxiv", "cloudtrail-flaws", "sigma"}:
            continue
        source_survivor_count = con.sql(f"""
            SELECT count(*)
            FROM v2_non_arxiv_survivors
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
                FROM v2_non_arxiv_survivors
                WHERE source_id = {_sql_string(source_id)}
                  AND v2_source_file = {_sql_string(source_file)}
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
                        FROM v2_non_arxiv_survivors
                        WHERE source_id = {_sql_string(source_id)}
                          AND v2_source_file = {_sql_string(source_file)}
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


def _process_arxiv(
    arxiv_files: Sequence[Path],
    *,
    output_dir: Path,
    policy: ArxivPolicy,
    dry_run: bool,
    write_batch_size: int,
) -> ArxivSummary:
    summary = ArxivSummary()
    writer: pq.ParquetWriter | None = None
    output_path = output_dir / "source_id=arxiv" / "part-00000.parquet"
    buffer: list[dict[str, Any]] = []

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        for path in arxiv_files:
            parquet = pq.ParquetFile(path)
            schema = parquet.schema_arrow
            if writer is None and not dry_run:
                writer = pq.ParquetWriter(output_path, schema, compression="snappy")

            for batch in parquet.iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    summary.input_records += 1
                    summary.input_tokens += int(row.get("content_length") or 0)
                    chunk_rows = _arxiv_chunk_rows(row, policy)
                    if not chunk_rows:
                        summary.skipped_empty_records += 1
                        continue
                    summary.output_records += len(chunk_rows)
                    summary.output_tokens += sum(int(chunk["content_length"] or 0) for chunk in chunk_rows)
                    if dry_run:
                        continue
                    buffer.extend(chunk_rows)
                    if len(buffer) >= write_batch_size:
                        _write_arrow_rows(writer, schema, buffer)
                        buffer.clear()

        if writer is not None and buffer:
            _write_arrow_rows(writer, writer.schema, buffer)
    finally:
        if writer is not None:
            writer.close()

    if arxiv_files:
        action = "Would write" if dry_run else "Writing"
        print(f"{action} arxiv chunks: {summary.output_records:,} records")
    return summary


def _process_cloudtrail(
    cloudtrail_files: Sequence[Path],
    *,
    output_dir: Path,
    policy: CloudTrailPolicy,
    dry_run: bool,
    write_batch_size: int,
) -> CloudTrailSummary:
    summary = CloudTrailSummary()
    writer: pq.ParquetWriter | None = None
    output_path = output_dir / "source_id=cloudtrail-flaws" / "part-00000.parquet"
    buffer: list[dict[str, Any]] = []

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        for path in cloudtrail_files:
            parquet = pq.ParquetFile(path)
            schema = parquet.schema_arrow
            if writer is None and not dry_run:
                writer = pq.ParquetWriter(output_path, schema, compression="snappy")

            for batch in parquet.iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    summary.input_records += 1
                    summary.input_tokens += int(row.get("content_length") or 0)
                    chunk_rows, row_stats = _cloudtrail_chunk_rows(row, policy)
                    summary.output_records += len(chunk_rows)
                    summary.output_tokens += sum(int(chunk["content_length"] or 0) for chunk in chunk_rows)
                    summary.dropped_extreme_records += row_stats["dropped_extreme_records"]
                    summary.dropped_extreme_tokens += row_stats["dropped_extreme_tokens"]
                    summary.chunked_sessions += row_stats["chunked_sessions"]
                    summary.capped_sessions += row_stats["capped_sessions"]
                    summary.dropped_chunks_from_cap += row_stats["dropped_chunks_from_cap"]
                    summary.oversized_events += row_stats["oversized_events"]
                    summary.unchunkable_sessions += row_stats["unchunkable_sessions"]
                    if dry_run:
                        continue
                    buffer.extend(chunk_rows)
                    if len(buffer) >= write_batch_size:
                        _write_arrow_rows(writer, schema, buffer)
                        buffer.clear()

        if writer is not None and buffer:
            _write_arrow_rows(writer, writer.schema, buffer)
    finally:
        if writer is not None:
            writer.close()

    if cloudtrail_files:
        action = "Would write" if dry_run else "Writing"
        print(f"{action} cloudtrail-flaws: {summary.output_records:,} records")
    return summary


def _process_sigma(
    sigma_files: Sequence[Path],
    *,
    output_dir: Path,
    policy: SigmaPolicy,
    dry_run: bool,
    write_batch_size: int,
) -> SigmaSummary:
    summary = SigmaSummary()
    writer: pq.ParquetWriter | None = None
    output_path = output_dir / "source_id=sigma" / "part-00000.parquet"
    buffer: list[dict[str, Any]] = []

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        for path in sigma_files:
            parquet = pq.ParquetFile(path)
            schema = parquet.schema_arrow
            if writer is None and not dry_run:
                writer = pq.ParquetWriter(output_path, schema, compression="snappy")

            for batch in parquet.iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    summary.input_records += 1
                    summary.input_tokens += int(row.get("content_length") or 0)
                    chunk_rows, row_stats = _sigma_chunk_rows(row, policy)
                    summary.output_records += len(chunk_rows)
                    summary.output_tokens += sum(int(chunk["content_length"] or 0) for chunk in chunk_rows)
                    summary.chunked_rules += row_stats["chunked_rules"]
                    summary.dropped_extreme_records += row_stats["dropped_extreme_records"]
                    summary.dropped_extreme_tokens += row_stats["dropped_extreme_tokens"]
                    summary.capped_rules += row_stats["capped_rules"]
                    summary.dropped_chunks_from_cap += row_stats["dropped_chunks_from_cap"]
                    if dry_run:
                        continue
                    buffer.extend(chunk_rows)
                    if len(buffer) >= write_batch_size:
                        _write_arrow_rows(writer, schema, buffer)
                        buffer.clear()

        if writer is not None and buffer:
            _write_arrow_rows(writer, writer.schema, buffer)
    finally:
        if writer is not None:
            writer.close()

    if sigma_files:
        action = "Would write" if dry_run else "Writing"
        print(f"{action} sigma: {summary.output_records:,} records")
    return summary


def _write_arrow_rows(writer: pq.ParquetWriter | None, schema: pa.Schema, rows: list[dict[str, Any]]) -> None:
    if writer is None:
        raise RuntimeError("Parquet writer is not initialized")
    table = pa.Table.from_pylist(rows, schema=schema)
    writer.write_table(table)


def _arxiv_chunk_rows(row: dict[str, Any], policy: ArxivPolicy) -> list[dict[str, Any]]:
    content = row.get("content") or ""
    cleaned, abstract = _clean_arxiv_content(content)
    blocks = _arxiv_blocks(
        cleaned,
        title=row.get("title") or row.get("arxiv_id") or row.get("source_record_id") or "arXiv paper",
        abstract=abstract or row.get("abstract"),
    )
    chunk_policy = _adaptive_arxiv_chunk_policy(blocks, policy)
    chunks = _build_chunks(blocks, chunk_policy)
    chunks = _enforce_arxiv_hard_cap(row, chunks, chunk_policy)
    output_rows: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, start=1):
        chunk_content = _format_arxiv_chunk(row, chunk, idx, len(chunks))
        token_count = compute_token_count(chunk_content)
        if token_count <= 0:
            continue
        chunk_row = dict(row)
        original_source_record_id = row.get("source_record_id") or row.get("arxiv_id") or row.get("record_id")
        original_record_id = row.get("record_id") or f"arxiv:{original_source_record_id}"
        chunk_row["source_record_id"] = f"{original_source_record_id}:chunk-{idx:04d}"
        chunk_row["record_id"] = f"{original_record_id}:chunk-{idx:04d}"
        chunk_row["title"] = _chunk_title(row.get("title"), chunk.title, idx)
        chunk_row["content"] = chunk_content
        chunk_row["content_length"] = token_count
        chunk_row["content_hash"] = compute_content_hash(chunk_content)
        chunk_row["raw"] = None
        output_rows.append(chunk_row)
    return output_rows


def _adaptive_arxiv_chunk_policy(
    blocks: Sequence[Chunk],
    policy: ArxivPolicy,
) -> ArxivChunkPolicy:
    """Choose one arXiv chunk bucket for this paper without duplicating it."""
    paper_tokens = sum(compute_token_count(_format_block(block)) for block in blocks)
    bucket = policy.chunk_sizes[-1]
    for candidate in policy.chunk_sizes:
        if paper_tokens <= candidate:
            bucket = candidate
            break

    # Leave room for the chunk metadata header so final formatted chunks stay
    # inside the selected bucket.
    header_margin = min(512, max(128, bucket // 32))
    content_budget = max(1, bucket - header_margin)
    return ArxivChunkPolicy(
        target_tokens=content_budget,
        min_tokens=min(content_budget, max(1000, bucket // 4)),
        max_tokens=content_budget,
        hard_max_tokens=bucket,
    )


def _clean_arxiv_content(content: str) -> tuple[str, str | None]:
    text = _remove_tokenizer_sentinels(content).replace("\r\n", "\n").replace("\r", "\n")
    begin_match = BEGIN_DOCUMENT_RE.search(text)
    if begin_match:
        text = text[begin_match.end():]
    end_match = END_DOCUMENT_RE.search(text)
    if end_match:
        text = text[:end_match.start()]

    abstract = None
    abstract_match = ABSTRACT_RE.search(text)
    if abstract_match:
        abstract = _clean_latex_text(abstract_match.group("body"))
        text = text[:abstract_match.start()] + "\n" + text[abstract_match.end():]

    text = THE_BIB_RE.sub("", text)
    text = BIBLIOGRAPHY_CMD_RE.sub("", text)
    text = re.sub(
        r"\\appendix\b",
        lambda _match: "\n\\section{Appendix}\n",
        text,
        flags=re.IGNORECASE,
    )
    text = _remove_latex_rendering_artifacts(text)
    text = _remove_latex_noise_lines(text)
    text = _clean_latex_text(text)
    return text, abstract


def _remove_tokenizer_sentinels(text: str) -> str:
    return TOKENIZER_SENTINEL_RE.sub(" ", text)


def _remove_latex_rendering_artifacts(text: str) -> str:
    text = FIGURE_ENV_RE.sub(_figure_env_replacement, text)
    text = GRAPHICS_ENV_RE.sub(" ", text)
    text = ADDPLOT_COORDINATES_RE.sub(" ", text)
    return text


def _figure_env_replacement(match: re.Match[str]) -> str:
    captions = []
    for caption_match in CAPTION_RE.finditer(match.group(0)):
        caption = _clean_latex_text(caption_match.group("body"))
        caption = re.sub(r"[{}]", "", caption).strip()
        if caption:
            captions.append(f"Figure caption: {caption}")
    return "\n\n".join(captions)


def _remove_latex_noise_lines(text: str) -> str:
    drop_line_re = re.compile(
        r"^\s*\\("
        r"documentclass|usepackage|RequirePackage|newcommand|renewcommand|providecommand|"
        r"DeclareMathOperator|DeclareRobustCommand|DeclareUnicodeCharacter|newtheorem|"
        r"theoremstyle|setlength|addtolength|graphicspath|hypersetup|bibliographystyle|"
        r"title|author|date|maketitle|tableofcontents|pagestyle|thispagestyle"
        r")\b",
        re.IGNORECASE,
    )
    drop_only_re = re.compile(
        r"^\s*\\(label|vspace|hspace|smallskip|medskip|bigskip|clearpage|newpage|pagebreak)\b.*$",
        re.IGNORECASE,
    )
    render_command_re = re.compile(
        r"^\s*\\(?:"
        r"begin\{(?:tikzpicture|axis|pgfpicture|pspicture|picture)\*?\}|"
        r"end\{(?:tikzpicture|axis|pgfpicture|pspicture|picture)\*?\}|"
        r"(?:pgf[a-zA-Z]*|addplot|addlegendentry|draw|path|coordinate|fill|shade)\b"
        r")",
        re.IGNORECASE,
    )
    kept_lines = []
    for line in text.splitlines():
        stripped = re.sub(r"(?<!\\)%.*$", "", line).strip()
        if not stripped:
            kept_lines.append("")
            continue
        coordinate_pairs = len(COORDINATE_PAIR_RE.findall(stripped))
        if (
            drop_line_re.match(stripped)
            or drop_only_re.match(stripped)
            or render_command_re.match(stripped)
            or coordinate_pairs >= 20
            or stripped.count(r"\pgfqpoint") >= 3
        ):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def _clean_latex_text(text: str) -> str:
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    text = re.sub(r"\\(emph|textbf|textit|texttt|mathrm|mathbf)\{([^{}]*)\}", r"\2", text)
    text = re.sub(r"~", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _arxiv_blocks(cleaned: str, *, title: str, abstract: str | None) -> list[Chunk]:
    blocks: list[Chunk] = []
    if abstract and abstract.strip():
        blocks.append(Chunk(title="Abstract", text=abstract.strip()))

    matches = list(SECTION_RE.finditer(cleaned))
    if not matches:
        body = cleaned.strip()
        if body:
            blocks.append(Chunk(title="Body", text=body))
        return blocks

    front_matter = cleaned[:matches[0].start()].strip()
    front_matter = _strip_empty_latex_commands(front_matter)
    if front_matter:
        blocks.append(Chunk(title="Front Matter", text=front_matter))

    for idx, match in enumerate(matches):
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
        heading = _clean_heading(match.group("title"))
        body = cleaned[match.end():next_start].strip()
        body = _strip_empty_latex_commands(body)
        if not body:
            continue
        if _is_low_value_section(heading):
            continue
        blocks.append(Chunk(title=heading or title, text=body))
    return blocks


def _strip_empty_latex_commands(text: str) -> str:
    text = re.sub(r"^\s*\\(begin|end)\{document\}\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_heading(title: str) -> str:
    title = _clean_latex_text(title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def _is_low_value_section(title: str) -> bool:
    normalized = re.sub(r"[^a-z]+", " ", title.lower()).strip()
    return normalized in LOW_VALUE_SECTION_TITLES


def _build_chunks(blocks: Sequence[Chunk], policy: ArxivChunkPolicy) -> list[Chunk]:
    expanded_blocks: list[Chunk] = []
    for block in blocks:
        block_text = _format_block(block)
        block_tokens = compute_token_count(block_text)
        if block_tokens > policy.max_tokens:
            expanded_blocks.extend(_split_large_block(block, policy.max_tokens))
        else:
            expanded_blocks.append(block)

    chunks: list[Chunk] = []
    current_titles: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for block in expanded_blocks:
        block_text = _format_block(block)
        block_tokens = compute_token_count(block_text)
        should_flush = (
            current_parts
            and current_tokens >= policy.min_tokens
            and current_tokens + block_tokens > policy.target_tokens
        )
        if should_flush:
            chunks.append(Chunk(title=_join_titles(current_titles), text="\n\n".join(current_parts).strip()))
            current_titles = []
            current_parts = []
            current_tokens = 0

        current_titles.append(block.title)
        current_parts.append(block_text)
        current_tokens += block_tokens

        if current_tokens >= policy.max_tokens:
            chunks.append(Chunk(title=_join_titles(current_titles), text="\n\n".join(current_parts).strip()))
            current_titles = []
            current_parts = []
            current_tokens = 0

    if current_parts:
        tail = Chunk(title=_join_titles(current_titles), text="\n\n".join(current_parts).strip())
        if chunks and compute_token_count(tail.text) < policy.min_tokens:
            previous = chunks.pop()
            merged_title = _join_titles([previous.title, tail.title])
            merged_text = f"{previous.text}\n\n{tail.text}".strip()
            if compute_token_count(merged_text) <= policy.max_tokens:
                chunks.append(Chunk(title=merged_title, text=merged_text))
            else:
                chunks.append(previous)
                chunks.append(tail)
        else:
            chunks.append(tail)
    return [chunk for chunk in chunks if chunk.text.strip()]


def _enforce_arxiv_hard_cap(
    row: dict[str, Any],
    chunks: Sequence[Chunk],
    policy: ArxivChunkPolicy,
) -> list[Chunk]:
    capped_chunks: list[Chunk] = []
    for chunk in chunks:
        if _arxiv_chunk_fits_hard_cap(row, chunk, policy):
            capped_chunks.append(chunk)
            continue

        split_chunks = _split_large_block(chunk, policy.max_tokens)
        for split_chunk in split_chunks:
            if _arxiv_chunk_fits_hard_cap(row, split_chunk, policy):
                capped_chunks.append(split_chunk)
                continue

            for idx, piece in enumerate(_split_text_by_words(split_chunk.text, policy.max_tokens), start=1):
                piece_chunk = Chunk(title=f"{split_chunk.title} part {idx}", text=piece)
                if piece.strip():
                    capped_chunks.append(piece_chunk)
    return [chunk for chunk in capped_chunks if chunk.text.strip()]


def _arxiv_chunk_fits_hard_cap(
    row: dict[str, Any],
    chunk: Chunk,
    policy: ArxivChunkPolicy,
) -> bool:
    # Leave a small margin for final chunk numbering once the total chunk count is known.
    return compute_token_count(_format_arxiv_chunk(row, chunk, 1, 1)) <= policy.hard_max_tokens - 64


def _split_large_block(block: Chunk, max_tokens: int) -> list[Chunk]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", block.text) if paragraph.strip()]
    if not paragraphs:
        return []

    parts: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0
    part_idx = 1

    for paragraph in paragraphs:
        paragraph_tokens = compute_token_count(paragraph)
        if paragraph_tokens > max_tokens:
            if current:
                parts.append(Chunk(title=f"{block.title} part {part_idx}", text="\n\n".join(current)))
                part_idx += 1
                current = []
                current_tokens = 0
            for piece in _split_text_by_words(paragraph, max_tokens):
                parts.append(Chunk(title=f"{block.title} part {part_idx}", text=piece))
                part_idx += 1
            continue

        if current and current_tokens + paragraph_tokens > max_tokens:
            parts.append(Chunk(title=f"{block.title} part {part_idx}", text="\n\n".join(current)))
            part_idx += 1
            current = []
            current_tokens = 0

        current.append(paragraph)
        current_tokens += paragraph_tokens

    if current:
        parts.append(Chunk(title=f"{block.title} part {part_idx}", text="\n\n".join(current)))
    return parts


def _split_text_by_words(text: str, max_tokens: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        if compute_token_count(word) > max_tokens:
            if current:
                pieces.append(" ".join(current))
                current = []
            pieces.extend(_split_long_text_by_chars(word, max_tokens))
            continue

        candidate = " ".join([*current, word])
        if current and compute_token_count(candidate) > max_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _split_long_text_by_chars(text: str, max_tokens: int) -> list[str]:
    if max_tokens <= 0:
        return [text]
    token_count = compute_token_count(text)
    if token_count <= max_tokens:
        return [text]

    target_chars = max(1, int(len(text) * max_tokens / token_count * 0.85))
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + target_chars)
        piece = text[start:end]
        while compute_token_count(piece) > max_tokens and end > start + 1:
            end = start + max(1, (end - start) // 2)
            piece = text[start:end]
        pieces.append(piece)
        start = end
    return pieces


def _format_block(block: Chunk) -> str:
    if block.title:
        return f"## {block.title}\n\n{block.text.strip()}"
    return block.text.strip()


def _join_titles(titles: Sequence[str]) -> str:
    cleaned = []
    for title in titles:
        if title and title not in cleaned:
            cleaned.append(title)
    if not cleaned:
        return "Chunk"
    if len(cleaned) <= 2:
        return " / ".join(cleaned)
    return f"{cleaned[0]} / {cleaned[1]} / ..."


def _format_arxiv_chunk(row: dict[str, Any], chunk: Chunk, chunk_index: int, chunk_count: int) -> str:
    title = row.get("title") or row.get("arxiv_id") or "arXiv paper"
    arxiv_id = row.get("arxiv_id") or row.get("source_record_id") or ""
    header = [
        f"# {title}",
        "",
        f"arXiv ID: {arxiv_id}",
        f"Chunk: {chunk_index}/{chunk_count}",
        f"Section(s): {chunk.title}",
    ]
    return "\n".join(header).strip() + "\n\n" + chunk.text.strip()


def _chunk_title(title: str | None, chunk_title: str, chunk_index: int) -> str:
    base = title or "arXiv paper"
    return f"{base} - {chunk_title} (chunk {chunk_index})"


def _cloudtrail_chunk_rows(
    row: dict[str, Any],
    policy: CloudTrailPolicy,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = {
        "dropped_extreme_records": 0,
        "dropped_extreme_tokens": 0,
        "chunked_sessions": 0,
        "capped_sessions": 0,
        "dropped_chunks_from_cap": 0,
        "oversized_events": 0,
        "unchunkable_sessions": 0,
    }
    content_length = int(row.get("content_length") or 0)
    if content_length > policy.drop_above_tokens:
        stats["dropped_extreme_records"] = 1
        stats["dropped_extreme_tokens"] = content_length
        return [], stats
    if content_length <= policy.trigger_tokens:
        return [dict(row)], stats

    header, events = _split_cloudtrail_content(row.get("content") or "")
    if not events:
        stats["unchunkable_sessions"] = 1
        return [], stats

    stats["chunked_sessions"] = 1
    chunk_specs, oversized_events = _cloudtrail_event_chunks(header, events, row, policy)
    stats["oversized_events"] = oversized_events
    total_chunks = len(chunk_specs)
    selected_indexes = (
        _select_evenly_spaced_indexes(total_chunks, policy.max_chunks_per_session)
        if policy.max_chunks_per_session > 0
        else list(range(total_chunks))
    )
    if len(selected_indexes) < total_chunks:
        stats["capped_sessions"] = 1
        stats["dropped_chunks_from_cap"] = total_chunks - len(selected_indexes)

    rows: list[dict[str, Any]] = []
    for output_idx, chunk_idx in enumerate(selected_indexes, start=1):
        chunk_events, start_event_idx, end_event_idx = chunk_specs[chunk_idx]
        chunk_content = _format_cloudtrail_chunk(
            row,
            chunk_events,
            chunk_index=output_idx,
            kept_chunk_count=len(selected_indexes),
            original_chunk_index=chunk_idx + 1,
            original_chunk_count=total_chunks,
            start_event_idx=start_event_idx,
            end_event_idx=end_event_idx,
            original_event_count=len(events),
        )
        token_count = compute_token_count(chunk_content)
        chunk_row = dict(row)
        original_source_record_id = row.get("source_record_id") or row.get("record_id")
        original_record_id = row.get("record_id") or f"cloudtrail-flaws:{original_source_record_id}"
        chunk_row["source_record_id"] = f"{original_source_record_id}:chunk-{output_idx:04d}"
        chunk_row["record_id"] = f"{original_record_id}:chunk-{output_idx:04d}"
        chunk_row["content"] = chunk_content
        chunk_row["content_length"] = token_count
        chunk_row["content_hash"] = compute_content_hash(chunk_content)
        chunk_row["event_count"] = len(chunk_events)
        rows.append(chunk_row)
    return rows, stats


def _split_cloudtrail_content(content: str) -> tuple[str, list[str]]:
    lines = content.splitlines()
    event_start_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == "## Events":
            event_start_idx = idx + 1
            break

    if event_start_idx is None:
        event_start_idx = next(
            (idx for idx, line in enumerate(lines) if line.strip().startswith("{")),
            len(lines),
        )

    header = "\n".join(lines[:event_start_idx]).strip()
    events = [
        line.strip()
        for line in lines[event_start_idx:]
        if line.strip().startswith("{")
    ]
    return header, events


def _cloudtrail_event_chunks(
    header: str,
    events: Sequence[str],
    row: dict[str, Any],
    policy: CloudTrailPolicy,
) -> tuple[list[tuple[list[str], int, int]], int]:
    header_tokens = compute_token_count(_cloudtrail_chunk_header(row, 1, 1, 1, len(events), len(events)))
    chunks: list[tuple[list[str], int, int]] = []
    current: list[str] = []
    current_tokens = 0
    current_start = 1
    oversized_events = 0

    for event_idx, event in enumerate(events, start=1):
        event_tokens = compute_token_count(event)
        if event_tokens + header_tokens > policy.max_tokens:
            oversized_events += 1

        would_exceed_target = current and (header_tokens + current_tokens + event_tokens > policy.target_tokens)
        would_exceed_max = current and (header_tokens + current_tokens + event_tokens > policy.max_tokens)
        if would_exceed_target or would_exceed_max:
            chunks.append((current, current_start, event_idx - 1))
            current = []
            current_tokens = 0
            current_start = event_idx

        current.append(event)
        current_tokens += event_tokens

    if current:
        chunks.append((current, current_start, len(events)))
    if not chunks:
        chunks.append((list(events), 1, len(events)))
    return chunks, oversized_events


def _select_evenly_spaced_indexes(total: int, cap: int) -> list[int]:
    if total <= cap:
        return list(range(total))
    if cap <= 1:
        return [0]
    indexes = {
        round(idx * (total - 1) / (cap - 1))
        for idx in range(cap)
    }
    selected = sorted(indexes)
    cursor = 0
    while len(selected) < cap and cursor < total:
        if cursor not in selected:
            selected.append(cursor)
        cursor += 1
    return sorted(selected[:cap])


def _format_cloudtrail_chunk(
    row: dict[str, Any],
    events: Sequence[str],
    *,
    chunk_index: int,
    kept_chunk_count: int,
    original_chunk_index: int,
    original_chunk_count: int,
    start_event_idx: int,
    end_event_idx: int,
    original_event_count: int,
) -> str:
    first_event_time = _cloudtrail_event_time(events[0]) if events else None
    last_event_time = _cloudtrail_event_time(events[-1]) if events else None
    header = _cloudtrail_chunk_header(
        row,
        chunk_index,
        kept_chunk_count,
        start_event_idx,
        end_event_idx,
        original_event_count,
        original_chunk_index=original_chunk_index,
        original_chunk_count=original_chunk_count,
        first_event_time=first_event_time,
        last_event_time=last_event_time,
    )
    return header + "\n\n## Events\n\n" + "\n".join(events)


def _cloudtrail_chunk_header(
    row: dict[str, Any],
    chunk_index: int,
    kept_chunk_count: int,
    start_event_idx: int,
    end_event_idx: int,
    original_event_count: int,
    *,
    original_chunk_index: int | None = None,
    original_chunk_count: int | None = None,
    first_event_time: str | None = None,
    last_event_time: str | None = None,
) -> str:
    original_record_id = row.get("record_id") or ""
    original_chunk = ""
    if original_chunk_index is not None and original_chunk_count is not None:
        original_chunk = f"\nOriginal chunk position: {original_chunk_index}/{original_chunk_count}"
    time_range = ""
    if first_event_time or last_event_time:
        time_range = f"\nChunk event time range: {first_event_time or ''} - {last_event_time or ''}"
    return "\n".join([
        "# CloudTrail Session Chunk",
        f"Original session: {original_record_id}",
        f"Source IP: {row.get('source_ip') or ''}",
        f"Chunk: {chunk_index}/{kept_chunk_count}{original_chunk}",
        f"Original session duration seconds: {row.get('session_duration_seconds') or ''}",
        f"Events in chunk: {max(0, end_event_idx - start_event_idx + 1)}",
        f"Original event range: {start_event_idx}-{end_event_idx} of {original_event_count}",
        f"Principals: {_join_cloudtrail_values(row.get('principals'))}",
        f"Services: {_join_cloudtrail_values(row.get('aws_services'))}",
        f"Actions: {_join_cloudtrail_values(row.get('actions'))}",
        f"Regions: {_join_cloudtrail_values(row.get('regions'))}{time_range}",
    ]).strip()


def _join_cloudtrail_values(values: Any, limit: int = 80) -> str:
    if not values:
        return ""
    if isinstance(values, str):
        return values
    values = [str(value) for value in values]
    if len(values) <= limit:
        return ", ".join(values)
    return ", ".join(values[:limit]) + f", ... (+{len(values) - limit} more)"


def _cloudtrail_event_time(event: str) -> str | None:
    match = re.search(r'"eventTime"\s*:\s*"([^"]+)"', event)
    if match:
        return match.group(1)
    return None


def _sigma_chunk_rows(
    row: dict[str, Any],
    policy: SigmaPolicy,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = {
        "chunked_rules": 0,
        "dropped_extreme_records": 0,
        "dropped_extreme_tokens": 0,
        "capped_rules": 0,
        "dropped_chunks_from_cap": 0,
    }
    content_length = int(row.get("content_length") or 0)
    if content_length > policy.drop_above_tokens:
        stats["dropped_extreme_records"] = 1
        stats["dropped_extreme_tokens"] = content_length
        return [], stats
    return [dict(row)], stats


def _input_summary_sql(metadata_sql: str) -> str:
    return f"""
        SELECT
            count(*) AS input_records,
            coalesce(sum(content_length), 0)::BIGINT AS input_tokens,
            count(DISTINCT source_id) AS input_sources
        FROM ({metadata_sql})
    """


def _non_arxiv_output_summary_sql(classified_sql: str) -> str:
    return f"""
        SELECT
            count(*) AS output_records,
            coalesce(sum(content_length), 0)::BIGINT AS output_tokens,
            count(DISTINCT source_id) AS output_sources
        FROM ({classified_sql})
        WHERE source_id NOT IN ('arxiv', 'cloudtrail-flaws', 'sigma')
          AND v2_drop_reason IS NULL
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


def _qa_thresholds_output_sql(qa_thresholds_sql: str) -> str:
    return f"""
        SELECT *
        FROM ({qa_thresholds_sql})
        ORDER BY source_id
    """


def _drop_summary_sql(classified_sql: str) -> str:
    return f"""
        SELECT
            v2_drop_reason AS drop_reason,
            count(*) AS dropped_records,
            coalesce(sum(content_length), 0)::BIGINT AS dropped_tokens
        FROM ({classified_sql})
        WHERE source_id NOT IN ('arxiv', 'cloudtrail-flaws', 'sigma')
          AND v2_drop_reason IS NOT NULL
        GROUP BY v2_drop_reason
        ORDER BY dropped_records DESC
    """


def _drop_examples_sql(classified_sql: str, limit: int) -> str:
    return f"""
        SELECT
            source_id,
            record_id,
            v2_drop_reason,
            p50_score,
            p50_content_length,
            score,
            answer_count,
            content_length,
            left(coalesce(title::VARCHAR, ''), 160) AS title
        FROM ({classified_sql})
        WHERE source_id NOT IN ('arxiv', 'cloudtrail-flaws', 'sigma')
          AND v2_drop_reason IS NOT NULL
        ORDER BY v2_drop_reason, source_id, content_length DESC NULLS LAST, record_id
        LIMIT {limit}
    """


def _per_source_filter_summary_sql(classified_sql: str) -> str:
    return f"""
        WITH input AS (
            SELECT
                source_id,
                count(*) AS input_records,
                coalesce(sum(content_length), 0)::BIGINT AS input_tokens
            FROM ({classified_sql})
            GROUP BY source_id
        ),
        drops AS (
            SELECT
                source_id,
                count(*) AS dropped_records,
                coalesce(sum(content_length), 0)::BIGINT AS dropped_tokens
            FROM ({classified_sql})
            WHERE source_id NOT IN ('arxiv', 'cloudtrail-flaws', 'sigma')
              AND v2_drop_reason IS NOT NULL
            GROUP BY source_id
        ),
        output AS (
            SELECT
                source_id,
                count(*) AS output_records,
                coalesce(sum(content_length), 0)::BIGINT AS output_tokens
            FROM ({classified_sql})
            WHERE source_id NOT IN ('arxiv', 'cloudtrail-flaws', 'sigma')
              AND v2_drop_reason IS NULL
            GROUP BY source_id
        )
        SELECT
            input.source_id,
            input.input_records,
            input.input_tokens,
            coalesce(drops.dropped_records, 0) AS dropped_records,
            coalesce(drops.dropped_tokens, 0) AS dropped_tokens,
            coalesce(output.output_records, 0) AS output_records,
            coalesce(output.output_tokens, 0) AS output_tokens
        FROM input
        LEFT JOIN drops USING (source_id)
        LEFT JOIN output USING (source_id)
        ORDER BY input_tokens DESC
    """


def _manifest(
    *,
    paths: BuildPaths,
    input_summary: dict[str, Any],
    output_summary: dict[str, Any],
    drop_summary: list[dict[str, Any]],
    source_file_count: int,
    dry_run: bool,
    arxiv_policy: ArxivPolicy,
    arxiv_summary: ArxivSummary,
    cloudtrail_policy: CloudTrailPolicy,
    cloudtrail_summary: CloudTrailSummary,
    sigma_policy: SigmaPolicy,
    sigma_summary: SigmaSummary,
) -> dict[str, Any]:
    return {
        "name": "training-clean-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "source_file_count": source_file_count,
        "data_dir": paths.data_dir.as_posix(),
        "output_dir": paths.output_dir.as_posix(),
        "report_dir": paths.report_dir.as_posix(),
        "rules": [
            {
                "name": "qa_low_score_low_length",
                "description": (
                    "Drop Stack Overflow, Stack Exchange, and Reddit records where "
                    "score <= source-specific p50 score and content_length <= "
                    "source-specific p50 content length."
                ),
            },
            {
                "name": "qa_so_se_unanswered_low_score",
                "description": (
                    "Drop Stack Overflow and Stack Exchange records where score <= "
                    "source-specific p50 score and answer_count = 0."
                ),
            },
            {
                "name": "nvd_rejected",
                "description": "Drop NVD CVE records whose content starts with 'Rejected reason:'.",
            },
            {
                "name": "arxiv_semantic_chunking",
                "description": (
                    "Remove obvious LaTeX/source artifacts, references/bibliographies, "
                    "acknowledgements, tokenizer sentinels, and raw figure rendering "
                    "source such as TikZ/PGF coordinate dumps where detectable; then "
                    "assign each paper to one adaptive chunk-size bucket and chunk by "
                    "section/subsection/paragraph boundaries without duplicating papers."
                ),
                "chunk_sizes": list(arxiv_policy.chunk_sizes),
                "hard_max_tokens": max(arxiv_policy.chunk_sizes),
            },
            {
                "name": "cloudtrail_event_boundary_chunking",
                "description": (
                    "Drop extreme CloudTrail sessions above the outlier threshold. "
                    "Keep sessions at or below the trigger size intact, and chunk "
                    "larger retained sessions by intact chronological JSON events."
                ),
                "drop_above_tokens": cloudtrail_policy.drop_above_tokens,
                "trigger_tokens": cloudtrail_policy.trigger_tokens,
                "target_tokens": cloudtrail_policy.target_tokens,
                "max_tokens": cloudtrail_policy.max_tokens,
                "max_chunks_per_session": cloudtrail_policy.max_chunks_per_session,
            },
            {
                "name": "sigma_outlier_drop",
                "description": (
                    "Drop extreme Sigma records above the outlier threshold. Retained "
                    "Sigma rules keep their original record shape and are not chunked."
                ),
                "drop_above_tokens": sigma_policy.drop_above_tokens,
            },
        ],
        "input_summary": input_summary,
        "drop_summary": drop_summary,
        "arxiv_summary": arxiv_summary.__dict__,
        "cloudtrail_summary": cloudtrail_summary.__dict__,
        "sigma_summary": sigma_summary.__dict__,
        "output_summary": output_summary,
    }


def _write_markdown_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    drop_summary: list[dict[str, Any]],
    per_source_filter_summary: list[dict[str, Any]],
) -> None:
    lines = [
        "# training-clean-v2",
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
        "## arXiv Chunking",
        "",
        _markdown_table_from_dicts([manifest["arxiv_summary"]]),
        "",
        "## CloudTrail Chunking",
        "",
        _markdown_table_from_dicts([manifest["cloudtrail_summary"]]),
        "",
        "## Sigma Chunking",
        "",
        _markdown_table_from_dicts([manifest["sigma_summary"]]),
        "",
        "## Per Source",
        "",
        _markdown_table_from_dicts(per_source_filter_summary),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _print_summary(paths: BuildPaths, manifest: dict[str, Any], dry_run: bool) -> None:
    prefix = "Dry run complete" if dry_run else "training-clean-v2 written"
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
