#!/usr/bin/env python3
"""Merge Qwen shard outputs and write an audit manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import duckdb


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))

from classify.io import ensure_parent, write_json  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    part_files = sorted(args.input_dir.glob("**/*.parquet"))
    if not part_files:
        raise FileNotFoundError(f"No Qwen shard parquet files found in {args.input_dir}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {args.output}")

    ensure_parent(args.output)
    con = duckdb.connect()
    file_expr = _read_parquet_expr(part_files)
    _validate_parts(con, file_expr)
    con.sql(f"""
        COPY (
            SELECT * EXCLUDE (rn)
            FROM (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY source_id, record_id, content_hash
                        ORDER BY
                            qwen_scored_at DESC,
                            qwen_prompt_version DESC,
                            qwen_model_revision DESC,
                            qwen_shard_id DESC
                    ) AS rn
                FROM {file_expr}
            )
            WHERE rn = 1
        )
        TO {_sql_string(args.output.as_posix())}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    run_configs = sorted(args.input_dir.glob("**/run-config.json"))
    manifest = _build_manifest(con, args.output, part_files, run_configs)
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    print(f"Merged {manifest['total_records']} Qwen rows into {args.output}")
    print(f"Wrote manifest to {manifest_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _build_manifest(
    con: duckdb.DuckDBPyConnection,
    output: Path,
    part_files: Sequence[Path],
    run_configs: Sequence[Path],
) -> dict:
    summary = con.sql(f"""
        SELECT
            count(*) AS total_records,
            sum(CASE WHEN qwen_should_keep THEN 1 ELSE 0 END) AS kept,
            sum(CASE WHEN qwen_should_keep = false THEN 1 ELSE 0 END) AS dropped,
            sum(CASE WHEN qwen_parse_status NOT IN ('ok', 'extracted_json')
                     THEN 1 ELSE 0 END) AS parse_failures,
            sum(CASE WHEN qwen_should_keep IS NULL THEN 1 ELSE 0 END) AS undecided
        FROM read_parquet({_sql_string(output.as_posix())})
    """).fetchone()
    status_rows = con.sql(f"""
        SELECT qwen_parse_status, count(*) AS records
        FROM read_parquet({_sql_string(output.as_posix())})
        GROUP BY qwen_parse_status
        ORDER BY qwen_parse_status
    """).fetchall()
    model_rows = con.sql(f"""
        SELECT DISTINCT
            qwen_model,
            qwen_model_revision,
            qwen_prompt_version,
            qwen_task
        FROM read_parquet({_sql_string(output.as_posix())})
        ORDER BY qwen_model, qwen_prompt_version, qwen_task
    """).fetchall()
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": output.as_posix(),
        "input_part_files": [path.as_posix() for path in part_files],
        "input_run_configs": [
            {
                "path": path.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "config": json.loads(path.read_text(encoding="utf-8")),
            }
            for path in run_configs
        ],
        "total_records": int(summary[0] or 0),
        "kept": int(summary[1] or 0),
        "dropped": int(summary[2] or 0),
        "parse_failures": int(summary[3] or 0),
        "undecided": int(summary[4] or 0),
        "parse_status_counts": {
            str(status): int(count)
            for status, count in status_rows
        },
        "model_prompt_tasks": [
            {
                "qwen_model": model,
                "qwen_model_revision": model_revision,
                "qwen_prompt_version": prompt_version,
                "qwen_task": task,
            }
            for model, model_revision, prompt_version, task in model_rows
        ],
    }


def _validate_parts(con: duckdb.DuckDBPyConnection, file_expr: str) -> None:
    columns = {
        row[0] for row in con.sql(f"DESCRIBE SELECT * FROM {file_expr}").fetchall()
    }
    required = {
        "source_id",
        "record_id",
        "content_hash",
        "qwen_should_keep",
        "qwen_parse_status",
        "qwen_model",
        "qwen_model_revision",
        "qwen_prompt_version",
        "qwen_scored_at",
        "qwen_shard_id",
        "qwen_task",
    }
    missing = sorted(required - columns)
    if missing:
        raise ValueError("Qwen shards are missing columns: " + ", ".join(missing))
    invalid = con.sql(f"""
        SELECT count(*)
        FROM {file_expr}
        WHERE qwen_model_revision IS NULL OR trim(qwen_model_revision) = ''
           OR qwen_scored_at IS NULL OR trim(qwen_scored_at) = ''
    """).fetchone()[0]
    if invalid:
        raise ValueError(
            f"Qwen shards contain {invalid} rows without model revision or scored_at"
        )


def _read_parquet_expr(files: Sequence[Path]) -> str:
    paths = ", ".join(_sql_string(path.as_posix()) for path in files)
    return f"read_parquet([{paths}], union_by_name = true)"


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
