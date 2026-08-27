#!/usr/bin/env python3
"""Create a deterministic human-audit sample from complete Qwen decisions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import duckdb


VALID_PARSE_STATUSES = {"ok", "extracted_json"}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.per_stratum <= 0:
        raise ValueError("--per-stratum must be positive")
    if args.output.suffix.lower() not in {".parquet", ".csv"}:
        raise ValueError("--output must end in .parquet or .csv")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {args.output}")

    connection = duckdb.connect()
    corpus = _parquet_expression(args.corpus)
    decisions = _parquet_expression(args.decisions)
    connection.execute(f"CREATE TEMP VIEW corpus AS SELECT * FROM {corpus}")
    connection.execute(f"CREATE TEMP VIEW decisions AS SELECT * FROM {decisions}")
    _validate_inputs(connection)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    try:
        _write_sample(connection, temporary, args)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = _build_manifest(connection, args)
    connection.close()
    manifest_path = args.manifest or args.output.with_suffix(
        args.output.suffix + ".manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--per-stratum",
        type=int,
        required=True,
        help="Records to sample for every source_id x keep/drop stratum.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _validate_inputs(connection: duckdb.DuckDBPyConnection) -> None:
    corpus_columns = _columns(connection, "corpus")
    decision_columns = _columns(connection, "decisions")
    keys = {"source_id", "record_id", "content_hash"}
    missing_corpus = sorted(keys - corpus_columns)
    missing_decisions = sorted(
        (
            keys
            | {
            "qwen_should_keep",
            "qwen_parse_status",
            "qwen_model",
            "qwen_model_revision",
            "qwen_prompt_version",
            }
        )
        - decision_columns
    )
    if missing_corpus:
        raise ValueError("Corpus is missing columns: " + ", ".join(missing_corpus))
    if missing_decisions:
        raise ValueError(
            "Decisions are missing columns: " + ", ".join(missing_decisions)
        )

    checks = {
        "duplicate corpus keys": """
            SELECT coalesce(sum(copies - 1), 0) FROM (
                SELECT count(*) copies FROM corpus GROUP BY source_id, record_id, content_hash
            )
        """,
        "duplicate decision keys": """
            SELECT coalesce(sum(copies - 1), 0) FROM (
                SELECT count(*) copies FROM decisions GROUP BY source_id, record_id, content_hash
            )
        """,
        "missing decisions": """
            SELECT count(*) FROM corpus c
            LEFT JOIN decisions d USING (source_id, record_id, content_hash)
            WHERE d.record_id IS NULL
        """,
        "orphaned decisions": """
            SELECT count(*) FROM decisions d
            LEFT JOIN corpus c USING (source_id, record_id, content_hash)
            WHERE c.record_id IS NULL
        """,
        "unusable decisions": """
            SELECT count(*) FROM decisions
            WHERE qwen_should_keep IS NULL
               OR qwen_parse_status NOT IN ('ok', 'extracted_json')
        """,
    }
    failures = {
        label: int(connection.execute(sql).fetchone()[0])
        for label, sql in checks.items()
    }
    failures = {label: count for label, count in failures.items() if count}
    if failures:
        rendered = ", ".join(f"{label}={count}" for label, count in failures.items())
        raise ValueError(
            "Qwen coverage is not release-auditable; run audit_qwen_coverage.py: "
            + rendered
        )


def _write_sample(
    connection: duckdb.DuckDBPyConnection,
    destination: Path,
    args: argparse.Namespace,
) -> None:
    seed = _sql_string(str(args.seed))
    query = f"""
        SELECT
            c.*,
            d.* EXCLUDE (source_id, record_id, content_hash),
            CAST(NULL AS BOOLEAN) AS manual_should_keep,
            CAST(NULL AS VARCHAR) AS manual_notes,
            CAST(NULL AS VARCHAR) AS reviewer,
            CAST(NULL AS TIMESTAMP) AS reviewed_at
        FROM corpus c
        JOIN decisions d USING (source_id, record_id, content_hash)
        QUALIFY row_number() OVER (
            PARTITION BY c.source_id, d.qwen_should_keep
            ORDER BY sha256(
                concat_ws(chr(31), {seed}, c.source_id, c.record_id, c.content_hash)
            )
        ) <= {args.per_stratum}
        ORDER BY c.source_id, d.qwen_should_keep, c.record_id
    """
    destination_sql = _sql_string(destination.as_posix())
    if args.output.suffix.lower() == ".parquet":
        options = "FORMAT PARQUET, COMPRESSION ZSTD"
    else:
        options = "FORMAT CSV, HEADER TRUE"
    connection.execute(f"COPY ({query}) TO {destination_sql} ({options})")


def _build_manifest(
    connection: duckdb.DuckDBPyConnection,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_expression = (
        _parquet_expression(args.output)
        if args.output.suffix.lower() == ".parquet"
        else None
    )
    if output_expression:
        rows = connection.execute(
            f"SELECT source_id, qwen_should_keep, count(*) FROM {output_expression} GROUP BY ALL ORDER BY ALL"
        ).fetchall()
    else:
        rows = connection.execute(
            f"""
            SELECT source_id, qwen_should_keep, count(*)
            FROM read_csv_auto({_sql_string(args.output.as_posix())}, header = true)
            GROUP BY ALL ORDER BY ALL
            """
        ).fetchall()
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus": args.corpus.resolve().as_posix(),
        "decisions": args.decisions.resolve().as_posix(),
        "output": args.output.resolve().as_posix(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "per_stratum": args.per_stratum,
        "seed": args.seed,
        "strata": [
            {
                "source_id": source_id,
                "qwen_should_keep": keep,
                "records": int(records),
            }
            for source_id, keep, records in rows
        ],
        "total_records": sum(int(row[2]) for row in rows),
        "review_fields": [
            "manual_should_keep",
            "manual_notes",
            "reviewer",
            "reviewed_at",
        ],
    }


def _columns(connection: duckdb.DuckDBPyConnection, relation: str) -> set[str]:
    return {
        row[0]
        for row in connection.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
    }


def _parquet_expression(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_file():
        files = [resolved]
    elif resolved.is_dir():
        files = sorted(resolved.rglob("*.parquet"))
    else:
        raise FileNotFoundError(resolved)
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {resolved}")
    rendered = ", ".join(_sql_string(file.as_posix()) for file in files)
    return f"read_parquet([{rendered}], union_by_name = true, hive_partitioning = false)"


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
