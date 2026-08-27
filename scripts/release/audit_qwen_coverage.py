#!/usr/bin/env python3
"""Verify complete, one-to-one Qwen decision coverage for a corpus universe."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import duckdb


VALID_PARSE_STATUSES = {"ok", "extracted_json"}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    connection = duckdb.connect()
    corpus = _parquet_expression(args.corpus)
    decisions = _parquet_expression(args.decisions)
    _require_columns(
        connection,
        corpus,
        {"source_id", "record_id", "content_hash"},
        "corpus",
    )
    _require_columns(
        connection,
        decisions,
        {
            "source_id",
            "record_id",
            "content_hash",
            "qwen_should_keep",
            "qwen_parse_status",
            "qwen_model",
            "qwen_model_revision",
            "qwen_prompt_version",
            "qwen_scored_at",
            "qwen_task",
        },
        "decisions",
    )
    connection.execute(f"CREATE TEMP VIEW corpus AS SELECT * FROM {corpus}")
    connection.execute(f"CREATE TEMP VIEW decisions AS SELECT * FROM {decisions}")
    report = _build_report(connection, args)
    connection.close()

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 2 if report["release_blocking_issues"] else 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--expected-model")
    parser.add_argument("--expected-revision")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _build_report(
    connection: duckdb.DuckDBPyConnection,
    args: argparse.Namespace,
) -> dict[str, Any]:
    corpus_records, corpus_duplicate_keys = connection.execute(
        """
        SELECT coalesce(sum(copies), 0), coalesce(sum(copies - 1), 0)
        FROM (
            SELECT count(*) AS copies
            FROM corpus
            GROUP BY source_id, record_id, content_hash
        )
        """
    ).fetchone()
    decision_records, decision_duplicate_keys = connection.execute(
        """
        SELECT coalesce(sum(copies), 0), coalesce(sum(copies - 1), 0)
        FROM (
            SELECT count(*) AS copies
            FROM decisions
            GROUP BY source_id, record_id, content_hash
        )
        """
    ).fetchone()
    missing = connection.execute(
        """
        SELECT count(*)
        FROM corpus c
        LEFT JOIN decisions d USING (source_id, record_id, content_hash)
        WHERE d.record_id IS NULL
        """
    ).fetchone()[0]
    orphaned = connection.execute(
        """
        SELECT count(*)
        FROM decisions d
        LEFT JOIN corpus c USING (source_id, record_id, content_hash)
        WHERE c.record_id IS NULL
        """
    ).fetchone()[0]
    undecided, parse_failures = connection.execute(
        """
        SELECT
            count(*) FILTER (WHERE qwen_should_keep IS NULL),
            count(*) FILTER (
                WHERE qwen_parse_status NOT IN ('ok', 'extracted_json')
            )
        FROM decisions
        """
    ).fetchone()
    invalid_provenance = connection.execute(
        """
        SELECT count(*)
        FROM decisions
        WHERE qwen_model IS NULL OR trim(qwen_model) = ''
           OR qwen_model_revision IS NULL
           OR NOT regexp_full_match(qwen_model_revision, '[0-9a-f]{40}')
           OR qwen_prompt_version IS NULL OR trim(qwen_prompt_version) = ''
           OR qwen_scored_at IS NULL OR trim(qwen_scored_at) = ''
           OR qwen_task IS NULL OR trim(qwen_task) = ''
        """
    ).fetchone()[0]
    outcome_rows = connection.execute(
        """
        SELECT
            qwen_should_keep,
            count(*),
            coalesce(sum(c.content_length), 0)
        FROM decisions d
        JOIN corpus c USING (source_id, record_id, content_hash)
        GROUP BY qwen_should_keep
        ORDER BY qwen_should_keep
        """
        if _has_column(connection, "corpus", "content_length")
        else """
        SELECT qwen_should_keep, count(*), NULL
        FROM decisions d
        JOIN corpus c USING (source_id, record_id, content_hash)
        GROUP BY qwen_should_keep
        ORDER BY qwen_should_keep
        """
    ).fetchall()
    configurations = connection.execute(
        """
        SELECT qwen_model, qwen_model_revision, qwen_prompt_version,
               qwen_task, count(*)
        FROM decisions
        GROUP BY ALL
        ORDER BY ALL
        """
    ).fetchall()
    model_mismatches = _expected_mismatches(connection, args)

    issues = {
        "corpus_duplicate_keys": int(corpus_duplicate_keys),
        "decision_duplicate_keys": int(decision_duplicate_keys),
        "missing_decisions": int(missing),
        "orphaned_decisions": int(orphaned),
        "undecided_rows": int(undecided),
        "invalid_parse_rows": int(parse_failures),
        "invalid_provenance_rows": int(invalid_provenance),
        "expected_model_or_revision_mismatches": int(model_mismatches),
    }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus": args.corpus.resolve().as_posix(),
        "decisions": args.decisions.resolve().as_posix(),
        "corpus_records": int(corpus_records),
        "decision_records": int(decision_records),
        "outcomes": [
            {
                "qwen_should_keep": keep,
                "records": int(records),
                "tokens": int(tokens) if tokens is not None else None,
            }
            for keep, records, tokens in outcome_rows
        ],
        "configurations": [
            {
                "model": model,
                "model_revision": revision,
                "prompt_version": prompt,
                "task": task,
                "records": int(records),
            }
            for model, revision, prompt, task, records in configurations
        ],
        "release_blocking_issues": {
            name: count for name, count in issues.items() if count
        },
    }


def _expected_mismatches(
    connection: duckdb.DuckDBPyConnection,
    args: argparse.Namespace,
) -> int:
    clauses = []
    parameters = []
    if args.expected_model:
        clauses.append("qwen_model <> ?")
        parameters.append(args.expected_model)
    if args.expected_revision:
        clauses.append("qwen_model_revision <> ?")
        parameters.append(args.expected_revision)
    if not clauses:
        return 0
    return int(
        connection.execute(
            "SELECT count(*) FROM decisions WHERE " + " OR ".join(clauses),
            parameters,
        ).fetchone()[0]
    )


def _require_columns(
    connection: duckdb.DuckDBPyConnection,
    relation: str,
    required: set[str],
    label: str,
) -> None:
    columns = {
        row[0]
        for row in connection.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
    }
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _has_column(
    connection: duckdb.DuckDBPyConnection,
    relation: str,
    column: str,
) -> bool:
    return column in {
        row[0] for row in connection.execute(f"DESCRIBE {relation}").fetchall()
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
    paths = ", ".join(_sql_string(file.as_posix()) for file in files)
    return f"read_parquet([{paths}], union_by_name = true, hive_partitioning = false)"


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
