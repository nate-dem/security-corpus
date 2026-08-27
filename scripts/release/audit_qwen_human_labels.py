#!/usr/bin/env python3
"""Validate a completed Qwen human audit and report agreement without a threshold."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import duckdb


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    connection = duckdb.connect()
    relation = _input_expression(args.input)
    connection.execute(f"CREATE TEMP VIEW audit AS SELECT * FROM {relation}")
    _require_columns(connection)
    report = _build_report(connection, args.input)
    connection.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 2 if report["release_blocking_issues"] else 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _require_columns(connection: duckdb.DuckDBPyConnection) -> None:
    columns = {
        row[0]
        for row in connection.execute("DESCRIBE SELECT * FROM audit").fetchall()
    }
    required = {
        "source_id",
        "record_id",
        "content_hash",
        "qwen_should_keep",
        "manual_should_keep",
        "manual_notes",
        "reviewer",
        "reviewed_at",
    }
    missing = sorted(required - columns)
    if missing:
        raise ValueError("Human audit is missing columns: " + ", ".join(missing))


def _build_report(
    connection: duckdb.DuckDBPyConnection,
    input_path: Path,
) -> dict[str, Any]:
    total_records, duplicate_keys = connection.execute(
        """
        SELECT coalesce(sum(copies), 0), coalesce(sum(copies - 1), 0)
        FROM (
            SELECT count(*) AS copies
            FROM audit
            GROUP BY source_id, record_id, content_hash
        )
        """
    ).fetchone()
    incomplete_labels, invalid_qwen_labels, blank_reviewers, invalid_dates = (
        connection.execute(
            """
            SELECT
                count(*) FILTER (
                    WHERE try_cast(manual_should_keep AS BOOLEAN) IS NULL
                ),
                count(*) FILTER (
                    WHERE try_cast(qwen_should_keep AS BOOLEAN) IS NULL
                ),
                count(*) FILTER (
                    WHERE reviewer IS NULL OR trim(cast(reviewer AS VARCHAR)) = ''
                ),
                count(*) FILTER (
                    WHERE try_cast(reviewed_at AS TIMESTAMPTZ) IS NULL
                )
            FROM audit
            """
        ).fetchone()
    )
    confusion = connection.execute(
        """
        SELECT
            source_id,
            try_cast(qwen_should_keep AS BOOLEAN) AS qwen_keep,
            try_cast(manual_should_keep AS BOOLEAN) AS manual_keep,
            count(*)
        FROM audit
        WHERE try_cast(qwen_should_keep AS BOOLEAN) IS NOT NULL
          AND try_cast(manual_should_keep AS BOOLEAN) IS NOT NULL
        GROUP BY ALL
        ORDER BY ALL
        """
    ).fetchall()
    labeled = sum(int(row[3]) for row in confusion)
    agreements = sum(int(row[3]) for row in confusion if row[1] == row[2])
    issues = {
        "duplicate_keys": int(duplicate_keys),
        "incomplete_manual_labels": int(incomplete_labels),
        "invalid_qwen_labels": int(invalid_qwen_labels),
        "blank_reviewers": int(blank_reviewers),
        "missing_or_invalid_reviewed_at": int(invalid_dates),
    }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": input_path.resolve().as_posix(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "total_records": int(total_records),
        "labeled_records": labeled,
        "agreements": agreements,
        "disagreements": labeled - agreements,
        "agreement_rate": agreements / labeled if labeled else None,
        "confusion_by_source": [
            {
                "source_id": source_id,
                "qwen_should_keep": qwen_keep,
                "manual_should_keep": manual_keep,
                "records": int(records),
            }
            for source_id, qwen_keep, manual_keep, records in confusion
        ],
        "release_blocking_issues": {
            label: count for label, count in issues.items() if count
        },
        "agreement_threshold_applied": False,
    }


def _input_expression(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    rendered = _sql_string(resolved.as_posix())
    if resolved.suffix.lower() == ".parquet":
        return f"read_parquet({rendered})"
    if resolved.suffix.lower() == ".csv":
        return f"read_csv_auto({rendered}, header = true, all_varchar = true)"
    raise ValueError("--input must end in .parquet or .csv")


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
