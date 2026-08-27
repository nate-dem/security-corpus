#!/usr/bin/env python3
"""Write accepted citation-paper IDs after a complete pinned Qwen pass."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import duckdb


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {args.output}")

    connection = duckdb.connect()
    connection.execute(
        f"CREATE TEMP VIEW universe AS SELECT * FROM {_parquet_expression(args.universe)}"
    )
    connection.execute(
        f"CREATE TEMP VIEW decisions AS SELECT * FROM {_parquet_expression(args.decisions)}"
    )
    _validate(connection, args)
    accepted_ids = [
        row[0]
        for row in connection.execute(
            """
            SELECT u.arxiv_id
            FROM universe u
            JOIN decisions d USING (source_id, record_id, content_hash)
            WHERE d.qwen_should_keep
            ORDER BY u.arxiv_id
            """
        ).fetchall()
    ]
    configurations = connection.execute(
        """
        SELECT qwen_model, qwen_model_revision, qwen_prompt_version,
               qwen_task, count(*)
        FROM decisions
        GROUP BY ALL
        ORDER BY ALL
        """
    ).fetchall()
    total_records, kept_records = connection.execute(
        """
        SELECT count(*), count(*) FILTER (WHERE qwen_should_keep)
        FROM decisions
        """
    ).fetchone()
    connection.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text("".join(f"{paper_id}\n" for paper_id in accepted_ids), encoding="utf-8")
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "universe": args.universe.resolve().as_posix(),
        "decisions": args.decisions.resolve().as_posix(),
        "output": args.output.resolve().as_posix(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "universe_records": int(total_records),
        "accepted_records": int(kept_records),
        "expected_model": args.expected_model,
        "expected_revision": args.expected_revision,
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
    }
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe",
        type=Path,
        default=Path("data/filtering/v4/citation_abstract_universe.parquet"),
    )
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/filtering/v4/citation_accepted_ids.txt"),
    )
    parser.add_argument("--expected-model", default=DEFAULT_MODEL)
    parser.add_argument("--expected-revision", default=DEFAULT_REVISION)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _validate(
    connection: duckdb.DuckDBPyConnection,
    args: argparse.Namespace,
) -> None:
    required_universe = {"source_id", "record_id", "content_hash", "arxiv_id"}
    required_decisions = {
        "source_id",
        "record_id",
        "content_hash",
        "qwen_should_keep",
        "qwen_parse_status",
        "qwen_model",
        "qwen_model_revision",
        "qwen_prompt_version",
        "qwen_task",
    }
    for relation, required in (
        ("universe", required_universe),
        ("decisions", required_decisions),
    ):
        columns = {
            row[0]
            for row in connection.execute(
                f"DESCRIBE SELECT * FROM {relation}"
            ).fetchall()
        }
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"{relation} is missing columns: {', '.join(missing)}")

    universe_count, universe_keys, universe_ids = connection.execute(
        """
        SELECT count(*),
               count(DISTINCT (source_id, record_id, content_hash)),
               count(DISTINCT arxiv_id)
        FROM universe
        """
    ).fetchone()
    decision_count, decision_keys = connection.execute(
        """
        SELECT count(*), count(DISTINCT (source_id, record_id, content_hash))
        FROM decisions
        """
    ).fetchone()
    matched = connection.execute(
        """
        SELECT count(*) FROM universe
        JOIN decisions USING (source_id, record_id, content_hash)
        """
    ).fetchone()[0]
    invalid = connection.execute(
        """
        SELECT count(*) FROM decisions
        WHERE qwen_should_keep IS NULL
           OR qwen_parse_status NOT IN ('ok', 'extracted_json')
           OR qwen_model <> ?
           OR qwen_model_revision <> ?
        """,
        [args.expected_model, args.expected_revision],
    ).fetchone()[0]
    problems: dict[str, int] = {}
    for label, value in (
        ("duplicate universe keys", universe_count - universe_keys),
        ("duplicate universe arxiv_ids", universe_count - universe_ids),
        ("duplicate decision keys", decision_count - decision_keys),
        ("missing or orphaned decisions", universe_count + decision_count - 2 * matched),
        ("invalid or wrong-model decisions", invalid),
    ):
        if value:
            problems[label] = int(value)
    if problems:
        rendered = ", ".join(f"{label}={count}" for label, count in problems.items())
        raise ValueError("Citation decision coverage is incomplete: " + rendered)


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
