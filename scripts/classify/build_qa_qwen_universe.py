#!/usr/bin/env python3
"""Build the complete, exact-deduplicated QA corpus and Qwen work queues.

Only structural validity checks are applied here. Quality and relevance remain
Qwen decisions. The source inputs are the original normalized Stack Overflow,
Stack Exchange, and Reddit datasets, including the recovered 2026 Reddit data.

Outputs are partitioned by source without dropping the ``source_id`` column:

* ``qa_universe``: one canonical record per content hash;
* ``qa_pending``: canonical records without an existing exact-key decision;
* ``qa_rescore``: prior decisions that are not fully reproducible or auditable;
* ``qa_to_score``: the union of pending and re-score records;
* ``qa_exact_duplicates.parquet``: every non-canonical duplicate and its winner;
* ``manifest.json``: input files, counts, token totals, and exclusion counts.

The script writes to temporary directories and publishes them only after every
query succeeds. Existing outputs are never replaced without ``--overwrite``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any

import duckdb


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "filtering" / "v4"
DEFAULT_DECISIONS = ROOT / "data" / "filtering" / "v3" / "qwen_qa_decisions.parquet"
DEFAULT_MAX_CONTENT_CHARS = 24_000

NORMALIZED_COLUMNS = (
    "source_id",
    "source_record_id",
    "record_id",
    "content",
    "title",
    "content_length",
    "content_hash",
    "ingested_at",
    "published_at",
    "source_url",
    "license",
    "raw",
    "score",
    "answer_count",
    "has_accepted_answer",
    "closed",
    "tags",
)


def main() -> None:
    args = _parse_args()
    source_files = _discover_source_files(args.root)
    if not source_files:
        raise FileNotFoundError("No normalized QA Parquet files were found")
    if not args.decisions.is_file():
        raise FileNotFoundError(f"Qwen decision sidecar not found: {args.decisions}")

    output_root = args.output_root.resolve()
    targets = {
        "universe": output_root / "qa_universe",
        "pending": output_root / "qa_pending",
        "rescore": output_root / "qa_rescore",
        "to_score": output_root / "qa_to_score",
        "duplicates": output_root / "qa_exact_duplicates.parquet",
        "manifest": output_root / "manifest.json",
    }
    _prepare_targets(targets, overwrite=args.overwrite)

    output_root.mkdir(parents=True, exist_ok=True)
    temporary_root = output_root / f".qa-universe-{os.getpid()}.tmp"
    temporary_root.mkdir()
    connection = duckdb.connect()
    connection.execute("SET preserve_insertion_order = false")
    connection.execute(f"SET temp_directory = '{_sql_path(temporary_root / 'duckdb-tmp')}'")
    connection.execute(f"SET memory_limit = '{args.memory_limit}'")

    try:
        _register_inputs(connection, source_files, args.decisions)
        _materialize_ranked_records(connection)
        _materialize_outputs(
            connection,
            temporary_root,
            max_content_chars=args.max_content_chars,
        )
        manifest = _build_manifest(
            connection,
            source_files=source_files,
            decisions=args.decisions,
            max_content_chars=args.max_content_chars,
        )
        manifest_path = temporary_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        connection.close()
        _publish_outputs(temporary_root, targets)
    except Exception:
        connection.close()
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    print(json.dumps(manifest, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument(
        "--max-content-chars",
        type=int,
        default=DEFAULT_MAX_CONTENT_CHARS,
        help="Re-score prior decisions whose source content exceeded this old prompt limit.",
    )
    parser.add_argument(
        "--memory-limit",
        default="8GB",
        help="DuckDB memory limit; larger intermediate state spills under the output root.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_content_chars <= 0:
        parser.error("--max-content-chars must be positive")
    return args


def _discover_source_files(root: Path) -> list[Path]:
    data = root.resolve() / "data"
    patterns = (
        "stackoverflow/normalized/**/*.parquet",
        "stackexchange-*/normalized/**/*.parquet",
        "reddit/normalized/**/*.parquet",
        "reddit/normalized-2026/**/*.parquet",
    )
    files = {path.resolve() for pattern in patterns for path in data.glob(pattern)}
    return sorted(path for path in files if path.is_file())


def _prepare_targets(targets: dict[str, Path], *, overwrite: bool) -> None:
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        rendered = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(f"Refusing to replace existing QA outputs:\n{rendered}")
    if overwrite:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def _register_inputs(
    connection: duckdb.DuckDBPyConnection,
    source_files: list[Path],
    decisions: Path,
) -> None:
    parquet_paths = ", ".join(f"'{_sql_path(path)}'" for path in source_files)
    connection.execute(
        f"""
        CREATE TEMP VIEW qa_input AS
        SELECT
            * EXCLUDE (license),
            CASE
                WHEN source_id LIKE 'reddit-%' THEN 'NOASSERTION'
                ELSE license
            END AS license
        FROM read_parquet([{parquet_paths}], union_by_name = true, hive_partitioning = false)
        """
    )
    available = {
        row[0]
        for row in connection.execute("DESCRIBE qa_input").fetchall()
    }
    missing = sorted(set(NORMALIZED_COLUMNS) - available)
    if missing:
        raise ValueError(f"Normalized QA input is missing columns: {', '.join(missing)}")

    connection.execute(
        f"""
        CREATE TEMP VIEW decision_input AS
        SELECT *
        FROM read_parquet('{_sql_path(decisions)}', hive_partitioning = false)
        """
    )
    decision_columns = {
        row[0]
        for row in connection.execute("DESCRIBE decision_input").fetchall()
    }
    required_decision_columns = {
        "source_id",
        "record_id",
        "content_hash",
        "qwen_parse_status",
    }
    missing_decision = sorted(required_decision_columns - decision_columns)
    if missing_decision:
        raise ValueError(
            "Qwen decision sidecar is missing columns: " + ", ".join(missing_decision)
        )
    if "qwen_model_revision" in decision_columns:
        revision_expression = (
            "qwen_model_revision IS NOT NULL AND trim(qwen_model_revision) <> ''"
        )
    else:
        revision_expression = "false"
    connection.execute(
        f"""
        CREATE TEMP VIEW normalized_decisions AS
        SELECT *, {revision_expression} AS has_immutable_model_revision
        FROM decision_input
        """
    )


def _materialize_ranked_records(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TEMP TABLE decision_keys AS
        SELECT
            source_id,
            record_id,
            content_hash,
            min(qwen_parse_status) AS qwen_parse_status,
            bool_or(has_immutable_model_revision) AS has_immutable_model_revision
        FROM normalized_decisions
        WHERE source_id IS NOT NULL
          AND record_id IS NOT NULL
          AND content_hash IS NOT NULL
        GROUP BY ALL
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE qa_ranked AS
        SELECT
            q.*,
            d.record_id IS NOT NULL AS has_existing_decision,
            d.qwen_parse_status,
            coalesce(d.has_immutable_model_revision, false)
                AS has_immutable_model_revision,
            row_number() OVER (
                PARTITION BY q.content_hash
                ORDER BY
                    (d.record_id IS NOT NULL) DESC,
                    q.source_id,
                    q.record_id
            ) AS exact_dedup_rank,
            first_value(q.source_id) OVER (
                PARTITION BY q.content_hash
                ORDER BY
                    (d.record_id IS NOT NULL) DESC,
                    q.source_id,
                    q.record_id
            ) AS canonical_source_id,
            first_value(q.record_id) OVER (
                PARTITION BY q.content_hash
                ORDER BY
                    (d.record_id IS NOT NULL) DESC,
                    q.source_id,
                    q.record_id
            ) AS canonical_record_id
        FROM qa_input q
        LEFT JOIN decision_keys d
          ON q.source_id = d.source_id
         AND q.record_id = d.record_id
         AND q.content_hash = d.content_hash
        WHERE q.source_id IS NOT NULL
          AND trim(q.source_id) <> ''
          AND q.record_id IS NOT NULL
          AND trim(q.record_id) <> ''
          AND q.content_hash IS NOT NULL
          AND trim(q.content_hash) <> ''
          AND q.content IS NOT NULL
          AND trim(q.content) <> ''
        """
    )


def _materialize_outputs(
    connection: duckdb.DuckDBPyConnection,
    temporary_root: Path,
    *,
    max_content_chars: int,
) -> None:
    columns = ", ".join(_quote_identifier(column) for column in NORMALIZED_COLUMNS)
    universe_where = "exact_dedup_rank = 1"
    pending_where = f"{universe_where} AND NOT has_existing_decision"
    rescore_where = (
        f"{universe_where} AND has_existing_decision AND "
        f"(NOT has_immutable_model_revision OR "
        f"coalesce(qwen_parse_status, '') NOT IN ('ok', 'extracted_json') OR "
        f"length(content) > {max_content_chars})"
    )
    to_score_where = f"({pending_where}) OR ({rescore_where})"

    for name, where in (
        ("qa_universe", universe_where),
        ("qa_pending", pending_where),
        ("qa_rescore", rescore_where),
        ("qa_to_score", to_score_where),
    ):
        _write_partitioned(connection, temporary_root / name, columns, where)

    duplicate_path = temporary_root / "qa_exact_duplicates.parquet"
    connection.execute(
        f"""
        COPY (
            SELECT
                content_hash,
                canonical_source_id,
                canonical_record_id,
                source_id AS duplicate_source_id,
                record_id AS duplicate_record_id,
                content_length AS duplicate_content_length
            FROM qa_ranked
            WHERE exact_dedup_rank > 1
            ORDER BY content_hash, duplicate_source_id, duplicate_record_id
        ) TO '{_sql_path(duplicate_path)}'
          (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def _write_partitioned(
    connection: duckdb.DuckDBPyConnection,
    output_path: Path,
    columns: str,
    where: str,
) -> None:
    output_path.mkdir()
    source_ids = [
        row[0]
        for row in connection.execute(
            f"SELECT DISTINCT source_id FROM qa_ranked WHERE {where} ORDER BY source_id"
        ).fetchall()
    ]
    for source_id in source_ids:
        safe_source_id = _partition_value(source_id)
        source_dir = output_path / f"source_id={safe_source_id}"
        source_dir.mkdir()
        output_file = source_dir / "part-00000.parquet"
        source_literal = str(source_id).replace("'", "''")
        connection.execute(
            f"""
            COPY (
                SELECT {columns}
                FROM qa_ranked
                WHERE ({where}) AND source_id = '{source_literal}'
                ORDER BY record_id
            ) TO '{_sql_path(output_file)}'
              (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
            """
        )


def _build_manifest(
    connection: duckdb.DuckDBPyConnection,
    *,
    source_files: list[Path],
    decisions: Path,
    max_content_chars: int,
) -> dict[str, Any]:
    total_input, invalid_input = connection.execute(
        """
        SELECT
            count(*),
            count(*) FILTER (WHERE
                source_id IS NULL OR trim(source_id) = '' OR
                record_id IS NULL OR trim(record_id) = '' OR
                content_hash IS NULL OR trim(content_hash) = '' OR
                content IS NULL OR trim(content) = ''
            )
        FROM qa_input
        """
    ).fetchone()

    sets = {}
    predicates = {
        "universe": "exact_dedup_rank = 1",
        "pending": "exact_dedup_rank = 1 AND NOT has_existing_decision",
        "rescore": (
            "exact_dedup_rank = 1 AND has_existing_decision AND "
            f"(NOT has_immutable_model_revision OR "
            f"coalesce(qwen_parse_status, '') NOT IN ('ok', 'extracted_json') OR "
            f"length(content) > {max_content_chars})"
        ),
    }
    predicates["to_score"] = f"({predicates['pending']}) OR ({predicates['rescore']})"
    for name, predicate in predicates.items():
        rows, tokens = connection.execute(
            f"""
            SELECT count(*), coalesce(sum(content_length), 0)
            FROM qa_ranked
            WHERE {predicate}
            """
        ).fetchone()
        sets[name] = {"records": int(rows), "tokens": int(tokens)}

    per_source_rows = connection.execute(
        """
        SELECT
            source_id,
            count(*) AS records,
            coalesce(sum(content_length), 0) AS tokens,
            count(*) FILTER (WHERE NOT has_existing_decision) AS pending_records,
            count(*) FILTER (WHERE has_existing_decision) AS previously_scored_records
        FROM qa_ranked
        WHERE exact_dedup_rank = 1
        GROUP BY source_id
        ORDER BY source_id
        """
    ).fetchall()
    per_source = {
        row[0]: {
            "records": int(row[1]),
            "tokens": int(row[2]),
            "pending_records": int(row[3]),
            "previously_scored_records": int(row[4]),
        }
        for row in per_source_rows
    }
    duplicate_records = connection.execute(
        "SELECT count(*) FROM qa_ranked WHERE exact_dedup_rank > 1"
    ).fetchone()[0]
    decision_rows, distinct_decision_keys = connection.execute(
        """
        SELECT count(*), count(DISTINCT (source_id, record_id, content_hash))
        FROM decision_input
        """
    ).fetchone()
    matched_decision_keys = connection.execute(
        """
        SELECT count(*)
        FROM qa_ranked
        WHERE exact_dedup_rank = 1 AND has_existing_decision
        """
    ).fetchone()[0]

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "deduplication": {
            "algorithm": "SHA-256 exact content hash",
            "canonical_preference": "existing exact-key Qwen decision, then source_id and record_id",
            "duplicate_records": int(duplicate_records),
        },
        "filtering": {
            "quality_thresholds_applied": False,
            "structurally_invalid_records_excluded": int(invalid_input),
            "license_metadata_corrections": {"reddit-*": "NOASSERTION"},
            "rescore_rule": (
                "existing decision lacks an immutable model revision, has an invalid "
                "parse status, or source content exceeded the prior "
                f"{max_content_chars}-character prompt limit"
            ),
        },
        "input": {
            "files": [str(path) for path in source_files],
            "file_count": len(source_files),
            "records": int(total_input),
            "decisions_file": str(decisions.resolve()),
            "decision_rows": int(decision_rows),
            "distinct_decision_keys": int(distinct_decision_keys),
            "matched_canonical_decision_keys": int(matched_decision_keys),
        },
        "sets": sets,
        "per_source": per_source,
    }


def _publish_outputs(temporary_root: Path, targets: dict[str, Path]) -> None:
    mapping = {
        temporary_root / "qa_universe": targets["universe"],
        temporary_root / "qa_pending": targets["pending"],
        temporary_root / "qa_rescore": targets["rescore"],
        temporary_root / "qa_to_score": targets["to_score"],
        temporary_root / "qa_exact_duplicates.parquet": targets["duplicates"],
        temporary_root / "manifest.json": targets["manifest"],
    }
    for source, destination in mapping.items():
        source.rename(destination)
    shutil.rmtree(temporary_root)


def _partition_value(value: Any) -> str:
    rendered = str(value)
    if not rendered or "/" in rendered or rendered in {".", ".."}:
        raise ValueError(f"Unsafe source_id for a partition directory: {rendered!r}")
    return rendered


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


if __name__ == "__main__":
    main()
