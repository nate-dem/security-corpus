#!/usr/bin/env python3
"""Exact-deduplicate citation-paper metadata for a reproducible Qwen re-score."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT / "data" / "arxiv" / "normalized" /
    "source_id=arxiv" / "citation_metadata_for_qwen.parquet"
)
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "filtering" / "v4"


def main() -> None:
    args = _parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    universe = args.output_root / "citation_abstract_universe.parquet"
    duplicates = args.output_root / "citation_abstract_exact_duplicates.parquet"
    manifest = args.output_root / "citation_abstract_manifest.json"
    targets = (universe, duplicates, manifest)
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Refusing to replace: " + ", ".join(map(str, existing)))
    if args.overwrite:
        for path in existing:
            path.unlink()
    args.output_root.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    connection.execute(
        f"""
        CREATE VIEW metadata AS
        SELECT * FROM read_parquet('{_sql_path(args.input)}')
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE ranked AS
        SELECT *, row_number() OVER (
            PARTITION BY content_hash
            ORDER BY arxiv_id, record_id
        ) AS exact_dedup_rank,
        first_value(arxiv_id) OVER (
            PARTITION BY content_hash
            ORDER BY arxiv_id, record_id
        ) AS canonical_arxiv_id,
        first_value(record_id) OVER (
            PARTITION BY content_hash
            ORDER BY arxiv_id, record_id
        ) AS canonical_record_id
        FROM metadata
        WHERE source_id IS NOT NULL AND trim(source_id) <> ''
          AND record_id IS NOT NULL AND trim(record_id) <> ''
          AND arxiv_id IS NOT NULL AND trim(arxiv_id) <> ''
          AND content_hash IS NOT NULL AND trim(content_hash) <> ''
          AND abstract IS NOT NULL AND trim(abstract) <> ''
        """
    )
    columns = [
        row[0] for row in connection.execute("DESCRIBE metadata").fetchall()
    ]
    projected = ", ".join(
        _legacy_license_expression() if column == "license" else _quote(column)
        for column in columns
    )
    connection.execute(
        f"""
        COPY (
            SELECT {projected} FROM ranked
            WHERE exact_dedup_rank = 1
            ORDER BY arxiv_id
        ) TO '{_sql_path(universe)}'
          (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
        """
    )
    connection.execute(
        f"""
        COPY (
            SELECT
                content_hash,
                canonical_arxiv_id,
                canonical_record_id,
                arxiv_id AS duplicate_arxiv_id,
                record_id AS duplicate_record_id
            FROM ranked
            WHERE exact_dedup_rank > 1
            ORDER BY content_hash, duplicate_arxiv_id
        ) TO '{_sql_path(duplicates)}'
          (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    input_records = connection.execute("SELECT count(*) FROM metadata").fetchone()[0]
    valid_records, unique_records = connection.execute(
        "SELECT count(*), count(*) FILTER (WHERE exact_dedup_rank = 1) FROM ranked"
    ).fetchone()
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "input_records": int(input_records),
        "structurally_valid_records": int(valid_records),
        "exact_unique_records": int(unique_records),
        "exact_duplicate_records": int(valid_records - unique_records),
        "quality_thresholds_applied": False,
        "legacy_license_labels_normalized": True,
        "purpose": (
            "Re-score every unique abstract with a pinned Qwen model revision; "
            "the recovered decisions recorded only a mutable model name."
        ),
        "universe": str(universe.resolve()),
        "duplicates": str(duplicates.resolve()),
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    connection.close()
    print(json.dumps(payload, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _legacy_license_expression() -> str:
    return """
        CASE license
            WHEN 'Public Domain' THEN 'CC0-1.0'
            WHEN 'http://creativecommons.org/licenses/by/3.0/' THEN 'CC-BY-3.0'
            WHEN 'http://creativecommons.org/licenses/by-nc-sa/3.0/'
                THEN 'CC-BY-NC-SA-3.0'
            WHEN 'http://creativecommons.org/licenses/publicdomain/'
                THEN 'Public Domain'
            ELSE license
        END AS license
    """.strip()


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


if __name__ == "__main__":
    main()
