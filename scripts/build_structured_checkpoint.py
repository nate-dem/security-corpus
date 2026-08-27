#!/usr/bin/env python3
"""Build a structurally valid, policy-neutral checkpoint of non-QA sources.

The script applies no quality or length threshold. It removes only malformed
rows and resolves duplicate MITRE ATT&CK entity IDs by the newest STIX
``modified`` timestamp. Exact-content duplicates are reported, not dropped,
so the researcher can make the final cross-record deduplication decision.
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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "checkpoints" / "structured-v1"
DEFAULT_REPORT = ROOT / "reports" / "recovery" / "structured-v1"

SOURCE_PATTERNS = {
    "nvd": "data/nvd/normalized/**/*.parquet",
    "cisa-kev": "data/cisa-kev/normalized/**/*.parquet",
    "mitre-attack": "data/mitre-attack/normalized/**/*.parquet",
    "mitre-cwe": "data/mitre-cwe/normalized/**/*.parquet",
    "capec": "data/mitre-capec/normalized/**/*.parquet",
    "sigma": "data/sigma/normalized/**/*.parquet",
    "cloudtrail-flaws": "data/cloudtrail/normalized/**/*.parquet",
}

# Corrections for legacy normalized checkpoints whose license labels were
# inaccurate or asserted rights that were not documented.
LICENSE_METADATA_CORRECTIONS = {
    "nvd": "CVE Terms of Use / NIST public data",
    "cisa-kev": "CC0-1.0",
    "sigma": "DRL-1.1",
    "cloudtrail-flaws": "NOASSERTION",
}

REQUIRED_COLUMNS = {
    "source_id",
    "source_record_id",
    "record_id",
    "content",
    "content_length",
    "content_hash",
    "license",
}

STRUCTURAL_VALIDITY = """
    source_id IS NOT NULL AND trim(source_id) <> ''
    AND source_record_id IS NOT NULL AND trim(source_record_id) <> ''
    AND record_id IS NOT NULL AND trim(record_id) <> ''
    AND content IS NOT NULL AND trim(content) <> ''
    AND content_length IS NOT NULL AND content_length > 0
    AND content_hash IS NOT NULL AND length(content_hash) = 64
    AND license IS NOT NULL AND trim(license) <> ''
"""


def main() -> None:
    args = _parse_args()
    inputs = _discover_inputs(args.root)
    missing_sources = sorted(set(SOURCE_PATTERNS) - set(inputs))
    if missing_sources:
        raise FileNotFoundError(
            "No normalized Parquet found for: " + ", ".join(missing_sources)
        )
    _prepare_destination(args.output, overwrite=args.overwrite)
    _prepare_destination(args.report, overwrite=args.overwrite)

    temporary_output = args.output.parent / f".{args.output.name}-{os.getpid()}.tmp"
    temporary_report = args.report.parent / f".{args.report.name}-{os.getpid()}.tmp"
    temporary_output.mkdir(parents=True)
    temporary_report.mkdir(parents=True)
    connection = duckdb.connect()
    connection.execute("SET preserve_insertion_order = false")
    connection.execute(
        f"SET temp_directory = '{_sql_path(temporary_output / 'duckdb-tmp')}'"
    )
    connection.execute("SET memory_limit = '8GB'")
    connection.execute(
        """
        CREATE TEMP TABLE checkpoint_metadata (
            source_id VARCHAR,
            record_id VARCHAR,
            content_hash VARCHAR,
            content_length BIGINT
        )
        """
    )

    try:
        per_source = {}
        for source_id, paths in inputs.items():
            per_source[source_id] = _process_source(
                connection,
                source_id,
                paths,
                temporary_output,
            )
        duplicate_summary = _write_duplicate_audit(connection, temporary_report)
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "policy": {
                "quality_or_length_thresholds_applied": False,
                "exact_content_duplicates_removed": False,
                "structurally_invalid_rows_removed": True,
                "mitre_attack_duplicate_record_ids": (
                    "keep newest raw.modified value, then deterministic tie-break"
                ),
            },
            "sources": per_source,
            "exact_duplicates": duplicate_summary,
            "totals": _totals(connection),
        }
        (temporary_report / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        connection.close()
        _publish(temporary_output, args.output)
        _publish(temporary_report, args.report)
    except Exception:
        connection.close()
        shutil.rmtree(temporary_output, ignore_errors=True)
        shutil.rmtree(temporary_report, ignore_errors=True)
        raise

    print(json.dumps(manifest, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _discover_inputs(root: Path) -> dict[str, list[Path]]:
    discovered = {}
    for source_id, pattern in SOURCE_PATTERNS.items():
        files = sorted(path.resolve() for path in root.glob(pattern) if path.is_file())
        if files:
            discovered[source_id] = files
    return discovered


def _prepare_destination(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace existing path: {path}")
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def _process_source(
    connection: duckdb.DuckDBPyConnection,
    source_id: str,
    paths: list[Path],
    output_root: Path,
) -> dict[str, Any]:
    view = "source_input"
    connection.execute(f"DROP VIEW IF EXISTS {view}")
    file_list = ", ".join(f"'{_sql_path(path)}'" for path in paths)
    connection.execute(
        f"""
        CREATE TEMP VIEW {view} AS
        SELECT * FROM read_parquet(
            [{file_list}],
            union_by_name = true,
            hive_partitioning = false,
            filename = true
        )
        """
    )
    columns = [row[0] for row in connection.execute(f"DESCRIBE {view}").fetchall()]
    missing = sorted(REQUIRED_COLUMNS - set(columns))
    if missing:
        raise ValueError(f"{source_id} is missing columns: {', '.join(missing)}")
    source_values = connection.execute(
        f"SELECT DISTINCT source_id FROM {view} ORDER BY source_id"
    ).fetchall()
    if source_values != [(source_id,)]:
        raise ValueError(f"{source_id} files contain unexpected source_id values: {source_values}")

    input_records, input_tokens, invalid_records, invalid_tokens = connection.execute(
        f"""
        SELECT
            count(*),
            coalesce(sum(content_length), 0),
            count(*) FILTER (WHERE NOT ({STRUCTURAL_VALIDITY})),
            coalesce(sum(content_length) FILTER (WHERE NOT ({STRUCTURAL_VALIDITY})), 0)
        FROM {view}
        """
    ).fetchone()
    duplicate_record_ids = connection.execute(
        f"""
        SELECT coalesce(sum(copies - 1), 0)
        FROM (
            SELECT record_id, count(*) AS copies
            FROM {view}
            WHERE {STRUCTURAL_VALIDITY}
            GROUP BY record_id
            HAVING count(*) > 1
        )
        """
    ).fetchone()[0]
    if source_id != "mitre-attack" and duplicate_record_ids:
        raise ValueError(
            f"{source_id} has {duplicate_record_ids} duplicate record IDs; "
            "no structural resolution rule is defined"
        )

    original_columns = [column for column in columns if column != "filename"]
    projected = ", ".join(_quote(column) for column in original_columns)
    connection.execute("DROP TABLE IF EXISTS source_survivors")
    if source_id == "mitre-attack":
        connection.execute(
            f"""
            CREATE TEMP TABLE source_survivors AS
            SELECT {projected}
            FROM {view}
            WHERE {STRUCTURAL_VALIDITY}
            QUALIFY row_number() OVER (
                PARTITION BY record_id
                ORDER BY
                    try_cast(json_extract_string(raw, '$.modified') AS TIMESTAMPTZ)
                        DESC NULLS LAST,
                    content_hash,
                    filename
            ) = 1
            """
        )
    else:
        connection.execute(
            f"""
            CREATE TEMP TABLE source_survivors AS
            SELECT {projected}
            FROM {view}
            WHERE {STRUCTURAL_VALIDITY}
            """
        )

    input_licenses = _license_counts(connection, "source_survivors")
    corrected_license = LICENSE_METADATA_CORRECTIONS.get(source_id)
    if corrected_license is not None:
        connection.execute(
            "UPDATE source_survivors SET license = ?",
            [corrected_license],
        )
    output_licenses = _license_counts(connection, "source_survivors")

    output_records, output_tokens, exact_hashes, min_tokens, p50, p90, p99, max_tokens = (
        connection.execute(
            """
            SELECT
                count(*),
                coalesce(sum(content_length), 0),
                count(DISTINCT content_hash),
                min(content_length),
                quantile_cont(content_length, 0.50),
                quantile_cont(content_length, 0.90),
                quantile_cont(content_length, 0.99),
                max(content_length)
            FROM source_survivors
            """
        ).fetchone()
    )
    partition = output_root / f"source_id={source_id}"
    partition.mkdir()
    connection.execute(
        f"""
        COPY (
            SELECT * FROM source_survivors ORDER BY record_id
        ) TO '{_sql_path(partition / 'part-00000.parquet')}'
          (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
        """
    )
    connection.execute(
        """
        INSERT INTO checkpoint_metadata
        SELECT source_id, record_id, content_hash, content_length
        FROM source_survivors
        """
    )
    longest = connection.execute(
        """
        SELECT record_id, content_length
        FROM source_survivors
        ORDER BY content_length DESC, record_id
        LIMIT 10
        """
    ).fetchall()
    summary: dict[str, Any] = {
        "input_files": [str(path) for path in paths],
        "input_records": int(input_records),
        "input_tokens": int(input_tokens),
        "structurally_invalid_records": int(invalid_records),
        "structurally_invalid_tokens": int(invalid_tokens),
        "duplicate_record_id_rows_resolved": int(duplicate_record_ids),
        "input_license_counts": input_licenses,
        "output_license_counts": output_licenses,
        "license_metadata_correction": corrected_license,
        "output_records": int(output_records),
        "output_tokens": int(output_tokens),
        "distinct_content_hashes": int(exact_hashes),
        "length_tokens": {
            "min": int(min_tokens),
            "p50": float(p50),
            "p90": float(p90),
            "p99": float(p99),
            "max": int(max_tokens),
        },
        "longest_records": [
            {"record_id": row[0], "tokens": int(row[1])} for row in longest
        ],
    }
    if source_id == "nvd":
        statuses = connection.execute(
            """
            SELECT json_extract_string(raw, '$.cve.vulnStatus'), count(*),
                   coalesce(sum(content_length), 0)
            FROM source_survivors
            GROUP BY 1
            ORDER BY count(*) DESC, 1
            """
        ).fetchall()
        summary["vulnerability_statuses"] = {
            str(row[0]): {"records": int(row[1]), "tokens": int(row[2])}
            for row in statuses
        }
    return summary


def _license_counts(
    connection: duckdb.DuckDBPyConnection,
    relation: str,
) -> dict[str, int]:
    rows = connection.execute(
        f"SELECT license, count(*) FROM {relation} GROUP BY license ORDER BY license"
    ).fetchall()
    return {str(license_value): int(count) for license_value, count in rows}


def _write_duplicate_audit(
    connection: duckdb.DuckDBPyConnection,
    report_root: Path,
) -> dict[str, Any]:
    duplicate_groups, extra_records, cross_source_groups = connection.execute(
        """
        SELECT
            count(*),
            coalesce(sum(records - 1), 0),
            count(*) FILTER (WHERE sources > 1)
        FROM (
            SELECT content_hash, count(*) AS records,
                   count(DISTINCT source_id) AS sources
            FROM checkpoint_metadata
            GROUP BY content_hash
            HAVING count(*) > 1
        )
        """
    ).fetchone()
    destination = report_root / "exact_duplicate_records.parquet"
    connection.execute(
        f"""
        COPY (
            SELECT
                content_hash,
                count(*) OVER (PARTITION BY content_hash) AS copies,
                count(DISTINCT source_id) OVER (PARTITION BY content_hash) AS sources,
                source_id,
                record_id,
                content_length
            FROM checkpoint_metadata
            QUALIFY count(*) OVER (PARTITION BY content_hash) > 1
            ORDER BY content_hash, source_id, record_id
        ) TO '{_sql_path(destination)}'
          (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    return {
        "groups": int(duplicate_groups),
        "extra_records": int(extra_records),
        "cross_source_groups": int(cross_source_groups),
        "records_path": "exact_duplicate_records.parquet",
    }


def _totals(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    records, tokens, sources = connection.execute(
        """
        SELECT count(*), coalesce(sum(content_length), 0), count(DISTINCT source_id)
        FROM checkpoint_metadata
        """
    ).fetchone()
    return {"records": int(records), "tokens": int(tokens), "sources": int(sources)}


def _publish(temporary: Path, destination: Path) -> None:
    temporary.rename(destination)


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


if __name__ == "__main__":
    main()
