#!/usr/bin/env python3
"""Inventory recovered arXiv checkpoints and write deterministic restart lists.

This is a read-only audit of corpus data. It does not apply quality thresholds
or modify normalized papers. The generated ID lists separate what is preserved
locally from work that should be repeated with the hardened normalizer and a
pinned Qwen model revision.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import duckdb


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ingest.connectors.arxiv.metadata import build_metadata_index  # noqa: E402


DEFAULT_SEED_FULL = (
    ROOT / "data" / "training-clean-v1" / "normalized" /
    "source_id=arxiv" / "part-00000.parquet"
)
DEFAULT_CITATION_METADATA = (
    ROOT / "data" / "arxiv" / "normalized" /
    "source_id=arxiv" / "citation_metadata_for_qwen.parquet"
)
DEFAULT_CITATION_DECISIONS = (
    ROOT / "data" / "filtering" / "v3" /
    "qwen_citation_abstract_decisions.parquet"
)
DEFAULT_CITATION_FULL = (
    ROOT / "data" / "filtering" / "v3" /
    "qwen_citation_abstract_kept_full.parquet"
)
DEFAULT_OUTPUT = ROOT / "reports" / "recovery" / "arxiv"


def main() -> None:
    args = _parse_args()
    required = (
        args.seed_full,
        args.citation_metadata,
        args.citation_decisions,
        args.citation_full,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing recovered arXiv artifacts: " + ", ".join(map(str, missing)))
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to replace existing audit directory: {args.output_dir}"
        )
    if args.output_dir.exists():
        import shutil

        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    connection = duckdb.connect()
    _register_views(connection, args)
    audit = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": _artifact_summaries(connection, args),
        "full_text_checkpoints": _full_text_summary(connection),
        "normalization_status": _normalization_status(args.normalized_source_root),
        "seed_metadata": _seed_metadata_summary(args.seed_metadata_dir),
        "citation_filter": _citation_filter_summary(connection),
        "content_audit": _content_audit(connection),
        "restart": _restart_summary(connection, args),
    }
    _write_restart_lists(connection, args.output_dir)
    report_path = args.output_dir / "audit.json"
    report_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    connection.close()
    print(json.dumps(audit, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-full", type=Path, default=DEFAULT_SEED_FULL)
    parser.add_argument("--citation-metadata", type=Path, default=DEFAULT_CITATION_METADATA)
    parser.add_argument("--citation-decisions", type=Path, default=DEFAULT_CITATION_DECISIONS)
    parser.add_argument("--citation-full", type=Path, default=DEFAULT_CITATION_FULL)
    parser.add_argument(
        "--seed-metadata-dir",
        type=Path,
        default=ROOT / "data" / "arxiv" / "raw" / "metadata" / "cs_CR",
    )
    parser.add_argument(
        "--normalized-source-root",
        type=Path,
        default=ROOT / "data" / "arxiv" / "raw" / "source" / "normalized",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _register_views(connection: duckdb.DuckDBPyConnection, args: argparse.Namespace) -> None:
    for view, path in (
        ("seed_full", args.seed_full),
        ("citation_metadata", args.citation_metadata),
        ("citation_decisions", args.citation_decisions),
        ("citation_full", args.citation_full),
    ):
        connection.execute(
            f"CREATE VIEW {view} AS SELECT * FROM read_parquet('{_sql_path(path)}')"
        )
    connection.execute(
        """
        CREATE VIEW recovered_full AS
        SELECT 'cs.CR seed' AS checkpoint, * FROM seed_full
        UNION ALL BY NAME
        SELECT 'citation-selected' AS checkpoint, * FROM citation_full
        """
    )


def _artifact_summaries(
    connection: duckdb.DuckDBPyConnection,
    args: argparse.Namespace,
) -> dict[str, Any]:
    summaries = {}
    for view, path in (
        ("seed_full", args.seed_full),
        ("citation_metadata", args.citation_metadata),
        ("citation_decisions", args.citation_decisions),
        ("citation_full", args.citation_full),
    ):
        columns = {
            row[0] for row in connection.execute(f"DESCRIBE {view}").fetchall()
        }
        selections = ["count(*)"]
        labels = ["records"]
        for column, expression in (
            ("arxiv_id", "count(DISTINCT arxiv_id)"),
            ("content_hash", "count(DISTINCT content_hash)"),
            ("content_length", "coalesce(sum(content_length), 0)"),
        ):
            if column in columns:
                selections.append(expression)
                if column == "content_length":
                    labels.append("tokens")
                elif column == "content_hash":
                    labels.append("distinct_content_hashes")
                else:
                    labels.append("distinct_arxiv_ids")
        values = connection.execute(
            f"SELECT {', '.join(selections)} FROM {view}"
        ).fetchone()
        summary = {label: int(value) for label, value in zip(labels, values)}
        summary.update({"path": str(path.resolve()), "bytes": path.stat().st_size})
        summaries[view] = summary
    return summaries


def _full_text_summary(connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    records, distinct_ids, distinct_hashes, tokens = connection.execute(
        """
        SELECT
            count(*),
            count(DISTINCT arxiv_id),
            count(DISTINCT content_hash),
            coalesce(sum(content_length), 0)
        FROM recovered_full
        """
    ).fetchone()
    deduplicated_records, deduplicated_tokens = connection.execute(
        """
        SELECT count(*), coalesce(sum(content_length), 0)
        FROM (
            SELECT *, row_number() OVER (
                PARTITION BY content_hash
                ORDER BY checkpoint, arxiv_id
            ) AS rank
            FROM recovered_full
        )
        WHERE rank = 1
        """
    ).fetchone()
    licenses = connection.execute(
        """
        SELECT license, count(*), coalesce(sum(content_length), 0)
        FROM recovered_full
        GROUP BY license
        ORDER BY count(*) DESC, license
        """
    ).fetchall()
    formats = connection.execute(
        """
        SELECT source_format, count(*), coalesce(sum(content_length), 0)
        FROM recovered_full
        GROUP BY source_format
        ORDER BY count(*) DESC, source_format
        """
    ).fetchall()
    return {
        "records": int(records),
        "distinct_arxiv_ids": int(distinct_ids),
        "distinct_content_hashes": int(distinct_hashes),
        "tokens_before_exact_deduplication": int(tokens),
        "records_after_exact_deduplication": int(deduplicated_records),
        "tokens_after_exact_deduplication": int(deduplicated_tokens),
        "licenses": {
            str(row[0]): {"records": int(row[1]), "tokens": int(row[2])}
            for row in licenses
        },
        "source_formats": {
            str(row[0]): {"records": int(row[1]), "tokens": int(row[2])}
            for row in formats
        },
    }


def _normalization_status(normalized_root: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    versions: Counter[str] = Counter()
    error_types: Counter[str] = Counter()
    for path in normalized_root.glob("*/*/status.json"):
        counts["status_files"] += 1
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            counts["unreadable_status"] += 1
            continue
        if status.get("completed"):
            counts["completed"] += 1
        else:
            counts["incomplete"] += 1
        if status.get("auto_ignore"):
            counts["auto_ignore"] += 1
        if status.get("tex_merged"):
            counts["tex_merged"] += 1
        if status.get("pdf_extracted"):
            counts["pdf_extracted"] += 1
        version = status.get("normalizer_version") or "unversioned-legacy"
        versions[str(version)] += 1
        for error in status.get("errors") or []:
            error_types[str(error).split(":", 1)[0][:120]] += 1
    counts["main_tex_files"] = sum(1 for _ in normalized_root.glob("*/*/main.tex"))
    counts["main_txt_files"] = sum(1 for _ in normalized_root.glob("*/*/main.txt"))
    return {
        **dict(counts),
        "versions": dict(versions),
        "top_error_types": dict(error_types.most_common(25)),
    }


def _seed_metadata_summary(metadata_dir: Path) -> dict[str, Any]:
    index = build_metadata_index(metadata_dir)
    return {
        "directory": str(metadata_dir.resolve()),
        "jsonl_files": sum(1 for _ in metadata_dir.glob("*.jsonl")),
        "parsed_unique_records": len(index),
    }


def _citation_filter_summary(connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    parse_status = connection.execute(
        """
        SELECT qwen_parse_status, count(*)
        FROM citation_decisions
        GROUP BY qwen_parse_status
        ORDER BY count(*) DESC
        """
    ).fetchall()
    models = connection.execute(
        """
        SELECT qwen_model, qwen_prompt_version, count(*)
        FROM citation_decisions
        GROUP BY ALL
        ORDER BY count(*) DESC
        """
    ).fetchall()
    decision_columns = {
        row[0] for row in connection.execute("DESCRIBE citation_decisions").fetchall()
    }
    return {
        "parse_status": {str(row[0]): int(row[1]) for row in parse_status},
        "model_prompt_pairs": [
            {"model": row[0], "prompt_version": row[1], "records": int(row[2])}
            for row in models
        ],
        "has_immutable_model_revision": "qwen_model_revision" in decision_columns,
        "requires_rescore_for_reproducible_release": (
            "qwen_model_revision" not in decision_columns
        ),
    }


def _content_audit(connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    include_pattern = r"\\(input|include|subfile|import|subimport|inputfrom|subinputfrom)"
    unresolved, legacy_missing, empty = connection.execute(
        """
        SELECT
            count(*) FILTER (WHERE regexp_matches(content, ?)),
            count(*) FILTER (WHERE content LIKE '%Skipped missing or circular include:%'),
            count(*) FILTER (WHERE content IS NULL OR trim(content) = '')
        FROM recovered_full
        """,
        [include_pattern],
    ).fetchone()
    return {
        "records_with_unresolved_include_commands": int(unresolved),
        "records_with_legacy_missing_or_circular_markers": int(legacy_missing),
        "empty_records": int(empty),
        "interpretation": (
            "Include-pattern counts are audit flags, not automatic exclusions; "
            "commands may occur in macro definitions or examples."
        ),
    }


def _restart_summary(
    connection: duckdb.DuckDBPyConnection,
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected_ids = connection.execute(
        "SELECT count(DISTINCT arxiv_id) FROM citation_full"
    ).fetchone()[0]
    citation_ids = connection.execute(
        "SELECT count(DISTINCT arxiv_id) FROM citation_metadata"
    ).fetchone()[0]
    seed_ids = connection.execute(
        "SELECT count(DISTINCT arxiv_id) FROM seed_full"
    ).fetchone()[0]
    downloads_root = args.normalized_source_root.parent / "downloads"
    return {
        "preserved_full_text_checkpoint_ids": int(seed_ids + selected_ids),
        "seed_ids_to_reextract_with_latex_v2": int(seed_ids),
        "citation_metadata_ids_to_rescore_with_pinned_qwen": int(citation_ids),
        "currently_selected_citation_ids_to_reextract": int(selected_ids),
        "local_source_downloads_present": downloads_root.is_dir(),
        "local_source_downloads_path": str(downloads_root.resolve()),
        "note": (
            "Recovered full text remains a checkpoint. Re-extraction is required "
            "for a research-ready release because legacy statuses do not identify "
            "the normalizer version or include-resolution failures."
        ),
    }


def _write_restart_lists(connection: duckdb.DuckDBPyConnection, output_dir: Path) -> None:
    _copy_query(
        connection,
        """
        SELECT checkpoint, arxiv_id, record_id, content_hash, content_length
        FROM recovered_full
        ORDER BY checkpoint, arxiv_id
        """,
        output_dir / "preserved_full_text_checkpoints.tsv",
        header=True,
    )
    _copy_query(
        connection,
        "SELECT DISTINCT arxiv_id FROM seed_full ORDER BY arxiv_id",
        output_dir / "seed_reextract_ids.txt",
        header=False,
    )
    _copy_query(
        connection,
        "SELECT DISTINCT arxiv_id FROM citation_metadata ORDER BY arxiv_id",
        output_dir / "citation_abstract_rescore_ids.txt",
        header=False,
    )
    _copy_query(
        connection,
        "SELECT DISTINCT arxiv_id FROM citation_full ORDER BY arxiv_id",
        output_dir / "citation_currently_selected_reextract_ids.txt",
        header=False,
    )


def _copy_query(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    destination: Path,
    *,
    header: bool,
) -> None:
    connection.execute(
        f"""
        COPY ({query}) TO '{_sql_path(destination)}'
        (FORMAT CSV, DELIMITER '\t', HEADER {str(header).lower()}, QUOTE '')
        """
    )


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


if __name__ == "__main__":
    main()
