#!/usr/bin/env python3
"""
Build the DSIR target corpus from CTIBench + structured security sources.

Reads text from each source, filters empty strings, and writes a flat parquet
with a single `text` column. This file is the input to scripts/fit_dsir.py,
which fits γ_target via DSIRFitter.

Sources included:
    CTIBench (5 tasks, per-task text column — see CTIBENCH_SOURCES)
    BRON graph knowledge base
    GitHub Advisory (if ingested)
    YouTube security transcripts (if ingested)

FineWeb is intentionally excluded — it is the raw corpus being filtered,
not the target distribution.

Output:
    data/dsir/target_texts.parquet   (single column: text)

Usage:
    python3 scripts/build_dsir_target.py             # build target corpus
    python3 scripts/build_dsir_target.py --dry-run   # print counts, write nothing
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source configuration
# ---------------------------------------------------------------------------

# CTIBench: map local parquet filename → text column containing raw source text.
# The "Prompt" column is always skipped — it contains prompt-formatted text.
CTIBENCH_SOURCES: dict[str, str] = {
    "cti_mcq.parquet": "Question",
    "cti_rcm.parquet": "Description",
    "cti_vsp.parquet": "Description",
    "cti_ate.parquet": "Description",
    "cti_taa.parquet": "Text",
}

# Structured sources — all use `content` as the primary text column per NormalizedData schema.
# Entries may be file paths or directories (which glob for *.parquet).
STRUCTURED_SOURCES: list[str] = [
    "data/bron/normalized/source_id=bron/raw.parquet",
    "data/github-advisory/normalized/source_id=github-advisory/raw.parquet",
    "data/youtube-transcripts/normalized/source_id=youtube-transcripts/raw.parquet",
]

TEXT_COLUMN_STRUCTURED = "content"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_paths(raw_paths: list[str], repo_root: Path) -> list[Path]:
    """Expand file or directory entries into a flat list of existing parquet paths."""
    files: list[Path] = []
    for raw in raw_paths:
        p = repo_root / raw
        if p.is_dir():
            found = sorted(p.glob("*.parquet"))
            if not found:
                logger.warning(f"Directory has no *.parquet files, skipping: {p}")
            files.extend(found)
        elif p.exists():
            files.append(p)
        else:
            logger.warning(f"Source not found, skipping: {p}")
    return files


def _read_column(path: Path, column: str) -> list[str]:
    """Read a single text column from a parquet file, filtering empty/null strings."""
    try:
        # ParquetFile.read() skips Hive partition inference, avoiding the
        # schema-merge conflict that occurs when source_id=X is in the path
        # AND as a column inside the file with different encodings.
        table = pq.ParquetFile(str(path)).read()
    except Exception as exc:
        logger.warning(f"Could not read '{column}' from {path}: {exc}")
        return []
    if column not in table.column_names:
        logger.warning(f"{path} has no '{column}' column; available: {table.column_names}")
        return []
    texts = table[column].to_pylist()
    return [t for t in texts if isinstance(t, str) and t.strip()]


# ---------------------------------------------------------------------------
# Core collection
# ---------------------------------------------------------------------------

def collect_texts(
    repo_root: Path,
    ctibench_dir: Path,
) -> tuple[list[str], dict[str, int]]:
    """
    Collect all target texts from CTIBench + structured sources.

    Returns:
        texts:  flat list of non-empty strings
        counts: per-source row counts for logging
    """
    texts: list[str] = []
    counts: dict[str, int] = {}

    # --- CTIBench ---
    for filename, col in CTIBENCH_SOURCES.items():
        path = ctibench_dir / filename
        if not path.exists():
            logger.warning(f"CTIBench file not found, skipping: {path}")
            continue
        rows = _read_column(path, col)
        counts[filename] = len(rows)
        texts.extend(rows)
        logger.info(f"  {filename:25s} col={col:15s} → {len(rows):>5,} rows")

    # --- Structured sources ---
    for src_path in _resolve_paths(STRUCTURED_SOURCES, repo_root):
        rows = _read_column(src_path, TEXT_COLUMN_STRUCTURED)
        label = src_path.parent.name + "/" + src_path.name
        counts[label] = len(rows)
        texts.extend(rows)
        logger.info(f"  {label:40s} col=content → {len(rows):>5,} rows")

    return texts, counts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build DSIR target corpus from CTIBench + structured security sources."
    )
    parser.add_argument(
        "--ctibench-dir", default="data/ctibench",
        help="Directory containing CTIBench parquets (default: data/ctibench)",
    )
    parser.add_argument(
        "--output", default="data/dsir/target_texts.parquet",
        help="Output parquet path (default: data/dsir/target_texts.parquet)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print source counts without writing output.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    ctibench_dir = repo_root / args.ctibench_dir
    output_path = repo_root / args.output

    logger.info("Collecting target texts...")
    texts, counts = collect_texts(repo_root, ctibench_dir)

    logger.info(f"\nTotal: {len(texts):,} texts across {len(counts)} sources")

    if args.dry_run:
        logger.info("Dry run — not writing output.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"text": texts})
    pq.write_table(table, str(output_path), compression="snappy")
    logger.info(f"Written → {output_path}  ({len(texts):,} rows)")


if __name__ == "__main__":
    main()
