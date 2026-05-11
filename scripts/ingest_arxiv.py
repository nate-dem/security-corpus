#!/usr/bin/env python3
"""Ingest preprocessed arXiv papers into Parquet.

Expects preprocessing scripts to have already run:
  1. scripts/arxiv_harvest_metadata.py  → data/arxiv/raw/metadata/cs_CR/
  2. scripts/arxiv_download_sources.py  → data/arxiv/raw/source/downloads/
  3. scripts/arxiv_normalize_sources.py → data/arxiv/raw/source/normalized/

Usage:
    python scripts/ingest_arxiv.py
"""

from pathlib import Path

from ingest.pipeline import ingest_and_store


def main():
    raw_dir = Path("data/arxiv/raw")
    output_dir = Path("data/arxiv/normalized")

    if not raw_dir.is_dir():
        print(f"Raw directory not found: {raw_dir}")
        print("Run the preprocessing scripts first:")
        print("  python scripts/arxiv_harvest_metadata.py --from-date 2007-01-01")
        print("  python scripts/arxiv_download_sources.py")
        print("  python scripts/arxiv_normalize_sources.py")
        return

    count = ingest_and_store(raw_dir, source="arxiv", output_dir=output_dir)
    print(f"arxiv: {count} records")


if __name__ == "__main__":
    main()
