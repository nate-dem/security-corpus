#!/usr/bin/env python3
"""Unified ingestion entrypoint for normalized Parquet output."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))

from ingest.commands import main


if __name__ == "__main__":
    raise SystemExit(main())
