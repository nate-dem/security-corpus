#!/usr/bin/env python3
"""Extract and normalize downloaded arXiv LaTeX sources.

For each downloaded source archive:
  1. Extract to a temporary directory
  2. Find the main .tex file, inline includes, strip comments
  3. Write cleaned main.tex + status.json to the normalized directory

Uses the LaTeX processing functions from
``src/ingest/connectors/arxiv/latex_processing.py``.

Usage:
    python scripts/arxiv_normalize_sources.py \\
        --downloads-dir data/arxiv/raw/source/downloads \\
        --output-dir data/arxiv/raw/source/normalized \\
        --workers 8
"""

import argparse
import gzip
import json
import logging
import shutil
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract and normalize arXiv LaTeX sources"
    )
    p.add_argument(
        "--downloads-dir",
        default="data/arxiv/raw/source/downloads",
        help="Directory containing downloaded source archives",
    )
    p.add_argument(
        "--output-dir",
        default="data/arxiv/raw/source/normalized",
        help="Directory for normalized output",
    )
    p.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: cpu count)",
    )
    p.add_argument(
        "--months", type=str, default=None,
        help="Comma-separated months to process (e.g. 2401,2402). Default: all",
    )
    return p.parse_args()


def _extract_source(archive_path: Path, extract_dir: Path) -> bool:
    """Extract a .tar.gz, .gz, or .tar source archive to extract_dir."""
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try as tar.gz / tar first
        if tarfile.is_tarfile(str(archive_path)):
            with tarfile.open(archive_path) as tf:
                tf.extractall(path=extract_dir)
            return True
    except (tarfile.TarError, EOFError, OSError):
        pass

    try:
        # Try as plain gzip (single file)
        with gzip.open(archive_path, "rb") as gz:
            content = gz.read()
        # Write as a .tex file
        arxiv_id = archive_path.stem.replace(".tar", "")
        out_file = extract_dir / f"{arxiv_id}.tex"
        out_file.write_bytes(content)
        return True
    except (gzip.BadGzipFile, EOFError, OSError):
        pass

    return False


def _normalize_one(args: tuple) -> tuple[str, int, int, int]:
    """Normalize a single paper. Returns (arxiv_id, processed, skipped, failed)."""
    # Import here to avoid issues with multiprocessing on some platforms
    from ingest.connectors.arxiv.latex_processing import (
        check_auto_ignore,
        merge_project,
        write_status_json,
    )

    archive_path, output_dir = args
    yymm = archive_path.parent.name
    # Strip .tar.gz or .gz extension to get ID
    arxiv_id = archive_path.name
    for suffix in (".tar.gz", ".gz", ".pdf"):
        if arxiv_id.endswith(suffix):
            arxiv_id = arxiv_id[: -len(suffix)]
            break

    tgt = output_dir / yymm / arxiv_id
    tgt.mkdir(parents=True, exist_ok=True)

    # Already done?
    status_file = tgt / "status.json"
    if status_file.exists():
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                status = json.load(f)
            if status.get("completed", False):
                return (arxiv_id, 0, 1, 0)
        except Exception:
            pass

    # Skip PDF-only papers (no LaTeX to normalize)
    if archive_path.name.endswith(".pdf"):
        write_status_json(tgt, {
            "aid": arxiv_id,
            "timestamp": datetime.now().isoformat(),
            "completed": False,
            "errors": ["PDF-only source, no LaTeX"],
        })
        return (arxiv_id, 0, 0, 1)

    status = {
        "aid": arxiv_id,
        "timestamp": datetime.now().isoformat(),
        "completed": False,
        "tex_merged": False,
        "errors": [],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        extract_dir = Path(tmpdir) / arxiv_id
        if not _extract_source(archive_path, extract_dir):
            status["errors"].append("Failed to extract archive")
            write_status_json(tgt, status)
            return (arxiv_id, 0, 0, 1)

        if check_auto_ignore(extract_dir, arxiv_id):
            write_status_json(tgt, {
                "aid": arxiv_id,
                "timestamp": datetime.now().isoformat(),
                "completed": True,
                "auto_ignore": True,
            })
            return (arxiv_id, 0, 1, 0)

        try:
            merge_project(extract_dir, tgt / "main.tex")
            status["tex_merged"] = True
            status["completed"] = True
            write_status_json(tgt, status)
            return (arxiv_id, 1, 0, 0)
        except Exception as e:
            status["errors"].append(str(e))
            write_status_json(tgt, status)
            return (arxiv_id, 0, 0, 1)


def main():
    args = parse_args()
    downloads_dir = Path(args.downloads_dir)
    output_dir = Path(args.output_dir)

    if not downloads_dir.is_dir():
        logger.error("Downloads directory not found: %s", downloads_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all archives to process
    months = None
    if args.months:
        months = set(m.strip() for m in args.months.split(","))

    tasks = []
    for month_dir in sorted(downloads_dir.iterdir()):
        if not month_dir.is_dir():
            continue
        if months and month_dir.name not in months:
            continue
        for archive in sorted(month_dir.iterdir()):
            if archive.is_file():
                tasks.append((archive, output_dir))

    logger.info("Found %d archives to normalize", len(tasks))

    if not tasks:
        logger.info("Nothing to normalize.")
        return

    processed = skipped = failed = 0
    workers = args.workers

    if workers == 1:
        for task in tasks:
            _, p, s, f = _normalize_one(task)
            processed += p
            skipped += s
            failed += f
    else:
        with ProcessPoolExecutor(max_workers=workers) as exe:
            for arxiv_id, p, s, f in exe.map(_normalize_one, tasks):
                processed += p
                skipped += s
                failed += f

    logger.info(
        "Done: processed=%d, skipped=%d, failed=%d",
        processed, skipped, failed,
    )


if __name__ == "__main__":
    main()
