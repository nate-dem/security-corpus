#!/usr/bin/env python3
"""Download individual arXiv paper sources from arxiv.org/e-print.

Reads metadata JSONL files to get paper IDs, then downloads LaTeX source
archives one at a time with rate limiting.

Resumable: skips papers whose files already exist on disk.

Usage:
    python scripts/arxiv_download_sources.py \\
        --metadata-dir data/arxiv/raw/metadata/cs_CR \\
        --output-dir data/arxiv/raw/source/downloads
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SOURCE_URL = "https://arxiv.org/src/{arxiv_id}"


def parse_args():
    p = argparse.ArgumentParser(description="Download arXiv LaTeX sources")
    p.add_argument(
        "--metadata-dir",
        default="data/arxiv/raw/metadata/cs_CR",
        help="Directory containing YYMM.jsonl metadata files",
    )
    p.add_argument(
        "--output-dir",
        default="data/arxiv/raw/source/downloads",
        help="Directory to save downloaded source archives",
    )
    p.add_argument(
        "--rate-limit",
        type=float, default=3.0,
        help="Seconds between requests (default: 3.0)",
    )
    p.add_argument(
        "--months",
        type=str, default=None,
        help="Comma-separated list of months to download (e.g. 2401,2402). Default: all",
    )
    p.add_argument(
        "--id-file",
        type=str, default=None,
        help="Plain text file of arXiv IDs (one per line), alternative to metadata JSONL",
    )
    return p.parse_args()


def _make_session():
    """Create a session with conservative retry logic."""
    session = requests.Session()
    retry = Retry(
        total=2,
        status_forcelist=[500, 502],
        backoff_factor=1,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _arxiv_id_to_filename(arxiv_id: str) -> str:
    """Convert an arXiv ID to a safe filename.

    Old format ``cs/0601001`` becomes ``cs-0601001``.
    New format ``2401.12345`` stays as-is.
    """
    return arxiv_id.replace("/", "-")


def _arxiv_id_to_yymm(arxiv_id: str) -> str:
    """Extract the YYMM portion from an arXiv ID.

    ``2401.12345`` → ``2401``
    ``cs/0601001`` → ``0601``
    """
    if "/" in arxiv_id:
        # old format: category/YYMMNNN
        return arxiv_id.split("/")[1][:4]
    return arxiv_id.split(".")[0]


def _load_paper_ids(metadata_dir: Path, months: list[str] | None) -> list[str]:
    """Read metadata JSONL and extract arXiv IDs."""
    ids: list[str] = []
    jsonl_files = sorted(metadata_dir.glob("*.jsonl"))

    if months:
        jsonl_files = [f for f in jsonl_files if f.stem in months]

    for jsonl_path in jsonl_files:
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue

                rec = raw.get("record", raw)
                header = rec.get("header", {})
                if header.get("@status") == "deleted":
                    continue

                metadata = rec.get("metadata", {})
                arxiv_meta = metadata.get("arXiv", {})
                arxiv_id = arxiv_meta.get("id", "")
                if arxiv_id:
                    ids.append(arxiv_id)

    return ids


def download_papers(paper_ids: list[str], output_dir: Path,
                    rate_limit: float):
    """Download source archives for a list of arXiv IDs."""
    session = _make_session()
    total = len(paper_ids)
    downloaded = 0
    skipped = 0
    failed = 0

    for i, arxiv_id in enumerate(paper_ids):
        yymm = _arxiv_id_to_yymm(arxiv_id)
        safe_name = _arxiv_id_to_filename(arxiv_id)
        month_dir = output_dir / yymm
        month_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing file (could be .tar.gz, .gz, or .pdf)
        existing = [f for f in month_dir.glob(f"{safe_name}.*")
                    if not f.name.endswith(".failed")]
        if existing:
            skipped += 1
            continue

        url = SOURCE_URL.format(arxiv_id=arxiv_id)
        try:
            resp = session.get(url, timeout=30)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 30))
                logger.warning(
                    "Rate limited — waiting %d seconds", retry_after
                )
                time.sleep(retry_after)
                resp = session.get(url, timeout=30)

            resp.raise_for_status()

            # Determine file extension from Content-Type
            content_type = resp.headers.get("Content-Type", "")
            if "application/x-eprint-tar" in content_type or "gzip" in content_type:
                ext = ".tar.gz"
            elif "application/pdf" in content_type:
                ext = ".pdf"
            else:
                ext = ".tar.gz"

            out_path = month_dir / f"{safe_name}{ext}"
            out_path.write_bytes(resp.content)
            downloaded += 1

            if (downloaded + skipped) % 100 == 0:
                logger.info(
                    "Progress: %d/%d (downloaded=%d, skipped=%d, failed=%d)",
                    i + 1, total, downloaded, skipped, failed,
                )

            # Only rate-limit after successful downloads
            time.sleep(rate_limit)

        except Exception as e:
            failed += 1
            logger.error("Failed to download %s: %s", arxiv_id, e)
            # Write a marker so we don't retry on restart
            marker = month_dir / f"{safe_name}.failed"
            marker.write_text(str(e), encoding="utf-8")

    logger.info(
        "Complete: downloaded=%d, skipped=%d, failed=%d, total=%d",
        downloaded, skipped, failed, total,
    )


def _load_ids_from_file(id_file: Path) -> list[str]:
    """Read arXiv IDs from a plain text file (one per line)."""
    ids: list[str] = []
    with open(id_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    return ids


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.id_file:
        id_file = Path(args.id_file)
        if not id_file.exists():
            logger.error("ID file not found: %s", id_file)
            sys.exit(1)
        logger.info("Loading paper IDs from %s", id_file)
        paper_ids = _load_ids_from_file(id_file)
    else:
        metadata_dir = Path(args.metadata_dir)
        if not metadata_dir.is_dir():
            logger.error("Metadata directory not found: %s", metadata_dir)
            sys.exit(1)
        months = None
        if args.months:
            months = [m.strip() for m in args.months.split(",")]
        logger.info("Loading paper IDs from %s", metadata_dir)
        paper_ids = _load_paper_ids(metadata_dir, months)

    logger.info("Found %d papers to download", len(paper_ids))

    if not paper_ids:
        logger.info("Nothing to download.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    download_papers(paper_ids, output_dir, args.rate_limit)


if __name__ == "__main__":
    main()
