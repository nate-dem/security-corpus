#!/usr/bin/env python3
"""Harvest OAI-PMH metadata for individual arXiv papers by ID.

Uses the OAI-PMH ``GetRecord`` verb to fetch metadata for specific papers
(e.g., those discovered via citation expansion). Outputs the same JSONL
format as ``scripts/arxiv/harvest_metadata.py`` so the connector can read both.

Resumable: tracks which IDs have been fetched in a checkpoint file.

Usage:
    python scripts/arxiv/harvest_citation_metadata.py \\
        --id-file data/arxiv/raw/metadata/citations/discovered_ids.txt \\
        --output-dir data/arxiv/raw/metadata/citations
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
import xmltodict
from requests.adapters import HTTPAdapter
from sickle import Sickle
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Harvest OAI-PMH metadata for individual arXiv papers"
    )
    p.add_argument(
        "--id-file",
        required=True,
        help="Plain text file of arXiv IDs (one per line)",
    )
    p.add_argument(
        "--output-dir",
        default="data/arxiv/raw/metadata/citations",
        help="Output directory for JSONL files",
    )
    p.add_argument(
        "--checkpoint-file",
        default=None,
        help="Checkpoint file for resumability (default: {output-dir}/.harvest_checkpoint)",
    )
    p.add_argument(
        "--rate-limit",
        type=float, default=3.0,
        help="Seconds between requests (default: 3.0)",
    )
    p.add_argument(
        "--oai-endpoint",
        default="https://oaipmh.arxiv.org/oai",
        help="OAI-PMH endpoint URL",
    )
    return p.parse_args()


def _make_session():
    """Create a requests.Session that retries on HTTP 503 with Retry-After."""
    session = requests.Session()
    retry = Retry(
        total=None,
        status_forcelist=[503],
        respect_retry_after_header=True,
        backoff_factor=0,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _arxiv_id_to_yymm(arxiv_id: str) -> str:
    """Extract the YYMM portion from an arXiv ID."""
    if "/" in arxiv_id:
        return arxiv_id.split("/")[1][:4]
    return arxiv_id.split(".")[0]


def _load_ids(id_file: Path) -> list[str]:
    """Read arXiv IDs from a plain text file."""
    ids = []
    with open(id_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    return ids


def _load_checkpoint(checkpoint_file: Path) -> set[str]:
    """Load set of already-fetched arXiv IDs from checkpoint."""
    if not checkpoint_file.exists():
        return set()
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _save_checkpoint(checkpoint_file: Path, arxiv_id: str):
    """Append a single ID to the checkpoint file."""
    with open(checkpoint_file, "a", encoding="utf-8") as f:
        f.write(arxiv_id + "\n")


def harvest_by_id(
    sickle: Sickle,
    arxiv_ids: list[str],
    output_dir: Path,
    checkpoint_file: Path,
    rate_limit: float,
):
    """Fetch metadata for each arXiv ID via GetRecord, writing YYMM.jsonl files."""
    completed = _load_checkpoint(checkpoint_file)
    remaining = [aid for aid in arxiv_ids if aid not in completed]

    logger.info(
        "Total: %d, already fetched: %d, remaining: %d",
        len(arxiv_ids), len(completed), len(remaining),
    )

    # Group by YYMM for organized output
    # We keep file handles open per-month and append
    file_handles: dict[str, object] = {}
    fetched = 0
    failed = 0

    try:
        for i, arxiv_id in enumerate(remaining):
            oai_identifier = f"oai:arXiv.org:{arxiv_id}"
            yymm = _arxiv_id_to_yymm(arxiv_id)

            try:
                record = sickle.GetRecord(
                    identifier=oai_identifier,
                    metadataPrefix="arXiv",
                )
                obj = xmltodict.parse(record.raw)

                # Get or open file handle for this month
                if yymm not in file_handles:
                    filepath = output_dir / f"{yymm}.jsonl"
                    file_handles[yymm] = open(filepath, "a", encoding="utf-8")

                file_handles[yymm].write(
                    json.dumps(obj, ensure_ascii=False) + "\n"
                )
                fetched += 1
                _save_checkpoint(checkpoint_file, arxiv_id)

            except Exception as e:
                err_str = str(e)
                if "idDoesNotExist" in err_str:
                    logger.warning("Paper not found: %s", arxiv_id)
                else:
                    logger.error("Error fetching %s: %s", arxiv_id, e)
                failed += 1
                # Still checkpoint so we don't retry
                _save_checkpoint(checkpoint_file, arxiv_id)

            if (i + 1) % 100 == 0:
                logger.info(
                    "Progress: %d/%d (fetched=%d, failed=%d)",
                    i + 1, len(remaining), fetched, failed,
                )

            time.sleep(rate_limit)

    finally:
        for fh in file_handles.values():
            fh.close()

    logger.info(
        "Complete: fetched=%d, failed=%d, total=%d",
        fetched, failed, len(remaining),
    )


def main():
    args = parse_args()
    id_file = Path(args.id_file)
    output_dir = Path(args.output_dir)
    checkpoint_file = Path(
        args.checkpoint_file or os.path.join(args.output_dir, ".harvest_checkpoint")
    )

    if not id_file.exists():
        logger.error("ID file not found: %s", id_file)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    arxiv_ids = _load_ids(id_file)
    logger.info("Loaded %d arXiv IDs from %s", len(arxiv_ids), id_file)

    if not arxiv_ids:
        logger.info("Nothing to harvest.")
        return

    session = _make_session()
    requests.get = session.get

    sickle = Sickle(args.oai_endpoint)
    harvest_by_id(sickle, arxiv_ids, output_dir, checkpoint_file, args.rate_limit)
    logger.info("Done.")


if __name__ == "__main__":
    main()
