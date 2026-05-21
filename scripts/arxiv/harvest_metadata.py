#!/usr/bin/env python3
"""Harvest arXiv metadata via OAI-PMH for a given category.

Adapted from Arxiv-Scraper (~/Desktop/Arxiv-Scraper/src/ingest/harvest_metadata.py).

Outputs one JSONL file per month (YYMM.jsonl) under the data directory.
Each line is an OAI-PMH record parsed to JSON via xmltodict.

Resumable: skips months for which a JSONL file already exists.

Usage:
    python scripts/arxiv/harvest_metadata.py \\
        --from-date 2007-01-01 \\
        --data-dir data/arxiv/raw/metadata/cs_CR
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta

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
        description="Harvest arXiv metadata via OAI-PMH"
    )
    p.add_argument(
        "--category", default="cs:cs:CR",
        help="arXiv OAI-PMH setSpec (default: cs:cs:CR)",
    )
    p.add_argument("--from-date", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument(
        "--until-date", default=None,
        help="End date (YYYY-MM-DD), default=today",
    )
    p.add_argument(
        "--data-dir", default="data/arxiv/raw/metadata/cs_CR",
        help="Output directory for JSONL files",
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


def _month_ranges(from_date: str, until_date: str):
    """Yield (range_from, range_until, month_key) for each month in the range."""
    start = datetime.strptime(from_date, "%Y-%m-%d")
    end = datetime.strptime(until_date, "%Y-%m-%d")

    current = start
    while current <= end:
        year, month = current.year, current.month
        month_key = current.strftime("%y%m")

        month_start = max(datetime(year, month, 1), start)
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        month_end = min(next_month - timedelta(days=1), end)

        yield (
            month_start.strftime("%Y-%m-%d"),
            month_end.strftime("%Y-%m-%d"),
            month_key,
        )

        current = next_month


def harvest(sickle: Sickle, category: str, from_date: str, until_date: str,
            out_dir: str):
    """Harvest records month-by-month, writing YYMM.jsonl files."""
    existing = {
        f[:4] for f in os.listdir(out_dir)
        if f.endswith(".jsonl") and len(f) == 10 and f[:4].isdigit()
    }
    if existing:
        logger.info("Found %d existing month files — will skip them", len(existing))

    for range_from, range_until, month_key in _month_ranges(from_date, until_date):
        if month_key in existing:
            logger.info("Skipping %s (already harvested)", month_key)
            continue

        logger.info("Harvesting %s (%s to %s)", month_key, range_from, range_until)
        params = {
            "metadataPrefix": "arXiv",
            "set": category,
            "from": range_from,
            "until": range_until,
        }

        filepath = os.path.join(out_dir, f"{month_key}.jsonl")
        count = 0
        try:
            records = sickle.ListRecords(**params)
            with open(filepath, "w", encoding="utf-8") as fh:
                for rec in records:
                    try:
                        obj = xmltodict.parse(rec.raw)
                        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        count += 1
                        if count % 500 == 0:
                            logger.info("  %s: %d records so far", month_key, count)
                    except Exception as e:
                        logger.error("Error parsing record: %s", e)
            logger.info("Completed %s: %d records", month_key, count)
        except Exception as e:
            # If no records exist for this month, Sickle raises NoRecordsMatch
            if "noRecordsMatch" in str(e):
                logger.info("No records for %s", month_key)
                # Write empty file so we don't re-query
                open(filepath, "w").close()
            else:
                logger.error("Error harvesting %s: %s", month_key, e)


def main():
    args = parse_args()
    until_date = args.until_date or datetime.now().strftime("%Y-%m-%d")

    for d in (args.from_date, until_date):
        try:
            datetime.strptime(d, "%Y-%m-%d")
        except ValueError:
            logger.error("Dates must be YYYY-MM-DD format")
            sys.exit(1)

    os.makedirs(args.data_dir, exist_ok=True)

    session = _make_session()
    requests.get = session.get  # Sickle uses requests.get internally

    sickle = Sickle(args.oai_endpoint)
    logger.info("Harvesting %s from %s to %s", args.category, args.from_date, until_date)

    harvest(sickle, args.category, args.from_date, until_date, args.data_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
