"""Helpers for parsing OAI-PMH metadata records harvested from arXiv.

The harvester (scripts/arxiv_harvest_metadata.py) writes JSONL where each line
is the result of ``xmltodict.parse(sickle_record.raw)``.  The arXiv-native
metadata prefix produces a nested dict rooted at ``record.metadata.arXiv``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ingest.utils import ARXIV_LICENSE_MAP, ARXIV_PERPETUAL_NON_EXCLUSIVE

logger = logging.getLogger(__name__)


def _parse_arxiv_id(oai_identifier: str) -> str:
    """Extract the bare arXiv ID from an OAI identifier.

    ``oai:arXiv.org:2401.12345`` → ``2401.12345``
    ``oai:arXiv.org:cs/0601001``  → ``cs/0601001``
    """
    prefix = "oai:arXiv.org:"
    if oai_identifier.startswith(prefix):
        return oai_identifier[len(prefix):]
    return oai_identifier


def _parse_authors(authors_field) -> list[str]:
    """Return a list of ``"Forenames Keyname"`` strings.

    The ``authors.author`` value from xmltodict is a *list* when there are
    multiple authors but a plain *dict* for a single author.
    """
    if authors_field is None:
        return []

    author_entries = authors_field.get("author", [])
    if isinstance(author_entries, dict):
        author_entries = [author_entries]

    names: list[str] = []
    for entry in author_entries:
        keyname = entry.get("keyname", "")
        forenames = entry.get("forenames", "")
        if forenames:
            names.append(f"{forenames} {keyname}".strip())
        else:
            names.append(keyname.strip())
    return names


def _parse_datestamp(datestamp: str) -> datetime:
    """Parse an OAI datestamp (``YYYY-MM-DD``) into a UTC datetime."""
    return datetime.strptime(datestamp, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _map_license(license_url: str | None) -> str:
    """Map an arXiv license URL to a human-readable constant.

    Falls back to the non-exclusive license constant for unknown URLs.
    """
    if not license_url:
        return ARXIV_PERPETUAL_NON_EXCLUSIVE
    return ARXIV_LICENSE_MAP.get(license_url, license_url)


def _parse_oai_record(record: dict) -> dict | None:
    """Extract a flat metadata dict from a parsed OAI-PMH record.

    Returns ``None`` for deleted or unparseable records.
    """
    rec = record.get("record", record)

    header = rec.get("header", {})
    if header.get("@status") == "deleted":
        return None

    metadata = rec.get("metadata", {})
    arxiv_meta = metadata.get("arXiv", {})
    if not arxiv_meta:
        return None

    arxiv_id = arxiv_meta.get("id", "")
    if not arxiv_id:
        arxiv_id = _parse_arxiv_id(header.get("identifier", ""))
    if not arxiv_id:
        return None

    categories_str = arxiv_meta.get("categories", "")
    categories = categories_str.split() if categories_str else []

    return {
        "arxiv_id": arxiv_id,
        "title": arxiv_meta.get("title", ""),
        "authors": _parse_authors(arxiv_meta.get("authors")),
        "abstract": (arxiv_meta.get("abstract") or "").strip(),
        "categories": categories,
        "primary_category": categories[0] if categories else None,
        "doi": arxiv_meta.get("doi"),
        "journal_ref": arxiv_meta.get("journal-ref"),
        "license_url": arxiv_meta.get("license"),
        "datestamp": header.get("datestamp"),
    }


def build_metadata_index(metadata_dirs: Path | list[Path]) -> dict[str, dict]:
    """Read all YYMM.jsonl files and return ``{arxiv_id: metadata_dict}``.

    Accepts a single directory or a list of directories. JSONL files from
    all directories are merged into one index.
    """
    if isinstance(metadata_dirs, Path):
        metadata_dirs = [metadata_dirs]

    index: dict[str, dict] = {}

    jsonl_files: list[Path] = []
    for d in metadata_dirs:
        found = sorted(d.glob("*.jsonl"))
        jsonl_files.extend(found)

    if not jsonl_files:
        logger.warning("No .jsonl files found in %s", metadata_dirs)
        return index

    for jsonl_path in jsonl_files:
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "%s:%d: skipping malformed JSON", jsonl_path.name, line_no
                    )
                    continue

                parsed = _parse_oai_record(raw)
                if parsed is None:
                    continue
                index[parsed["arxiv_id"]] = parsed

    logger.info("Metadata index: %d records from %d files",
                len(index), len(jsonl_files))
    return index
