"""arXiv academic paper connector.

In-memory two-pass approach:
  Pass 1: Read OAI-PMH metadata JSONL files, build {arxiv_id: metadata} index
  Pass 2: Walk normalized LaTeX directories, join with metadata, yield records

Scope filters:
  - Skip papers with incomplete normalization (status.json completed=false)
  - Skip papers without metadata (orphan normalizations)
  - Skip papers without normalized content (metadata-only)

Quality features computed in normalize():
  - content_length (tokens), content_hash (SHA-256)
  - Per-paper license from OAI-PMH metadata
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ingest.connectors.base import AcademicPaperData
from ingest.connectors.arxiv.metadata import (
    build_metadata_index,
    _map_license,
    _parse_datestamp,
)
from ingest.utils import compute_content_hash, compute_token_count

logger = logging.getLogger(__name__)


def _paper_dir_to_arxiv_id(paper_dir_name: str) -> str:
    """Map normalized directory names back to canonical arXiv IDs.

    Download filenames replace ``/`` with ``-`` for old-style arXiv IDs, so
    ``quant-ph/0408108`` is stored under ``quant-ph-0408108``. New-style IDs
    such as ``2401.12345`` are already filesystem-safe and unchanged.
    """
    if "." in paper_dir_name or "-" not in paper_dir_name:
        return paper_dir_name
    category, old_id = paper_dir_name.rsplit("-", 1)
    return f"{category}/{old_id}"


class ArxivConnector:
    """Connector for arXiv papers preprocessed into cleaned LaTeX.

    Expects the following directory layout under ``path``:

    {path}/
        metadata/cs_CR/       # OAI-PMH metadata JSONL (one file per month)
        0701.jsonl
        ...
        source/normalized/    # cleaned source text (one dir per paper)
        YYMM/{arxiv_id}/
            main.tex | main.txt
            status.json
    """

    source_id = "arxiv"

    def iter_records(self, path: Path) -> Iterator[dict]:
        """Yield one assembled record per successfully normalized paper.

        ``path`` is the root of the arXiv raw data directory
        (e.g. ``data/arxiv/raw``).
        """
        metadata_dirs = [path / "metadata" / "cs_CR"]
        citations_dir = path / "metadata" / "citations"
        if citations_dir.is_dir():
            metadata_dirs.append(citations_dir)
        normalized_dir = path / "source" / "normalized"

        # Pass 1: build metadata index (from all metadata directories)
        metadata_index = build_metadata_index(metadata_dirs)
        if not metadata_index:
            logger.warning("Empty metadata index — no records will be emitted")
            return

        # Pass 2: walk normalized papers
        if not normalized_dir.is_dir():
            logger.warning("Normalized directory does not exist: %s",
                           normalized_dir)
            return

        emitted = 0
        skipped_incomplete = 0
        skipped_no_metadata = 0
        skipped_empty = 0

        for yymm_dir in sorted(normalized_dir.iterdir()):
            if not yymm_dir.is_dir():
                continue
            for paper_dir in sorted(yymm_dir.iterdir()):
                if not paper_dir.is_dir():
                    continue

                arxiv_id = _paper_dir_to_arxiv_id(paper_dir.name)
                status: dict = {}

                # Check normalization status
                status_file = paper_dir / "status.json"
                if status_file.exists():
                    try:
                        with open(status_file, "r", encoding="utf-8") as f:
                            status = json.load(f)
                        if not status.get("completed", False):
                            skipped_incomplete += 1
                            continue
                        if status.get("auto_ignore", False):
                            skipped_incomplete += 1
                            continue
                    except (json.JSONDecodeError, OSError):
                        skipped_incomplete += 1
                        continue

                # Read cleaned source text. LaTeX normalizations write main.tex;
                # PDF-only normalizations write extracted text to main.txt.
                main_tex = paper_dir / "main.tex"
                main_txt = paper_dir / "main.txt"
                if main_tex.exists():
                    content_path = main_tex
                    inferred_source_format = "latex"
                elif main_txt.exists():
                    content_path = main_txt
                    inferred_source_format = "pdf"
                else:
                    skipped_incomplete += 1
                    continue
                source_format = status.get("source_format")
                if source_format not in {"latex", "pdf"}:
                    source_format = inferred_source_format

                content = content_path.read_text(encoding="utf-8", errors="ignore")
                if not content.strip():
                    skipped_empty += 1
                    continue

                # Join with metadata
                meta = metadata_index.get(arxiv_id)
                if meta is None:
                    skipped_no_metadata += 1
                    continue

                yield {
                    "arxiv_id": arxiv_id,
                    "source_format": source_format,
                    "content": content,
                    "title": meta.get("title", ""),
                    "authors": meta.get("authors", []),
                    "abstract": meta.get("abstract", ""),
                    "categories": meta.get("categories", []),
                    "primary_category": meta.get("primary_category"),
                    "doi": meta.get("doi"),
                    "journal_ref": meta.get("journal_ref"),
                    "license_url": meta.get("license_url"),
                    "datestamp": meta.get("datestamp"),
                }
                emitted += 1

        logger.info(
            "iter_records: emitted=%d, skipped_incomplete=%d, "
            "skipped_no_metadata=%d, skipped_empty=%d",
            emitted, skipped_incomplete, skipped_no_metadata, skipped_empty,
        )

    def normalize(self, record: dict) -> AcademicPaperData:
        """Convert an assembled record dict to an ``AcademicPaperData`` instance."""
        arxiv_id = record["arxiv_id"]
        content = record["content"]

        datestamp = record.get("datestamp")
        published_at = _parse_datestamp(datestamp) if datestamp else None

        return AcademicPaperData(
            source_id=self.source_id,
            source_record_id=arxiv_id,
            record_id=f"{self.source_id}:{arxiv_id}",
            content=content,
            title=record.get("title"),
            content_length=compute_token_count(content),
            content_hash=compute_content_hash(content),
            ingested_at=datetime.now(timezone.utc),
            published_at=published_at,
            source_url=f"https://arxiv.org/abs/{arxiv_id}",
            license=_map_license(record.get("license_url")),
            raw=None,
            arxiv_id=arxiv_id,
            source_format=record.get("source_format"),
            authors=record.get("authors", []),
            abstract=record.get("abstract"),
            categories=record.get("categories", []),
            primary_category=record.get("primary_category"),
            doi=record.get("doi"),
            journal_ref=record.get("journal_ref"),
        )
