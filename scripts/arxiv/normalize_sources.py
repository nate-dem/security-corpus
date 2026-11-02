#!/usr/bin/env python3
"""Extract and normalize downloaded arXiv paper sources.

For each downloaded source archive:
  1. Extract to a temporary directory
  2. Find the main .tex file, inline includes, strip comments
  3. Write cleaned main.tex + status.json to the normalized directory

For PDF-only downloads:
  1. Extract text from each page
  2. Write cleaned main.txt + status.json to the normalized directory

Uses the LaTeX processing functions from
``src/ingest/connectors/arxiv/latex_processing.py``.

Usage:
    python scripts/arxiv/normalize_sources.py \\
        --downloads-dir data/arxiv/raw/source/downloads \\
        --output-dir data/arxiv/raw/source/normalized \\
        --workers 8
"""

import argparse
import gzip
import hashlib
import json
import logging
import re
import sys
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ingest.connectors.arxiv.latex_processing import LATEX_NORMALIZER_VERSION  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".tar.gz", ".tar", ".gz", ".pdf")
PDF_NORMALIZER_VERSION = "pdf-text-v1"
MAX_ARCHIVE_MEMBERS = 20_000
MAX_ARCHIVE_UNPACKED_BYTES = 2 * 1024 * 1024 * 1024
PDF_WHITESPACE_RE = re.compile(r"[ \t]+")
PDF_BLANK_LINES_RE = re.compile(r"(\n\s*){3,}")


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
    """Extract one source archive with traversal and expansion protections."""
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try as tar.gz / tar first
        if tarfile.is_tarfile(str(archive_path)):
            with tarfile.open(archive_path) as tf:
                members = []
                unpacked_bytes = 0
                for member in tf:
                    members.append(member)
                    unpacked_bytes += member.size if member.isfile() else 0
                    if len(members) > MAX_ARCHIVE_MEMBERS:
                        raise ValueError(f"Archive exceeds {MAX_ARCHIVE_MEMBERS} members")
                    if unpacked_bytes > MAX_ARCHIVE_UNPACKED_BYTES:
                        raise ValueError(f"Archive exceeds {MAX_ARCHIVE_UNPACKED_BYTES} unpacked bytes")
                root = extract_dir.resolve()
                for member in members:
                    if member.issym() or member.islnk() or member.isdev():
                        raise ValueError(
                            f"Archive contains unsupported link/device: {member.name}"
                        )
                    destination = (root / member.name).resolve()
                    try:
                        destination.relative_to(root)
                    except ValueError as error:
                        raise ValueError(
                            f"Archive member escapes extraction root: {member.name}"
                        ) from error
                for member in members:
                    tf.extract(member, path=root, set_attrs=False)
            return True
    except (tarfile.TarError, EOFError, OSError, ValueError) as error:
        logger.warning("Tar extraction failed for %s: %s", archive_path, error)
        # A rejected tar must never be reinterpreted as a gzip-wrapped TeX file.
        return False

    try:
        # Try as plain gzip (single file)
        with gzip.open(archive_path, "rb") as gz:
            content = gz.read(MAX_ARCHIVE_UNPACKED_BYTES + 1)
        if len(content) > MAX_ARCHIVE_UNPACKED_BYTES:
            raise ValueError(
                f"Gzip expands beyond {MAX_ARCHIVE_UNPACKED_BYTES} bytes"
            )
        # Write as a .tex file
        arxiv_id = archive_path.stem.replace(".tar", "")
        out_file = extract_dir / f"{arxiv_id}.tex"
        out_file.write_bytes(content)
        return True
    except (gzip.BadGzipFile, EOFError, OSError, ValueError) as error:
        logger.warning("Gzip extraction failed for %s: %s", archive_path, error)
        pass

    return False


def _strip_supported_suffix(path: Path) -> str:
    """Return the arXiv ID represented by a downloaded source filename."""
    name = path.name
    for suffix in SUPPORTED_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _clean_pdf_text(text: str) -> str:
    """Normalize common whitespace noise from PDF page text extraction."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = PDF_WHITESPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = PDF_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def _extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF-only arXiv download."""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError(
            "PDF normalization requires the pypdf package. "
            "Run `pip install -e .` to install project dependencies."
        ) from e

    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        if page_text:
            pages.append(page_text)
    return _clean_pdf_text("\n\n".join(pages))


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
    arxiv_id = _strip_supported_suffix(archive_path)

    tgt = output_dir / yymm / arxiv_id
    tgt.mkdir(parents=True, exist_ok=True)

    expected_version = (
        PDF_NORMALIZER_VERSION
        if archive_path.name.endswith(".pdf")
        else LATEX_NORMALIZER_VERSION
    )
    with archive_path.open("rb") as source_handle:
        source_sha256 = hashlib.file_digest(source_handle, "sha256").hexdigest()

    # Skip only output produced by the current normalizer. Completed legacy
    # status files are deliberately reprocessed during recovery.
    status_file = tgt / "status.json"
    if status_file.exists():
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                status = json.load(f)
            if (
                status.get("completed", False)
                and status.get("normalizer_version") == expected_version
                and status.get("source_sha256") == source_sha256
                and (status.get("auto_ignore") or (
                    tgt / ("main.txt" if expected_version == PDF_NORMALIZER_VERSION else "main.tex")
                ).is_file())
            ):
                return (arxiv_id, 0, 1, 0)
        except Exception:
            pass

    if archive_path.name.endswith(".pdf"):
        status = {
            "aid": arxiv_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "completed": False,
            "normalizer_version": PDF_NORMALIZER_VERSION,
            "source_sha256": source_sha256,
            "source_filename": archive_path.name,
            "source_format": "pdf",
            "pdf_extracted": False,
            "errors": [],
        }
        try:
            text = _extract_pdf_text(archive_path)
            if not text:
                status["errors"].append("PDF text extraction produced empty content")
                write_status_json(tgt, status)
                return (arxiv_id, 0, 0, 1)
            temporary_text = tgt / "main.txt.tmp"
            temporary_text.write_text(text, encoding="utf-8")
            temporary_text.replace(tgt / "main.txt")
            status["pdf_extracted"] = True
            status["completed"] = True
            write_status_json(tgt, status)
            return (arxiv_id, 1, 0, 0)
        except Exception as e:
            status["errors"].append(str(e))
            write_status_json(tgt, status)
            return (arxiv_id, 0, 0, 1)

    status = {
        "aid": arxiv_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "completed": False,
        "normalizer_version": LATEX_NORMALIZER_VERSION,
        "source_sha256": source_sha256,
        "source_filename": archive_path.name,
        "source_format": "latex",
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
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "completed": True,
                "auto_ignore": True,
                "normalizer_version": LATEX_NORMALIZER_VERSION,
                "source_sha256": source_sha256,
                "source_filename": archive_path.name,
            })
            return (arxiv_id, 0, 1, 0)

        try:
            diagnostics = merge_project(extract_dir, tgt / "main.tex")
            status["include_diagnostics"] = diagnostics.to_dict()
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
        return 2

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
            if (
                archive.is_file()
                and not archive.name.startswith("._")
                and not archive.name.endswith(".failed")
                and archive.name.endswith(SUPPORTED_SUFFIXES)
                and archive.stat().st_size > 0
            ):
                tasks.append((archive, output_dir))

    logger.info("Found %d archives to normalize", len(tasks))

    if not tasks:
        logger.info("Nothing to normalize.")
        return 2

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
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
