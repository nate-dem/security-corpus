#!/usr/bin/env python3
"""Expand the arXiv corpus via single-hop citation links using Semantic Scholar bulk data.

Three phases:
  1. Download Semantic Scholar 'papers' and 'citations' bulk datasets
  2. Build a corpus_id ↔ arXiv ID mapping from the papers dataset
  3. Find all arXiv papers one citation hop from the seed set

Usage:
    python scripts/arxiv/citation_expand.py \\
        --metadata-dir data/arxiv/raw/metadata/cs_CR \\
        --s2-data-dir /Volumes/SECURITY/semantic_scholar \\
        --output-file data/arxiv/raw/metadata/citations/discovered_ids.txt \\
        --api-key $S2_API_KEY
"""

import argparse
import glob
import gzip
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

S2_DATASETS_API = "https://api.semanticscholar.org/datasets/v1/release"


def parse_args():
    p = argparse.ArgumentParser(
        description="Expand arXiv corpus via Semantic Scholar citation graph"
    )
    p.add_argument(
        "--metadata-dir",
        default="data/arxiv/raw/metadata/cs_CR",
        help="Directory containing seed paper metadata JSONL files",
    )
    p.add_argument(
        "--s2-data-dir",
        default="/Volumes/SECURITY/semantic_scholar",
        help="Directory for Semantic Scholar bulk data",
    )
    p.add_argument(
        "--output-file",
        default="data/arxiv/raw/metadata/citations/discovered_ids.txt",
        help="Output file for discovered arXiv IDs",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step if bulk data already exists",
    )
    p.add_argument(
        "--workers",
        type=int, default=None,
        help="Parallel workers for processing shards (default: cpu count)",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="Semantic Scholar API key (or set S2_API_KEY env var)",
    )
    return p.parse_args()


# ── Phase 1: Download S2 bulk datasets ─────────────────────────────────────

def _get_release_id(api_key: str | None) -> str:
    """Get the latest Semantic Scholar release ID."""
    import requests

    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    resp = requests.get(f"{S2_DATASETS_API}/latest", headers=headers, timeout=30)
    resp.raise_for_status()
    release_id = resp.json()["release_id"]
    logger.info("Latest S2 release: %s", release_id)
    return release_id


def _get_download_links(release_id: str, dataset_name: str, api_key: str | None) -> list[str]:
    """Fetch fresh download URLs for a Semantic Scholar dataset."""
    import requests

    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    resp = requests.get(
        f"{S2_DATASETS_API}/{release_id}/dataset/{dataset_name}",
        headers=headers, timeout=30,
    )
    resp.raise_for_status()
    links = resp.json().get("files", [])
    return links


def _download_shard(url: str, out_path: Path) -> bool:
    """Download a single shard using curl. Returns True on success."""
    result = subprocess.run(
        ["curl", "-s", "-S", "-o", str(out_path), "-L",
         "--connect-timeout", "10", "--max-time", "1200", url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error("curl failed for %s: %s", out_path.name, result.stderr)
        if out_path.exists():
            out_path.unlink()
        return False

    # Verify it's actually gzip, not an error page
    if out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path, "rb") as f:
            magic = f.read(2)
        if magic != b'\x1f\x8b':
            logger.error("Shard %s is not gzip — likely expired URL, removing", out_path.name)
            out_path.unlink()
            return False

    return True


def _is_valid_shard(path: Path) -> bool:
    """Check if a file is a valid gzip file."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "rb") as f:
        return f.read(2) == b'\x1f\x8b'


def download_dataset(dataset_name: str, dataset_dir: Path, release_id: str,
                     api_key: str | None):
    """Download all shards for a dataset, re-fetching URLs in batches to avoid expiry."""
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Get all download links
    all_links = _get_download_links(release_id, dataset_name, api_key)
    total = len(all_links)
    logger.info("Dataset '%s': %d shards total", dataset_name, total)

    # Check which shards we already have (valid gzip files)
    existing_names = set()
    for f in dataset_dir.glob("*.gz"):
        if _is_valid_shard(f):
            existing_names.add(f.name)

    # Remove any invalid files (expired URL error pages from previous runs)
    for f in dataset_dir.glob("*.gz"):
        if f.name not in existing_names:
            logger.info("Removing invalid shard: %s", f.name)
            f.unlink()

    # Figure out which shards still need downloading
    needed = []
    for url in all_links:
        filename = url.split("/")[-1].split("?")[0]
        if filename not in existing_names:
            needed.append((filename, url))

    if not needed:
        logger.info("All %d shards already downloaded for '%s'", total, dataset_name)
        return

    logger.info("Need to download %d/%d shards for '%s'", len(needed), total, dataset_name)

    # Download in batches of 20, re-fetching URLs each batch to avoid expiry
    batch_size = 20
    downloaded = 0

    for batch_start in range(0, len(needed), batch_size):
        batch_filenames = [name for name, _ in needed[batch_start:batch_start + batch_size]]

        # Re-fetch fresh URLs
        fresh_links = _get_download_links(release_id, dataset_name, api_key)
        url_by_name = {}
        for url in fresh_links:
            fname = url.split("/")[-1].split("?")[0]
            url_by_name[fname] = url

        for filename in batch_filenames:
            url = url_by_name.get(filename)
            if not url:
                logger.error("No URL found for shard %s", filename)
                continue

            out_path = dataset_dir / filename
            if _download_shard(url, out_path):
                downloaded += 1
                logger.info(
                    "  %s: downloaded %d/%d (%s)",
                    dataset_name, downloaded + len(existing_names),
                    total, filename,
                )
            else:
                logger.error("  Failed to download shard %s", filename)

    logger.info("Finished downloading '%s': %d new + %d existing = %d total",
                dataset_name, downloaded, len(existing_names),
                downloaded + len(existing_names))


def download_datasets(s2_data_dir: Path, api_key: str | None):
    """Download papers and citations datasets."""
    release_id = _get_release_id(api_key)

    for dataset_name in ("papers", "citations"):
        dataset_dir = s2_data_dir / dataset_name
        download_dataset(dataset_name, dataset_dir, release_id, api_key)


# ── Phase 2: Build arXiv ↔ corpus ID mapping ──────────────────────────────

def _process_papers_shard(gz_file: str) -> tuple[str, dict, dict, int]:
    """Extract arXiv ↔ corpus ID mappings from a single papers shard."""
    local_a2s: dict[str, int] = {}
    local_s2a: dict[str, str] = {}
    count = 0

    with gzip.open(gz_file, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                external_ids = rec.get("externalids") or rec.get("externalIds") or {}
                arxiv_id = external_ids.get("ArXiv")
                corpus_id = rec.get("corpusid") or rec.get("corpusId")
                if arxiv_id and corpus_id:
                    local_a2s[arxiv_id] = corpus_id
                    local_s2a[str(corpus_id)] = arxiv_id
                    count += 1
            except (json.JSONDecodeError, KeyError):
                continue

    return gz_file, local_a2s, local_s2a, count


def build_arxiv_mapping(s2_data_dir: Path, workers: int | None) -> tuple[dict, dict]:
    """Build arXiv ↔ corpus ID mapping from papers shards.

    Returns (arxiv_to_ss, ss_to_arxiv).
    """
    papers_dir = s2_data_dir / "papers"
    mapping_path = s2_data_dir / "ss_to_arxiv.json"
    reverse_path = s2_data_dir / "arxiv_to_ss.json"

    # Check for cached mapping (only use if non-empty)
    if mapping_path.exists() and reverse_path.exists():
        if mapping_path.stat().st_size > 10:
            logger.info("Loading cached arXiv mapping from %s", mapping_path)
            with open(reverse_path) as f:
                arxiv_to_ss = json.load(f)
            with open(mapping_path) as f:
                ss_to_arxiv = json.load(f)
            logger.info("Loaded mapping: %d arXiv papers", len(arxiv_to_ss))
            return arxiv_to_ss, ss_to_arxiv

    gz_files = sorted(glob.glob(str(papers_dir / "*.gz")))
    if not gz_files:
        logger.error("No .gz files found in %s", papers_dir)
        sys.exit(1)

    logger.info("Building arXiv mapping from %d shards", len(gz_files))

    arxiv_to_ss: dict[str, int] = {}
    ss_to_arxiv: dict[str, str] = {}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_papers_shard, f): f for f in gz_files
        }
        for future in as_completed(futures):
            try:
                _, local_a2s, local_s2a, count = future.result()
                arxiv_to_ss.update(
                    {k: v for k, v in local_a2s.items() if k not in arxiv_to_ss}
                )
                ss_to_arxiv.update(
                    {k: v for k, v in local_s2a.items() if k not in ss_to_arxiv}
                )
                logger.info("  Processed shard: %d arXiv entries", count)
            except Exception as e:
                logger.error("Error processing shard: %s", e)

    logger.info("Total arXiv papers in mapping: %d", len(arxiv_to_ss))

    # Cache to disk
    with open(reverse_path, "w") as f:
        json.dump(arxiv_to_ss, f)
    with open(mapping_path, "w") as f:
        json.dump(ss_to_arxiv, f)
    logger.info("Saved mapping to %s", s2_data_dir)

    return arxiv_to_ss, ss_to_arxiv


# ── Phase 3: Find citation-linked papers ──────────────────────────────────

def _process_citation_shard(
    gz_file: str, seed_corpus_ids: set[str], ss_to_arxiv: dict[str, str]
) -> tuple[str, set[str], int]:
    """Find arXiv IDs one hop from the seed set in a single citation shard."""
    discovered: set[str] = set()
    edges_examined = 0

    with gzip.open(gz_file, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                citation = json.loads(line)
                citing_id = str(citation.get("citingcorpusid", ""))
                cited_id = str(citation.get("citedcorpusid", ""))
                edges_examined += 1

                if citing_id in seed_corpus_ids:
                    arxiv_id = ss_to_arxiv.get(cited_id)
                    if arxiv_id:
                        discovered.add(arxiv_id)

                if cited_id in seed_corpus_ids:
                    arxiv_id = ss_to_arxiv.get(citing_id)
                    if arxiv_id:
                        discovered.add(arxiv_id)

            except (json.JSONDecodeError, KeyError):
                continue

    return gz_file, discovered, edges_examined


def find_citation_papers(
    seed_arxiv_ids: set[str],
    arxiv_to_ss: dict[str, int],
    ss_to_arxiv: dict[str, str],
    s2_data_dir: Path,
    workers: int | None,
) -> set[str]:
    """Find all arXiv papers one citation hop from the seed set."""
    citations_dir = s2_data_dir / "citations"
    gz_files = sorted(glob.glob(str(citations_dir / "*.gz")))
    if not gz_files:
        logger.error("No citation files found in %s", citations_dir)
        sys.exit(1)

    # Convert seed arXiv IDs to corpus IDs for fast lookup
    seed_corpus_ids: set[str] = set()
    for arxiv_id in seed_arxiv_ids:
        corpus_id = arxiv_to_ss.get(arxiv_id)
        if corpus_id is not None:
            seed_corpus_ids.add(str(corpus_id))

    logger.info(
        "Seed set: %d arXiv IDs → %d corpus IDs (%.1f%% mapped)",
        len(seed_arxiv_ids), len(seed_corpus_ids),
        100 * len(seed_corpus_ids) / max(len(seed_arxiv_ids), 1),
    )
    logger.info("Scanning %d citation shards", len(gz_files))

    all_discovered: set[str] = set()
    total_edges = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_citation_shard, f, seed_corpus_ids, ss_to_arxiv
            ): f
            for f in gz_files
        }
        for future in as_completed(futures):
            try:
                gz_file, discovered, edges = future.result()
                all_discovered.update(discovered)
                total_edges += edges
                logger.info(
                    "  Shard done: %d new IDs found (%d edges examined)",
                    len(discovered), edges,
                )
            except Exception as e:
                logger.error("Error processing citation shard: %s", e)

    new_papers = all_discovered - seed_arxiv_ids

    logger.info("Citation scan complete:")
    logger.info("  Total edges examined: %d", total_edges)
    logger.info("  Total arXiv IDs found (including seed): %d", len(all_discovered))
    logger.info("  New arXiv IDs (excluding seed): %d", len(new_papers))

    return new_papers


# ── Seed loading ──────────────────────────────────────────────────────────

def load_seed_ids(metadata_dir: Path) -> set[str]:
    """Load arXiv IDs from OAI-PMH metadata JSONL files."""
    ids: set[str] = set()
    for jsonl_path in sorted(metadata_dir.glob("*.jsonl")):
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
                    ids.add(arxiv_id)
    return ids


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    metadata_dir = Path(args.metadata_dir)
    s2_data_dir = Path(args.s2_data_dir)
    output_file = Path(args.output_file)
    api_key = args.api_key or os.environ.get("S2_API_KEY")

    if not metadata_dir.is_dir():
        logger.error("Metadata directory not found: %s", metadata_dir)
        sys.exit(1)

    # Load seed papers
    logger.info("Loading seed arXiv IDs from %s", metadata_dir)
    seed_ids = load_seed_ids(metadata_dir)
    logger.info("Loaded %d seed papers", len(seed_ids))

    # Phase 1: Download
    if not args.skip_download:
        logger.info("Phase 1: Downloading Semantic Scholar bulk datasets")
        download_datasets(s2_data_dir, api_key)
    else:
        logger.info("Phase 1: Skipping download (--skip-download)")

    # Phase 2: Build mapping
    logger.info("Phase 2: Building arXiv ↔ corpus ID mapping")
    arxiv_to_ss, ss_to_arxiv = build_arxiv_mapping(s2_data_dir, args.workers)

    # Phase 3: Find citation-linked papers
    logger.info("Phase 3: Scanning citations for seed-linked papers")
    new_ids = find_citation_papers(
        seed_ids, arxiv_to_ss, ss_to_arxiv, s2_data_dir, args.workers
    )

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for arxiv_id in sorted(new_ids):
            f.write(arxiv_id + "\n")

    logger.info("Wrote %d discovered arXiv IDs to %s", len(new_ids), output_file)


if __name__ == "__main__":
    main()
