#!/usr/bin/env python3
"""
Script 01 — Prepare tokenized data buckets.

For each bucket defined in config/buckets.yaml:
  1. Read raw text from the existing corpus (parquet files in data/).
  2. Tokenize with the base model tokenizer.
  3. Write tokenized sequences to data/buckets/<bucket_name>/*.parquet
  4. Write back the token count to buckets.yaml.

This script maps existing corpus sources to REGMIX buckets.
New sources (reddit, github, blogs) are skipped with a warning until
their ingestion connectors are built.

Usage:
    python -m regmix.scripts.01_prepare_buckets \
        --config regmix/config/buckets.yaml \
        --experiment regmix/config/experiment.yaml \
        --output data/buckets
"""

import argparse
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bucket → source mapping
# Paths are relative to the repo root (security-corpus/).
# Each entry maps a bucket name to one or more existing parquet sources.
# ---------------------------------------------------------------------------
BUCKET_SOURCES: dict[str, list[str]] = {
    "mitre_cve": [
        "data/bron/normalized/source_id=bron/raw.parquet",
        # add NVD, CWE, CAPEC, CISA KEV parquets here as they're ingested
    ],
    "sigma_rules": [
        # sigma outputs a parquet when ingest_sigma.py is run
        # placeholder: add after running scripts/ingest_sigma.py
    ],
    "bron_graph": [
        "data/bron/normalized/source_id=bron/raw.parquet",
    ],
    "stackexchange_security": [
        # produced by scripts/ingest_stackexchange.py
    ],
    "youtube_cyber": [
        "data/youtube-transcripts/normalized/source_id=youtube-transcripts/raw.parquet",
    ],
    "reddit_cyber": [],       # not yet ingested
    "github_security": [],    # not yet ingested
    "security_blogs": [],     # not yet ingested
    "general_technical": [],  # bring your own (C4, Wikipedia, etc.)
}

TEXT_COLUMN = "text"         # column name in source parquets


def tokenize_and_save(
    source_files: list[Path],
    out_dir: Path,
    tokenizer,
    max_length: int = 2048,
    shard_size: int = 50_000,
) -> int:
    """
    Tokenize text from source parquets and write chunked output parquets.

    Returns total token count.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    buffer: list[list[int]] = []
    total_tokens = 0
    shard_idx = 0

    def flush():
        nonlocal shard_idx
        if not buffer:
            return
        table = pa.table({"input_ids": buffer})
        pq.write_table(table, out_dir / f"shard_{shard_idx:04d}.parquet")
        shard_idx += 1
        buffer.clear()

    for src in source_files:
        if not src.exists():
            logger.warning(f"Source not found, skipping: {src}")
            continue

        table = pq.read_table(str(src))
        if TEXT_COLUMN not in table.column_names:
            logger.warning(f"{src} has no '{TEXT_COLUMN}' column; available: {table.column_names}")
            continue

        texts = table[TEXT_COLUMN].to_pylist()
        logger.info(f"Tokenizing {len(texts):,} documents from {src.name}")

        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            # sliding window: chunk into max_length segments
            for start in range(0, len(ids), max_length):
                chunk = ids[start : start + max_length]
                if len(chunk) < 64:   # discard very short sequences
                    continue
                buffer.append(chunk)
                total_tokens += len(chunk)
                if len(buffer) >= shard_size:
                    flush()

    flush()
    return total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="regmix/config/buckets.yaml")
    parser.add_argument("--experiment", default="regmix/config/experiment.yaml")
    parser.add_argument("--output", default="data/buckets")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    with open(repo_root / args.experiment) as f:
        exp_cfg = yaml.safe_load(f)

    base_model = exp_cfg["training"]["base_model"]
    max_length = exp_cfg["training"]["max_length"]

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except ImportError:
        raise ImportError("transformers required: pip install transformers")

    from regmix.buckets.registry import BucketRegistry
    registry = BucketRegistry(repo_root / args.config)

    out_root = repo_root / args.output

    for bucket_name in registry.names():
        sources = [repo_root / s for s in BUCKET_SOURCES.get(bucket_name, [])]
        if not sources:
            logger.warning(f"[{bucket_name}] No sources defined — skipping")
            continue

        out_dir = out_root / bucket_name
        logger.info(f"[{bucket_name}] → {out_dir}")

        if args.dry_run:
            logger.info(f"[{bucket_name}] dry-run, skipping tokenization")
            continue

        token_count = tokenize_and_save(sources, out_dir, tokenizer, max_length)
        logger.info(f"[{bucket_name}] {token_count:,} tokens written")
        registry.update_token_count(bucket_name, token_count)

    logger.info("\nBucket summary:")
    logger.info(registry.stats())


if __name__ == "__main__":
    main()
