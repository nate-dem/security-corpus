#!/usr/bin/env python3
"""
Script 01 — Prepare tokenized data buckets.

For each bucket defined in config/buckets.yaml:
  1. Read raw text from the existing corpus (parquet files in data/).
  2. Tokenize with the base model tokenizer.
  3. Write tokenized sequences to data/buckets/<bucket_name>/*.parquet
  4. Write back the token count to buckets.yaml.

This script maps existing corpus sources to REGMIX buckets.
Sources without normalized parquet paths yet are skipped with a warning.

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
TRAINING_CLEAN_V2 = "data/training-clean-v2/normalized"

BUCKET_SOURCES: dict[str, list[str]] = {
    "mitre_cve": [
        f"{TRAINING_CLEAN_V2}/source_id=nvd/*.parquet",
        f"{TRAINING_CLEAN_V2}/source_id=cisa-kev/*.parquet",
        f"{TRAINING_CLEAN_V2}/source_id=mitre-attack/*.parquet",
        f"{TRAINING_CLEAN_V2}/source_id=mitre-cwe/*.parquet",
        f"{TRAINING_CLEAN_V2}/source_id=capec/*.parquet",
    ],
    "sigma_rules": [
        f"{TRAINING_CLEAN_V2}/source_id=sigma/*.parquet",
    ],
    "bron_graph": [
        # RESEARCHER: add BRON normalized parquet path when that bucket is rebuilt.
    ],
    "stackexchange_security": [
        f"{TRAINING_CLEAN_V2}/source_id=stackexchange-infosec/*.parquet",
        f"{TRAINING_CLEAN_V2}/source_id=stackexchange-reverseengineering/*.parquet",
        f"{TRAINING_CLEAN_V2}/source_id=stackexchange-crypto/*.parquet",
        f"{TRAINING_CLEAN_V2}/source_id=stackexchange-tor/*.parquet",
        f"{TRAINING_CLEAN_V2}/source_id=stackoverflow/*.parquet",
    ],
    "youtube_cyber": [
        f"{TRAINING_CLEAN_V2}/source_id=youtube-transcripts/*.parquet",
    ],
    "reddit_cyber": [
        f"{TRAINING_CLEAN_V2}/source_id=reddit-*/*.parquet",
    ],
    "github_security": [],    # RESEARCHER: add source paths when available
    "security_blogs": [],     # RESEARCHER: add source paths when available
    "general_technical": [],  # RESEARCHER: bring your own control corpus
}

TEXT_COLUMN = "content"      # normalized corpus training text column


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

        parquet = pq.ParquetFile(src)
        if TEXT_COLUMN not in parquet.schema_arrow.names:
            logger.warning(
                f"{src} has no '{TEXT_COLUMN}' column; available: {parquet.schema_arrow.names}"
            )
            continue

        logger.info(f"Tokenizing documents from {src}")
        for batch in parquet.iter_batches(columns=[TEXT_COLUMN], batch_size=1_024):
            texts = batch.column(0).to_pylist()
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


def expand_source_patterns(repo_root: Path, patterns: list[str]) -> list[Path]:
    """Expand repo-relative glob patterns into parquet files."""
    files: list[Path] = []
    for pattern in patterns:
        matches = sorted(repo_root.glob(pattern))
        if matches:
            files.extend(path for path in matches if path.is_file())
        else:
            files.append(repo_root / pattern)
    return files


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
        sources = expand_source_patterns(repo_root, BUCKET_SOURCES.get(bucket_name, []))
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
