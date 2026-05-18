"""Download YouTube-Commons Parquet shards from HuggingFace.

Usage:
    python scripts/export/youtube_transcripts.py            # first 10 shards (~3.8 GB)
    python scripts/export/youtube_transcripts.py --shards 50
    python scripts/export/youtube_transcripts.py --all      # all 439 shards (~163 GB)

Shards already present on disk are skipped (safe to re-run).
"""
import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print(
        "huggingface_hub is required. Install it with:\n"
        "  pip install huggingface_hub",
        file=sys.stderr,
    )
    sys.exit(1)

REPO_ID = "PleIAs/YouTube-Commons"
TOTAL_SHARDS = 439
RAW_DIR = Path("data/youtube-transcripts/raw")


def shard_name(n: int) -> str:
    return f"cctube_{n}.parquet"


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--shards",
        type=int,
        default=10,
        metavar="N",
        help="Download the first N shards (default: 10)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Download all 439 shards (~163 GB)",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    n_shards = TOTAL_SHARDS if args.all else args.shards

    print(f"Downloading {n_shards} shard(s) from {REPO_ID} → {RAW_DIR}")

    downloaded = skipped = 0
    for i in range(n_shards):
        filename = shard_name(i)
        dest = RAW_DIR / filename
        if dest.exists():
            skipped += 1
            continue
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=str(RAW_DIR),
            )
            downloaded += 1
            print(f"  [{downloaded + skipped}/{n_shards}] {filename}")
        except Exception as exc:
            print(f"  WARNING: could not download {filename}: {exc}", file=sys.stderr)

    print(f"Done. {downloaded} downloaded, {skipped} already present.")


if __name__ == "__main__":
    main()
