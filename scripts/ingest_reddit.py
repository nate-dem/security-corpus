"""Ingest Reddit subreddit data from Arctic Shift JSONL dumps.

Usage:
    python scripts/ingest_reddit.py netsec
    python scripts/ingest_reddit.py --all
"""

import argparse
from pathlib import Path

from ingest.pipeline import ingest, REDDIT_SUBREDDITS
from ingest.writers import write_parquet


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Reddit subreddit data from Arctic Shift dumps.",
    )
    parser.add_argument(
        "subreddit",
        nargs="?",
        help="Subreddit name to ingest (case-sensitive, matches filename).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all security subreddits.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/reddit/raw"),
        help="Directory containing .zst files (default: data/reddit/raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reddit/normalized"),
        help="Output directory for Parquet files (default: data/reddit/normalized).",
    )
    args = parser.parse_args()

    if args.all:
        subreddits = REDDIT_SUBREDDITS
    elif args.subreddit:
        subreddits = [args.subreddit]
    else:
        parser.error("Provide a subreddit name or --all")

    for sub in subreddits:
        source_id = f"reddit-{sub.lower()}"
        submissions_file = args.data_dir / f"{sub}_submissions.zst"
        if not submissions_file.exists():
            print(f"Skipping {sub}: {submissions_file} not found")
            continue

        records = ingest(args.data_dir, source=source_id)
        # Use subreddit name as the Parquet filename (not the directory name)
        count = write_parquet(records, args.output_dir, source=source_id, input_path=Path(sub.lower()))
        print(f"{source_id}: {count} records")


if __name__ == "__main__":
    main()
