"""Ingest YouTube-Commons transcripts into normalized Parquet.

Usage
-----
# Without classifier cache — passes all English records with >= 50 words:
python3 scripts/ingest_youtube_transcripts.py

# With classifier cache — additionally filters by vLLM classifier decisions:
python3 scripts/ingest_youtube_transcripts.py \
    --cache-path data/youtube-transcripts/classifier_cache.jsonl

# Ingest everything unfiltered (for inspection or building the channel CSV):
python3 scripts/ingest_youtube_transcripts.py --no-filter

Prerequisites
-------------
  Raw shards must be downloaded first:
    python3 scripts/export_youtube_transcripts.py

  Classifier cache (optional) must be built first:
    sbatch scripts/run_youtube_classifier.slurm   # on Marlowe
    python3 scripts/merge_classifier_caches.py    # after array job completes
"""

import argparse
from pathlib import Path

from ingest.connectors.youtube_transcripts import YouTubeTranscriptsConnector
from ingest.writers import write_parquet

RAW_DIR = Path("data/youtube-transcripts/raw")
OUTPUT_DIR = Path("data/youtube-transcripts/normalized")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to the merged classifier cache JSONL file.  When provided, "
            "only records classified as should_keep=True pass gate 3.  "
            "If omitted, gate 3 is skipped and all English records with "
            ">= 50 words are kept."
        ),
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable all security filtering and ingest every record (useful for inspection).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Directory containing cctube_*.parquet shards (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for normalized Parquet (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    if not args.raw_dir.is_dir():
        print(f"Raw directory not found: {args.raw_dir}")
        print("Run scripts/export_youtube_transcripts.py first.")
        return

    connector = YouTubeTranscriptsConnector()

    # When --cache-path is given, iter_records passes the path down to
    # load_keep_set() which builds the frozenset used by gate 3.
    # When --no-filter is given, filter_security=False bypasses all gates.
    raw_records = connector.iter_records(
        args.raw_dir,
        filter_security=not args.no_filter,
        cache_path=args.cache_path,
    )
    normalized = (connector.normalize(r) for r in raw_records)

    # write_parquet expects an input_path for naming the output file.
    # We pass the raw_dir itself; the writer uses its stem as the filename.
    count = write_parquet(normalized, args.output_dir, source="youtube-transcripts", input_path=args.raw_dir)

    mode = "unfiltered" if args.no_filter else ("classifier+language+wordcount" if args.cache_path else "language+wordcount only")
    print(f"youtube-transcripts: {count:,} records written  [filter mode: {mode}]")
    if args.cache_path:
        print(f"  Cache: {args.cache_path}")


if __name__ == "__main__":
    main()
