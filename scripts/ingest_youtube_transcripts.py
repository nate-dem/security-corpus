from pathlib import Path

from ingest.pipeline import ingest_and_store


def main() -> None:
    raw_dir = Path("data/youtube-transcripts/raw")
    output_dir = Path("data/youtube-transcripts/normalized")

    if not raw_dir.is_dir():
        print(f"Raw directory not found: {raw_dir}")
        print("Run scripts/export_youtube_transcripts.py first.")
        return

    count = ingest_and_store(raw_dir, source="youtube-transcripts", output_dir=output_dir)
    print(f"youtube-transcripts: {count} records")


if __name__ == "__main__":
    main()
