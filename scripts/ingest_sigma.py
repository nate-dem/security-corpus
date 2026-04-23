from pathlib import Path

from ingest.pipeline import ingest_and_store


def main():
    raw_dir = Path("data/sigma/raw")
    output_dir = Path("data/sigma/normalized")

    if not raw_dir.is_dir():
        print(f"Raw directory not found: {raw_dir}")
        return

    count = ingest_and_store(raw_dir, source="sigma", output_dir=output_dir)
    print(f"sigma: {count} records")


if __name__ == "__main__":
    main()
