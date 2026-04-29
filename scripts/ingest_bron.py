from pathlib import Path

from ingest.pipeline import ingest_and_store


def main() -> None:
    raw_dir = Path("data/bron/raw")
    output_dir = Path("data/bron/normalized")

    if not raw_dir.is_dir():
        print(f"Raw directory not found: {raw_dir}")
        print("Run scripts/export_bron.py first.")
        return

    count = ingest_and_store(raw_dir, source="bron", output_dir=output_dir)
    print(f"bron: {count} records")


if __name__ == "__main__":
    main()
