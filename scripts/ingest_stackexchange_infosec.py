from pathlib import Path

from ingest.pipeline import ingest_and_store


def main():
    raw_dir = Path("data/stackexchange-infosec/raw/extracted")
    output_dir = Path("data/stackexchange-infosec/normalized")

    if not raw_dir.is_dir():
        print(f"Extracted directory not found: {raw_dir}")
        return

    count = ingest_and_store(raw_dir, source="stackexchange-infosec", output_dir=output_dir)
    print(f"stackexchange-infosec: {count} records")


if __name__ == "__main__":
    main()
