from pathlib import Path

from ingest.pipeline import ingest_and_store


def main():
    path = Path("data/mitre-capec/raw/capec_latest.xml")
    output_dir = Path("data/mitre-capec/normalized")

    if not path.exists():
        print(f"File not found: {path}")
        return

    count = ingest_and_store(path, source="capec", output_dir=output_dir)
    print(f"CAPEC: {count} records")


if __name__ == "__main__":
    main()
