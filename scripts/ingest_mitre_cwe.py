from pathlib import Path

from ingest.pipeline import ingest_and_store


def main():
    path = Path("data/mitre-cwe/raw/cwec_v4.19.1.xml")
    output_dir = Path("data/mitre-cwe/normalized")

    if not path.exists():
        print(f"File not found: {path}")
        return

    count = ingest_and_store(path, source="mitre-cwe", output_dir=output_dir)
    print(f"MITRE CWE: {count} records")


if __name__ == "__main__":
    main()
