from pathlib import Path

from ingest.pipeline import ingest_and_store


def main():
    raw_file = Path("data/cisa-kev/raw/known_exploited_vulnerabilities.json")
    output_dir = Path("data/cisa-kev/normalized")

    count = ingest_and_store(raw_file, source="cisa-kev", output_dir=output_dir)
    print(f"CISA KEV: {count} records")


if __name__ == "__main__":
    main()
