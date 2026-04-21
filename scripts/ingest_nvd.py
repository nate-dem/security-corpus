from pathlib import Path

from ingest.pipeline import ingest_and_store


def main():
    raw_dir = Path("data/nvd/raw")
    output_dir = Path("data/nvd/normalized")
    
    # process per-year files only
    files = sorted(raw_dir.glob("nvdcve-2.0-[0-9]*.json.gz"))
    
    total = 0
    for f in files:
        count = ingest_and_store(f, source="nvd", output_dir=output_dir)
        print(f"{f.name}: {count} records")
        total += count
    
    print(f"\nTotal: {total} records across {len(files)} files")


if __name__ == "__main__":
    main()