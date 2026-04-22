from pathlib import Path
from ingest.pipeline import ingest_and_store

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    raw_dir = Path("data/mitre-attack/raw")
    output_dir = Path("data/mitre-attack/normalized")
    
    # one bundle per domain
    domains = ["enterprise-attack", "mobile-attack", "ics-attack", "pre-attack"]
    
    total = 0
    for domain in domains:
        bundle = raw_dir / domain / f"{domain}.json"
        if not bundle.exists():
            print(f"Skipping {domain}: {bundle} not found")
            continue
        count = ingest_and_store(bundle, source="mitre-attack", output_dir=output_dir)
        print(f"{domain}: {count} records")
        total += count
    
    print(f"\nTotal: {total} records across {len(domains)} domains")


if __name__ == "__main__":
    main()