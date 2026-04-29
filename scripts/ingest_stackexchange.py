import argparse
from pathlib import Path

from ingest.pipeline import ingest_and_store


SITES = {
    "infosec": "stackexchange-infosec",
    "reverseengineering": "stackexchange-reverseengineering",
    "crypto": "stackexchange-crypto",
    "tor": "stackexchange-tor",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("site", choices=SITES.keys())
    args = parser.parse_args()

    source_id = SITES[args.site]
    raw_dir = Path(f"data/{source_id}/raw/extracted")
    output_dir = Path(f"data/{source_id}/normalized")

    if not raw_dir.is_dir():
        print(f"Extracted directory not found: {raw_dir}")
        return

    count = ingest_and_store(raw_dir, source=source_id, output_dir=output_dir)
    print(f"{source_id}: {count} records")


if __name__ == "__main__":
    main()