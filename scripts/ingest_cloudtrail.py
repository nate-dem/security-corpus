"""Ingest flaws.cloud CloudTrail logs into session-grouped records."""

from pathlib import Path

from ingest.pipeline import ingest
from ingest.writers import write_parquet


def main():
    data_dir = Path("data/cloudtrail/raw/flaws_cloudtrail_logs")
    output_dir = Path("data/cloudtrail/normalized")
    source_id = "cloudtrail-flaws"

    records = ingest(data_dir, source=source_id)
    count = write_parquet(records, output_dir, source=source_id, input_path=Path("flaws"))
    print(f"{source_id}: {count} records")


if __name__ == "__main__":
    main()
