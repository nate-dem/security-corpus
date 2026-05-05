import logging
import shutil
from pathlib import Path

from ingest.connectors.stackexchange.stackoverflow import StackOverflowConnector


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    archive_path = Path("/Volumes/SECURITY/stackoverflow/raw/stackoverflow.com.7z")
    output_dir = Path("data/stackoverflow/normalized")
    intermediate_dir = Path("/Volumes/SECURITY/stackoverflow/intermediate")

    if shutil.which("7z") is None:
        print("Error: 7z not found. Install with: brew install p7zip")
        return

    if not archive_path.exists():
        print(f"Archive not found: {archive_path}")
        print("Make sure the flash drive is mounted at /Volumes/SECURITY/")
        return

    connector = StackOverflowConnector()
    count = connector.ingest(
        archive_path=archive_path,
        output_dir=output_dir,
        intermediate_dir=intermediate_dir,
    )
    print(f"stackoverflow: {count} records")


if __name__ == "__main__":
    main()
