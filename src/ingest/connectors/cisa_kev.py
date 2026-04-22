from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ingest.connectors.base import NormalizedData
from ingest.readers import read


class CisaKevConnector:
    source_id = "cisa-kev"

    def iter_records(self, path: Path) -> Iterator[dict]:
        """Stream vulnerability records from the CISA KEV catalog."""
        yield from read(path, json_path="vulnerabilities.item")

    def normalize(self, record: dict) -> NormalizedData:
        """Convert one KEV entry into the normalized schema."""
        cve_id = record["cveID"]

        return NormalizedData(
            # identification
            record_id=f"cisa-kev:{cve_id}",
            source_id=self.source_id,
            source_record_id=cve_id,

            # main data
            content=record.get("shortDescription", ""),
            title=record.get("vulnerabilityName"),
            raw=record,

            # metadata
            ingested_at=datetime.now(timezone.utc),

            # security-specific
            cwe_ids=_extract_cwe_ids(record),

            # optional metadata
            published_at=_parse_date(record.get("dateAdded")),
            source_url=f"https://www.cisa.gov/known-exploited-vulnerabilities-catalog?search_api_fulltext={cve_id}",
            language="en",
        )


def _extract_cwe_ids(record: dict) -> list[str]:
    """Return CWE IDs from the cwes field, if present."""
    cwes = record.get("cwes")
    if not cwes:
        return []
    return [cwe for cwe in cwes if cwe.startswith("CWE-")]


def _parse_date(value: str | None) -> datetime | None:
    """Parse a YYYY-MM-DD date string into a timezone-aware datetime."""
    if not value:
        return None
    try:
        dt = datetime.strptime(value, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None