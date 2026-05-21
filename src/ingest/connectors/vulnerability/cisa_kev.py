from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ingest.connectors.base import VulnerabilityData
from ingest.readers import read
from ingest.utils import compute_content_hash, compute_token_count, CISA_TERMS


class CisaKevConnector:
    source_id = "cisa-kev"

    def iter_records(self, path: Path) -> Iterator[dict]:
        """Stream vulnerability records from the CISA KEV catalog."""
        yield from read(path, json_path="vulnerabilities.item")

    def normalize(self, record: dict) -> VulnerabilityData:
        """Convert one KEV entry into the normalized schema."""
        cve_id = record["cveID"]
        content = record.get("shortDescription", "")

        return VulnerabilityData(
            # identification
            record_id=f"cisa-kev:{cve_id}",
            source_id=self.source_id,
            source_record_id=cve_id,

            # main data
            content=content,
            content_length=compute_token_count(content),
            content_hash=compute_content_hash(content),
            title=record.get("vulnerabilityName"),
            raw=record,

            # metadata
            ingested_at=datetime.now(timezone.utc),
            license=CISA_TERMS,

            # vulnerability-specific
            cve_id=cve_id,
            cwe_ids=_extract_cwe_ids(record),
            exploited_in_wild=True,  # every KEV entry is exploited in the wild by definition

            # optional metadata
            published_at=_parse_date(record.get("dateAdded")),
            source_url=f"https://www.cisa.gov/known-exploited-vulnerabilities-catalog?search_api_fulltext={cve_id}",
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
