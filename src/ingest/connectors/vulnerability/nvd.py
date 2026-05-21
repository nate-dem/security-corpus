from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ingest.connectors.base import VulnerabilityData
from ingest.readers import read
from ingest.utils import compute_content_hash, compute_token_count, PUBLIC_DOMAIN


class NVDConnector:
    source_id = "nvd"

    def iter_records(self, path: Path) -> Iterator[dict]:
        """Stream raw CVE records from an NVD JSON feed file."""
        yield from read(path, json_path="vulnerabilities.item")

    def normalize(self, record: dict) -> VulnerabilityData:
        """Convert one NVD CVE record into the canonical schema."""
        cve = record["cve"]
        cve_id = cve["id"]
        content = _extract_english_description(cve.get("descriptions", []))

        return VulnerabilityData(
            # identification
            record_id=f"nvd:{cve_id}",
            source_id=self.source_id,
            source_record_id=cve_id,

            # main data
            content=content,
            content_length=compute_token_count(content),
            content_hash=compute_content_hash(content),
            raw=record,

            # metadata
            ingested_at=datetime.now(timezone.utc),
            license=PUBLIC_DOMAIN,

            # vulnerability-specific
            cve_id=cve_id,
            severity=_extract_severity(cve.get("metrics", {})),
            cvss_score=_extract_cvss_score(cve.get("metrics", {})),
            cwe_ids=_extract_cwe_ids(cve.get("weaknesses", [])),

            # optional metadata
            published_at=_parse_datetime(cve.get("published")),
            source_url=f"https://nvd.nist.gov/vuln/detail/{cve_id}",
        )

# Precedence order for picking a CVSS score when multiple versions are present.
_CVSS_VERSION_PRIORITY = ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2")

def _extract_english_description(descriptions: list[dict]) -> str:
    """Return the first English description, or empty string if none."""
    for desc in descriptions:
        if desc.get("lang") == "en":
            return desc.get("value", "")
    return ""

def _extract_cvss_score(metrics: dict) -> float | None:
    """Return the CVSS base score from the highest-priority version available."""
    for version_key in _CVSS_VERSION_PRIORITY:
        entries = metrics.get(version_key) or []
        if entries:
            return entries[0].get("cvssData", {}).get("baseScore")
    return None

def _extract_severity(metrics: dict) -> str | None:
    """Return the severity label (LOW/MEDIUM/HIGH/CRITICAL) in lowercase."""
    for version_key in _CVSS_VERSION_PRIORITY:
        entries = metrics.get(version_key) or []
        if entries:
            # v3+ puts severity in cvssData.baseSeverity
            # v2 uses a different shape
            severity = (
                entries[0].get("cvssData", {}).get("baseSeverity")
                or entries[0].get("baseSeverity")
            )
            if severity:
                return severity.lower()
    return None

def _extract_cwe_ids(weaknesses: list[dict]) -> list[str]:
    """Pull all CWE identifiers from the weaknesses list (deduplicated)."""
    cwe_ids = []
    for weakness in weaknesses:
        for desc in weakness.get("description", []):
            value = desc.get("value", "")
            if value.startswith("CWE-") and value not in cwe_ids:
                cwe_ids.append(value)
    return cwe_ids


def _parse_datetime(value: str | None) -> datetime | None:
    """Parse NVD's timestamps; return None if missing or not able to parse."""
    if not value:
        return None
    try:
        # NVD uses the format "2026-04-20T10:00:00.000"
        return datetime.fromisoformat(value)
    except ValueError:
        return None
