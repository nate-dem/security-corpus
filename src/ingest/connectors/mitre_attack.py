from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ingest.connectors.base import MitreData
from ingest.readers import read
from ingest.utils import compute_content_hash, compute_token_count, MITRE_TERMS


# meaningful STIX object types corresponding to ATT&CK techniques
_MEANINGFUL_TYPES = {
    "attack-pattern",
    "intrusion-set",
    "malware",
    "tool",
    "course-of-action",
    "x-mitre-tactic",
    "x-mitre-data-source",
    "x-mitre-data-component",
}


class MitreAttackConnector:
    source_id = "mitre-attack"

    def iter_records(self, path: Path) -> Iterator[dict]:
        """Stream STIX objects from an ATT&CK bundle, filtering to meaningful types."""
        for obj in read(path, json_path="objects.item"):
            if obj.get("type") not in _MEANINGFUL_TYPES:
                continue
            if obj.get("revoked") or obj.get("x_mitre_deprecated"):
                continue
            yield obj

    def normalize(self, record: dict) -> MitreData:
        """Convert one STIX object into the normalized schema."""
        external_id = _extract_attack_id(record)
        content = record.get("description", "")

        return MitreData(
            record_id=f"mitre-attack:{external_id}",
            source_id=self.source_id,
            source_record_id=external_id,
            content=content,
            content_length=compute_token_count(content),
            content_hash=compute_content_hash(content),
            title=record.get("name"),
            raw=record,
            ingested_at=datetime.now(timezone.utc),
            license=MITRE_TERMS,
            published_at=_parse_datetime(record.get("created")),
            source_url=_extract_attack_url(record),
            # knowledge base fields
            framework="attack",
            category_id=external_id,
        )


def _extract_attack_id(record: dict) -> str:
    """Return the ATT&CK external ID (e.g. T1055.011), falling back to the STIX id."""
    ref = _find_mitre_reference(record)
    if ref is not None:
        return ref.get("external_id", record["id"])
    return record["id"]


def _extract_attack_url(record: dict) -> str | None:
    """Return the ATT&CK URL from external_references if present."""
    ref = _find_mitre_reference(record)
    if ref is not None:
        return ref.get("url")
    return None


def _find_mitre_reference(record: dict) -> dict | None:
    """Find the external_references entry with source_name == 'mitre-attack'."""
    for ref in record.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref
    return None


def _parse_datetime(value: str | None) -> datetime | None:
    """Parse STIX timestamps (remove trailing 'Z'); return None if missing."""
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        return None
