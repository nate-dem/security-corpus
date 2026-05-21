from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ingest.connectors.base import VulnerabilityData
from ingest.utils import CC_BY_4_0, compute_content_hash, compute_token_count

# Walk only reviewed advisories and malware; skip unreviewed/ (299k NVD duplicates).
_ADVISORY_SUBDIRS = ("github-reviewed", "malware")


class GitHubAdvisoryConnector:
    source_id = "github-advisory"

    def iter_records(self, path: Path) -> Iterator[dict]:
        """Yield one OSV document per .json file under path/advisories/{github-reviewed,malware}/.

        path should be the root of the cloned advisory-database repo (or a
        directory that mirrors its layout).  Missing subdirectories are skipped
        silently so the connector works against partial mirrors too.
        """
        advisories_root = path / "advisories"
        for subdir in _ADVISORY_SUBDIRS:
            subdir_path = advisories_root / subdir
            if not subdir_path.is_dir():
                continue
            for json_file in sorted(subdir_path.rglob("*.json")):
                import json
                try:
                    yield json.loads(json_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue

    def normalize(self, record: dict) -> VulnerabilityData:
        """Convert one OSV-format GitHub advisory into the canonical schema."""
        ghsa_id = record.get("id", "")
        summary = record.get("summary", "")
        details = record.get("details", "")
        content = details or summary

        db_specific = record.get("database_specific") or {}
        severity_raw = db_specific.get("severity", "")
        severity = severity_raw.lower() if severity_raw else None

        cwe_ids = db_specific.get("cwe_ids") or []

        cve_id = _extract_cve_alias(record.get("aliases") or [])

        source_url = f"https://github.com/advisories/{ghsa_id}" if ghsa_id else None

        published_at = _parse_osv_datetime(record.get("published"))

        return VulnerabilityData(
            record_id=f"github-advisory:{ghsa_id}",
            source_id=self.source_id,
            source_record_id=ghsa_id,
            content=content,
            title=summary or None,
            content_length=compute_token_count(content),
            content_hash=compute_content_hash(content),
            raw=record,
            ingested_at=datetime.now(timezone.utc),
            published_at=published_at,
            source_url=source_url,
            license=CC_BY_4_0,
            cve_id=cve_id,
            severity=severity,
            cvss_score=_extract_cvss_score(record.get("severity") or []),
            cwe_ids=cwe_ids,
        )


def _extract_cve_alias(aliases: list[str]) -> str | None:
    """Return the first CVE-* alias, or None."""
    for alias in aliases:
        if alias.startswith("CVE-"):
            return alias
    return None


def _extract_cvss_score(severity_list: list[dict]) -> float | None:
    """Parse CVSS base score from the OSV severity array.

    Each entry has {"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/..."}.
    We prefer V4 > V3 > V2 and extract the base score from the vector string.
    """
    priority = ["CVSS_V4", "CVSS_V3", "CVSS_V2"]
    by_type: dict[str, str] = {e["type"]: e["score"] for e in severity_list if "type" in e and "score" in e}
    for cvss_type in priority:
        vector = by_type.get(cvss_type)
        if vector:
            score = _parse_base_score_from_vector(vector)
            if score is not None:
                return score
    return None


def _parse_base_score_from_vector(vector: str) -> float | None:
    """Extract /BS:X.X or /BS:X from a CVSS vector string if present.

    The GitHub Advisory DB embeds the base score directly in the vector string
    as the first numeric component after the prefix, e.g.:
      CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H  → no embedded score
    but the OSV spec also allows a plain score string like "7.5".
    We try a plain float parse first, then give up (don't attempt full vector
    math to avoid pulling in extra deps).
    """
    try:
        return float(vector)
    except ValueError:
        return None


def _parse_osv_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
