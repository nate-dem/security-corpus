import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator

from ingest.connectors.base import NormalizedData
from ingest.readers import read

logger = logging.getLogger(__name__)

_RULE_DIRS = {
    "rules",
    "rules-compliance",
    "rules-dfir",
    "rules-emerging-threats",
    "rules-threat-hunting",
}

_EXCLUDED_DIRS = {
    "deprecated",
    "unsupported",
    "rules-placeholder",
    "tests",
    "regression_data",
    "other",
    "documentation",
    "images",
}

_SKIP_STATUSES = {"deprecated"}

_ATTACK_TECHNIQUE_RE = re.compile(r"^attack\.t(\d+(?:\.\d+)?)$", re.IGNORECASE)


class SigmaConnector:
    source_id = "sigma"

    def iter_records(self, path: Path) -> Iterator[dict]:
        """Walk a Sigma rule directory tree, yielding one parsed rule dict per file."""
        for rule_dir_name in sorted(_RULE_DIRS):
            rule_dir = path / rule_dir_name
            if not rule_dir.is_dir():
                continue
            for pattern in ("*.yml", "*.yaml"):
                for yml_path in sorted(rule_dir.rglob(pattern)):
                    rel = yml_path.relative_to(path)
                    if _is_excluded(rel):
                        continue
                    try:
                        record = next(read(yml_path), None)
                    except Exception:
                        logger.warning("Failed to parse %s, skipping", yml_path, exc_info=True)
                        continue
                    if record is None or not isinstance(record, dict):
                        continue
                    if not record.get("id"):
                        logger.warning("Rule at %s has no id, skipping", yml_path)
                        continue
                    status = str(record.get("status", "")).lower()
                    if status in _SKIP_STATUSES:
                        continue
                    metadata = _derive_path_metadata(yml_path, path, rule_dir_name)
                    record.update(metadata)
                    yield record

    def normalize(self, record: dict) -> NormalizedData:
        """Convert one Sigma rule dict into the normalized schema."""
        rule_id = str(record["id"])
        description = str(record.get("description", "")).strip()
        published_at = _coerce_datetime(record.get("date"))
        modified_at = _coerce_datetime(record.get("modified")) or published_at

        tags = record.get("tags") or []
        attack_ids = _extract_attack_technique_ids(tags)

        raw = dict(record)
        raw["attack_technique_ids"] = attack_ids

        return NormalizedData(
            record_id=f"sigma:{rule_id}",
            source_id=self.source_id,
            source_record_id=rule_id,
            content=description,
            title=record.get("title"),
            raw=raw,
            ingested_at=datetime.now(timezone.utc),
            published_at=published_at,
            modified_at=modified_at,
            source_url=f"https://github.com/SigmaHQ/sigma/blob/master/{record.get('relative_path', '')}",
            language="en",
        )


def _is_excluded(relative_path: Path) -> bool:
    """Check if any segment of the relative path (from repo root) is in the excluded set."""
    return bool(set(relative_path.parts) & _EXCLUDED_DIRS)


def _derive_path_metadata(yml_path: Path, repo_root: Path, rule_dir_name: str) -> dict:
    """Derive rule_category, rule_source_dir, and relative_path from a rule's file path."""
    relative_path = yml_path.relative_to(repo_root)

    # category is the segments between the rule directory and the filename
    parts_after_rule_dir = relative_path.relative_to(rule_dir_name).parts[:-1]
    rule_category = "/".join(parts_after_rule_dir) if parts_after_rule_dir else ""
    return {
        "rule_category": rule_category,
        "rule_source_dir": rule_dir_name,
        "relative_path": str(relative_path),
    }


def _extract_attack_technique_ids(tags: list[str]) -> list[str]:
    """Extract and normalize ATT&CK technique IDs from Sigma tags.

    E.g. ['attack.defense-evasion', 'attack.t1486', 'attack.t1566.001']
    → ['T1486', 'T1566.001']
    """
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        m = _ATTACK_TECHNIQUE_RE.match(tag)
        if m:
            canonical = f"T{m.group(1)}"
            if canonical not in seen:
                seen.add(canonical)
                result.append(canonical)
    return result


def _coerce_datetime(value: str | date | datetime | None) -> datetime | None:
    """Coerce a date/datetime/string to a timezone-aware datetime, or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None