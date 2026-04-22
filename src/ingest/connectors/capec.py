from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from lxml import etree

from ingest.connectors.base import NormalizedData
from ingest.readers import read


_NS = "http://capec.mitre.org/capec-3"
_ATTACK_PATTERN_TAG = f"{{{_NS}}}Attack_Pattern"
_SKIP_STATUSES = {"Deprecated", "Obsolete"}


class CapecConnector:
    source_id = "capec"

    def iter_records(self, path: Path) -> Iterator[dict]:
        """Stream attack pattern dicts from CAPEC XML catalog, skipping deprecated/obsolete entries."""
        for elem in read(path, xml_tag=_ATTACK_PATTERN_TAG):
            status = elem.get("Status", "")
            if status in _SKIP_STATUSES:
                continue
            yield _elem_to_dict(elem)

    def normalize(self, record: dict) -> NormalizedData:
        """Convert one CAPEC attack pattern dict into the normalized schema."""
        pattern_id = record["id"]
        capec_id = f"CAPEC-{pattern_id}"

        content = record.get("description", "")
        extended = record.get("extended_description", "")
        if extended:
            content = f"{content}\n\n{extended}" if content else extended

        severity = record.get("typical_severity")

        return NormalizedData(
            record_id=f"capec:{capec_id}",
            source_id=self.source_id,
            source_record_id=capec_id,
            content=content,
            title=record.get("name"),
            raw=record,
            ingested_at=datetime.now(timezone.utc),
            severity=severity.lower() if severity else None,
            cwe_ids=record.get("related_cwe_ids", []),
            source_url=f"https://capec.mitre.org/data/definitions/{pattern_id}.html",
            language="en",
        )


def _elem_to_dict(elem: etree._Element) -> dict:
    """Convert an <Attack_Pattern> element into a plain dict for normalization."""
    return {
        "id": elem.get("ID"),
        "name": elem.get("Name"),
        "abstraction": elem.get("Abstraction"),
        "status": elem.get("Status"),
        "description": _extract_text(elem, "Description"),
        "extended_description": _extract_text(elem, "Extended_Description"),
        "likelihood_of_attack": _extract_text(elem, "Likelihood_Of_Attack"),
        "typical_severity": _extract_text(elem, "Typical_Severity"),
        "related_cwe_ids": _extract_cwe_ids(elem),
        "raw_xml": etree.tostring(elem, encoding="unicode"),
    }


def _extract_text(elem: etree._Element, child_tag: str) -> str:
    """Extract concatenated text from a namespaced child element, or empty string."""
    child = elem.find(f"{{{_NS}}}{child_tag}")
    if child is None:
        return ""
    return "".join(child.itertext()).strip()


def _extract_cwe_ids(elem: etree._Element) -> list[str]:
    cwe_ids: list[str] = []
    weaknesses = elem.find(f"{{{_NS}}}Related_Weaknesses")
    if weaknesses is None:
        return cwe_ids
    for rw in weaknesses.findall(f"{{{_NS}}}Related_Weakness"):
        cwe_id = rw.get("CWE_ID")
        if cwe_id:
            formatted = f"CWE-{cwe_id}"
            if formatted not in cwe_ids:
                cwe_ids.append(formatted)
    return cwe_ids