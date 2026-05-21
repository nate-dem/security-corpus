from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from lxml import etree

from ingest.connectors.base import MitreData
from ingest.readers import read
from ingest.utils import compute_content_hash, compute_token_count, MITRE_TERMS


_NS = "http://capec.mitre.org/capec-3"
_ATTACK_PATTERN_TAG = f"{{{_NS}}}Attack_Pattern"
_SKIP_STATUSES = {"Deprecated", "Obsolete"}


class CapecConnector:
    source_id = "capec"

    def iter_records(self, path: Path) -> Iterator[dict]:
        for elem in read(path, xml_tag=_ATTACK_PATTERN_TAG):
            status = elem.get("Status", "")
            if status in _SKIP_STATUSES:
                continue
            record = _elem_to_dict(elem)
            # Skip records with no description at all
            if not record.get("description") and not record.get("extended_description"):
                continue
            yield record

    def normalize(self, record: dict) -> MitreData:
        """Convert one CAPEC attack pattern dict into the normalized schema."""
        pattern_id = record["id"]
        capec_id = f"CAPEC-{pattern_id}"

        content = record.get("description", "")
        extended = record.get("extended_description", "")
        if extended:
            content = f"{content}\n\n{extended}" if content else extended

        return MitreData(
            record_id=f"capec:{capec_id}",
            source_id=self.source_id,
            source_record_id=capec_id,
            content=content,
            content_length=compute_token_count(content),
            content_hash=compute_content_hash(content),
            title=record.get("name"),
            raw=record,
            ingested_at=datetime.now(timezone.utc),
            license=MITRE_TERMS,
            source_url=f"https://capec.mitre.org/data/definitions/{pattern_id}.html",
            # knowledge base fields
            framework="capec",
            category_id=capec_id,
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
        "raw_xml": etree.tostring(elem, encoding="unicode"),
    }


def _extract_text(elem: etree._Element, child_tag: str) -> str:
    """Extract concatenated text from a namespaced child element, or empty string."""
    child = elem.find(f"{{{_NS}}}{child_tag}")
    if child is None:
        return ""
    return "".join(child.itertext()).strip()
