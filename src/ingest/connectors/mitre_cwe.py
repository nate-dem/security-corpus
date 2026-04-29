from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from lxml import etree

from ingest.connectors.base import KnowledgeBaseData
from ingest.readers import read
from ingest.utils import compute_content_hash, compute_token_count, MITRE_TERMS


_NS = "http://cwe.mitre.org/cwe-7"
_WEAKNESS_TAG = f"{{{_NS}}}Weakness"
_CATEGORY_TAG = f"{{{_NS}}}Category"
_VIEW_TAG = f"{{{_NS}}}View"
_SKIP_STATUSES = {"Deprecated", "Obsolete"}


class MitreCweConnector:
    source_id = "mitre-cwe"

    def iter_records(self, path: Path) -> Iterator[dict]:
        """Stream weakness, category, and view dicts from CWE XML catalog,
        skipping deprecated/obsolete entries."""
        for tag, record_type in [
            (_WEAKNESS_TAG, "Weakness"),
            (_CATEGORY_TAG, "Category"),
            (_VIEW_TAG, "View"),
        ]:
            for elem in read(path, xml_tag=tag):
                status = elem.get("Status", "")
                if status in _SKIP_STATUSES:
                    continue
                record = _elem_to_dict(elem)
                record["record_type"] = record_type
                yield record

    def normalize(self, record: dict) -> KnowledgeBaseData:
        """Convert one CWE record dict into the normalized schema."""
        cwe_id = f"CWE-{record['id']}"

        content = record.get("description", "")
        extended = record.get("extended_description", "")
        if extended:
            content = f"{content}\n\n{extended}" if content else extended

        published_at = _parse_date(record.get("submission_date"))

        return KnowledgeBaseData(
            record_id=f"mitre-cwe:{cwe_id}",
            source_id=self.source_id,
            source_record_id=cwe_id,
            content=content,
            content_length=compute_token_count(content),
            content_hash=compute_content_hash(content),
            title=record.get("name"),
            raw=record,
            ingested_at=datetime.now(timezone.utc),
            license=MITRE_TERMS,
            published_at=published_at,
            source_url=f"https://cwe.mitre.org/data/definitions/{record['id']}.html",
            # knowledge base fields
            framework="cwe",
            category_id=cwe_id,
        )

# stored in raw field of NormalizedData class
def _elem_to_dict(elem: etree._Element) -> dict:
    """Convert a Weakness, Category, or View element into a plain dict."""
    return {
        "id": elem.get("ID"),
        "name": elem.get("Name"),
        "abstraction": elem.get("Abstraction"),
        "structure": elem.get("Structure"),
        "status": elem.get("Status"),
        "description": _extract_text(elem, "Description") or _extract_text(elem, "Summary") or _extract_text(elem, "Objective"),
        "extended_description": _extract_text(elem, "Extended_Description"),
        "likelihood_of_exploit": _extract_text(elem, "Likelihood_Of_Exploit"),
        "common_consequences": _extract_common_consequences(elem),
        "applicable_platforms": _extract_applicable_platforms(elem),
        "modes_of_introduction": _extract_modes_of_introduction(elem),
        "submission_date": _extract_submission_date(elem),
        "raw_xml": etree.tostring(elem, encoding="unicode"),
    }


def _extract_text(elem: etree._Element, child_tag: str) -> str:
    """Extract concatenated text from a namespaced child element, or empty string."""
    child = elem.find(f"{{{_NS}}}{child_tag}")
    if child is None:
        return ""
    return "".join(child.itertext()).strip()


def _extract_common_consequences(elem: etree._Element) -> list[dict]:
    """Extract common consequences as a list of {scope, impact, note} dicts."""
    results: list[dict] = []
    parent = elem.find(f"{{{_NS}}}Common_Consequences")
    if parent is None:
        return results
    for consequence in parent.findall(f"{{{_NS}}}Consequence"):
        scopes = [s.text for s in consequence.findall(f"{{{_NS}}}Scope") if s.text]
        impacts = [i.text for i in consequence.findall(f"{{{_NS}}}Impact") if i.text]
        note_elem = consequence.find(f"{{{_NS}}}Note")
        note = "".join(note_elem.itertext()).strip() if note_elem is not None else ""
        results.append({"scopes": scopes, "impacts": impacts, "note": note})
    return results


def _extract_applicable_platforms(elem: etree._Element) -> list[dict]:
    """Extract applicable platforms as a list of dicts with name/class/prevalence."""
    results: list[dict] = []
    parent = elem.find(f"{{{_NS}}}Applicable_Platforms")
    if parent is None:
        return results
    for child in parent:
        results.append({
            "tag": etree.QName(child.tag).localname,
            "name": child.get("Name", ""),
            "class": child.get("Class", ""),
            "prevalence": child.get("Prevalence", ""),
        })
    return results


def _extract_modes_of_introduction(elem: etree._Element) -> list[str]:
    """Extract modes of introduction phases as a list of strings."""
    phases: list[str] = []
    parent = elem.find(f"{{{_NS}}}Modes_Of_Introduction")
    if parent is None:
        return phases
    for intro in parent.findall(f"{{{_NS}}}Introduction"):
        phase = intro.findtext(f"{{{_NS}}}Phase", default="")
        if phase:
            phases.append(phase)
    return phases


def _extract_submission_date(elem: etree._Element) -> str | None:
    """Extract the earliest submission date from Content_History."""
    history = elem.find(f"{{{_NS}}}Content_History")
    if history is None:
        return None
    submission = history.find(f"{{{_NS}}}Submission")
    if submission is None:
        return None
    return submission.findtext(f"{{{_NS}}}Submission_Date", default=None)


def _parse_date(date_str: str | None) -> datetime | None:
    """Parse a YYYY-MM-DD date string into a timezone-aware datetime."""
    if not date_str:
        return None
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
