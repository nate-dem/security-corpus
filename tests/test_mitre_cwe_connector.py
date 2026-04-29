from pathlib import Path
from unittest.mock import patch

from lxml import etree

from ingest.connectors.base import NormalizedData, MitreData
from ingest.connectors.mitre_cwe import MitreCweConnector


FIXTURES = Path(__file__).parent / "fixtures" / "mitre-cwe"


def _parse_first_element(fixture_path: Path, tag: str) -> etree._Element:
    """Parse a fixture XML and return the first matching element."""
    ns = "http://cwe.mitre.org/cwe-7"
    tree = etree.parse(str(fixture_path))
    return tree.find(f".//{{{ns}}}{tag}")


def _load_record(fixture_path: Path, tag: str, record_type: str) -> dict:
    """Load a fixture, convert to dict via the connector's pipeline, and tag it."""
    from ingest.connectors.mitre_cwe import _elem_to_dict
    elem = _parse_first_element(fixture_path, tag)
    record = _elem_to_dict(elem)
    record["record_type"] = record_type
    return record


def test_normalize_weakness_returns_mitre_data():
    record = _load_record(FIXTURES / "sample_weakness.xml", "Weakness", "Weakness")
    result = MitreCweConnector().normalize(record)
    assert isinstance(result, MitreData)
    assert isinstance(result, NormalizedData)
    assert result.record_id == "mitre-cwe:CWE-79"
    assert result.source_id == "mitre-cwe"
    assert result.source_record_id == "CWE-79"


def test_normalize_maps_description_and_title():
    record = _load_record(FIXTURES / "sample_weakness.xml", "Weakness", "Weakness")
    result = MitreCweConnector().normalize(record)
    assert result.title == "Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')"
    assert record["description"] in result.content
    assert record["extended_description"] in result.content


def test_iter_records_filters_deprecated():
    """Deprecated records should be excluded; non-deprecated kept."""
    connector = MitreCweConnector()

    ns = "http://cwe.mitre.org/cwe-7"

    def _make_elem(weakness_id, status):
        elem = etree.Element(f"{{{ns}}}Weakness")
        elem.set("ID", weakness_id)
        elem.set("Name", f"Weakness {weakness_id}")
        elem.set("Abstraction", "Base")
        elem.set("Structure", "Simple")
        elem.set("Status", status)
        desc = etree.SubElement(elem, f"{{{ns}}}Description")
        desc.text = "Test description"
        return elem

    elements = [
        _make_elem("1", "Stable"),
        _make_elem("2", "Deprecated"),
        _make_elem("3", "Obsolete"),
        _make_elem("4", "Draft"),
    ]

    # Mock read to return elements only for the Weakness tag, empty for others
    def mock_read(path, xml_tag=None):
        weakness_tag = f"{{{ns}}}Weakness"
        if xml_tag == weakness_tag:
            return iter(elements)
        return iter([])

    with patch("ingest.connectors.mitre_cwe.read", side_effect=mock_read):
        results = list(connector.iter_records(Path("fake.xml")))

    assert len(results) == 2
    assert results[0]["id"] == "1"
    assert results[1]["id"] == "4"


def test_category_and_view_normalize_successfully():
    connector = MitreCweConnector()

    cat_record = _load_record(FIXTURES / "sample_category.xml", "Category", "Category")
    cat_result = connector.normalize(cat_record)
    assert isinstance(cat_result, MitreData)
    assert cat_result.record_id == "mitre-cwe:CWE-1001"
    assert cat_record["record_type"] == "Category"

    view_record = _load_record(FIXTURES / "sample_view.xml", "View", "View")
    view_result = connector.normalize(view_record)
    assert isinstance(view_result, MitreData)
    assert view_result.record_id == "mitre-cwe:CWE-1000"
    assert view_record["record_type"] == "View"

    # record_type distinguishes them
    assert cat_record["record_type"] != view_record["record_type"]


def test_normalize_sets_source_url():
    record = _load_record(FIXTURES / "sample_weakness.xml", "Weakness", "Weakness")
    result = MitreCweConnector().normalize(record)
    assert result.source_url == "https://cwe.mitre.org/data/definitions/79.html"


def test_normalize_populates_new_fields():
    record = _load_record(FIXTURES / "sample_weakness.xml", "Weakness", "Weakness")
    result = MitreCweConnector().normalize(record)
    assert result.content_hash is not None
    assert result.content_length is not None
    assert result.content_length > 0
    assert result.license is not None
    assert result.framework == "cwe"
    assert result.category_id == "CWE-79"
