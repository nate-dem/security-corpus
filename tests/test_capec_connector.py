import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from lxml import etree

from ingest.connectors.base import NormalizedData
from ingest.connectors.capec import CapecConnector


SAMPLE_PATTERN = json.loads(
    (Path(__file__).parent / "fixtures" / "capec" / "sample_pattern.json").read_text()
)


def test_normalize_returns_valid_normalized_data():
    result = CapecConnector().normalize(SAMPLE_PATTERN)
    assert isinstance(result, NormalizedData)
    assert result.record_id == "capec:CAPEC-66"
    assert result.source_id == "capec"
    assert result.source_record_id == "CAPEC-66"


def test_normalize_maps_description_and_title():
    result = CapecConnector().normalize(SAMPLE_PATTERN)
    assert result.title == "SQL Injection"
    assert SAMPLE_PATTERN["description"] in result.content
    assert SAMPLE_PATTERN["extended_description"] in result.content


def test_normalize_extracts_cwe_ids():
    result = CapecConnector().normalize(SAMPLE_PATTERN)
    assert result.cwe_ids == ["CWE-89", "CWE-1286"]


def test_normalize_maps_severity():
    result = CapecConnector().normalize(SAMPLE_PATTERN)
    assert result.severity == "high"


def test_normalize_sets_source_url():
    result = CapecConnector().normalize(SAMPLE_PATTERN)
    assert result.source_url == "https://capec.mitre.org/data/definitions/66.html"


def test_iter_records_filters_deprecated_and_obsolete():
    ns = "http://capec.mitre.org/capec-3"

    def _make_elem(pattern_id, status):
        elem = etree.Element(f"{{{ns}}}Attack_Pattern")
        elem.set("ID", pattern_id)
        elem.set("Name", f"Pattern {pattern_id}")
        elem.set("Abstraction", "Standard")
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

    connector = CapecConnector()
    with patch("ingest.connectors.capec.read", return_value=iter(elements)):
        results = list(connector.iter_records(Path("fake.xml")))

    assert len(results) == 2
    assert results[0]["id"] == "1"
    assert results[1]["id"] == "4"
