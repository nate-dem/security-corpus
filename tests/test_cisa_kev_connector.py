import json
from pathlib import Path

from ingest.connectors.base import NormalizedData
from ingest.connectors.cisa_kev import CisaKevConnector


SAMPLE_RECORD = json.loads(
    (Path(__file__).parent / "fixtures" / "cisa-kev" / "sample_record.json").read_text()
)


def test_normalize_returns_normalized_data():
    result = CisaKevConnector().normalize(SAMPLE_RECORD)
    assert isinstance(result, NormalizedData)
    assert result.record_id == "cisa-kev:CVE-2021-27104"
    assert result.source_id == "cisa-kev"
    assert result.source_record_id == "CVE-2021-27104"


def test_normalize_extracts_fields():
    result = CisaKevConnector().normalize(SAMPLE_RECORD)
    assert result.content == SAMPLE_RECORD["shortDescription"]
    assert result.title == SAMPLE_RECORD["vulnerabilityName"]
    assert "CVE-2021-27104" in result.source_url
    assert result.language == "en"
    assert result.severity is None
    assert result.cvss_score is None


def test_normalize_extracts_cwe_ids():
    result = CisaKevConnector().normalize(SAMPLE_RECORD)
    assert result.cwe_ids == ["CWE-78"]


def test_normalize_preserves_raw():
    result = CisaKevConnector().normalize(SAMPLE_RECORD)
    assert result.raw == SAMPLE_RECORD
    assert "requiredAction" in result.raw
    assert "knownRansomwareCampaignUse" in result.raw
