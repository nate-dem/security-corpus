# tests/test_nvd_connector.py
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ingest.connectors.base import NormalizedData, VulnerabilityData
from ingest.connectors.vulnerability.nvd import NVDConnector, _parse_datetime


SAMPLE_CVE = json.loads(
    (Path(__file__).parent / "fixtures" / "nvd" / "sample_cve.json").read_text()
)

def test_normalize_returns_vulnerability_data():
    result = NVDConnector().normalize(SAMPLE_CVE)
    assert isinstance(result, VulnerabilityData)
    assert isinstance(result, NormalizedData)
    assert result.record_id.startswith("nvd:")
    assert result.source_id == "nvd"
    assert result.source_record_id.startswith("CVE-")

def test_normalize_extracts_english_description():
    result = NVDConnector().normalize(SAMPLE_CVE)
    english = next(d["value"] for d in SAMPLE_CVE["cve"]["descriptions"] if d["lang"] == "en")
    assert result.content == english

def test_normalize_filters_nvd_pseudo_cwes():
    result = NVDConnector().normalize(SAMPLE_CVE)
    for cwe in result.cwe_ids:
        assert cwe.startswith("CWE-")

def test_normalize_preserves_raw():
    result = NVDConnector().normalize(SAMPLE_CVE)
    assert result.raw == SAMPLE_CVE

def test_normalize_populates_new_fields():
    result = NVDConnector().normalize(SAMPLE_CVE)
    assert result.content_hash is not None
    assert result.content_length is not None
    assert result.content_length > 0
    assert result.license is not None
    assert result.cve_id == SAMPLE_CVE["cve"]["id"]


@pytest.mark.parametrize("value", [
    "2020-01-01T04:00:00.123", "2020-01-01T04:00:00.123Z",
    "2020-01-01T05:00:00.123+01:00",
])
def test_nvd_datetime_preserves_utc_instant(value):
    assert _parse_datetime(value) == datetime(2020, 1, 1, 4, 0, 0, 123000, tzinfo=timezone.utc)


@pytest.mark.parametrize("value", [None, "", "not-a-date"])
def test_nvd_missing_or_invalid_timestamp_stays_missing(value):
    assert _parse_datetime(value) is None

