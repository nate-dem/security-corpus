# tests/test_nvd_connector.py
import json
from pathlib import Path

from ingest.connectors.base import NormalizedData
from ingest.connectors.nvd import NVDConnector


SAMPLE_CVE = json.loads(
    (Path(__file__).parent / "fixtures" / "nvd" / "sample_cve.json").read_text()
)

def test_normalize_returns_normalized_data():
    result = NVDConnector().normalize(SAMPLE_CVE)
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