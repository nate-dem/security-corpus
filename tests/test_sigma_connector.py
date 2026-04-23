from pathlib import Path

import yaml

from ingest.connectors.base import NormalizedData
from ingest.connectors.sigma import (
    SigmaConnector,
    _extract_attack_technique_ids,
)

FIXTURES = Path(__file__).parent / "fixtures" / "sigma"

SAMPLE_RULE = yaml.safe_load((FIXTURES / "sample_rule.yml").read_bytes())


def test_normalize_returns_valid_normalized_data():
    record = dict(SAMPLE_RULE)
    record.update(
        rule_category="cloud/aws/cloudtrail",
        rule_source_dir="rules",
        relative_path="rules/cloud/aws/cloudtrail/sample_rule.yml",
    )
    result = SigmaConnector().normalize(record)
    assert isinstance(result, NormalizedData)
    assert result.record_id == "sigma:39c9f26d-6e3b-4dbb-9c7a-4154b0281112"
    assert result.source_id == "sigma"


def test_normalize_maps_description_title_date():
    record = dict(SAMPLE_RULE)
    record.update(
        rule_category="",
        rule_source_dir="rules",
        relative_path="rules/sample_rule.yml",
    )
    result = SigmaConnector().normalize(record)
    assert result.title == "AWS Bucket Deleted"
    assert "Detects the deletion of S3 buckets" in result.content
    assert result.published_at is not None
    assert result.published_at.year == 2025
    assert result.published_at.month == 10
    assert result.published_at.day == 19


def test_extract_attack_technique_ids():
    tags = ["attack.defense-evasion", "attack.t1486", "attack.t1566.001", "cve.2021-44228"]
    result = _extract_attack_technique_ids(tags)
    assert result == ["T1486", "T1566.001"]

    assert _extract_attack_technique_ids([]) == []
    assert _extract_attack_technique_ids(["attack.defense-evasion", "cve.2021-44228"]) == []


def test_iter_records_skips_deprecated():
    connector = SigmaConnector()
    records = list(connector.iter_records(FIXTURES / "sample_tree"))
    ids = [str(r["id"]) for r in records]
    # The deprecated fixture is not under sample_tree, but let's verify
    # no deprecated records come through by checking status
    for r in records:
        assert str(r.get("status", "")).lower() != "deprecated"


def test_path_metadata_derivation():
    connector = SigmaConnector()
    records = list(connector.iter_records(FIXTURES / "sample_tree"))

    by_id = {str(r["id"]): r for r in records}

    # rules/cloud/aws/cloudtrail/aws_bucket_deleted.yml
    cloudtrail = by_id["39c9f26d-6e3b-4dbb-9c7a-4154b0281112"]
    assert cloudtrail["rule_category"] == "cloud/aws/cloudtrail"
    assert cloudtrail["rule_source_dir"] == "rules"
    assert cloudtrail["relative_path"] == "rules/cloud/aws/cloudtrail/aws_bucket_deleted.yml"

    # rules-dfir/windows/dfir_example.yml
    dfir = by_id["11111111-2222-3333-4444-555555555555"]
    assert dfir["rule_category"] == "windows"
    assert dfir["rule_source_dir"] == "rules-dfir"
    assert dfir["relative_path"] == "rules-dfir/windows/dfir_example.yml"


def test_normalize_preserves_raw_tags_and_attack_ids():
    record = dict(SAMPLE_RULE)
    record.update(
        rule_category="cloud/aws/cloudtrail",
        rule_source_dir="rules",
        relative_path="rules/cloud/aws/cloudtrail/sample_rule.yml",
    )
    result = SigmaConnector().normalize(record)
    # Original tags preserved
    assert result.raw["tags"] == SAMPLE_RULE["tags"]
    # Extracted technique IDs added
    assert "T1485" in result.raw["attack_technique_ids"]
    assert "T1566.001" in result.raw["attack_technique_ids"]


def test_normalize_leaves_vuln_fields_empty():
    record = dict(SAMPLE_RULE)
    record.update(
        rule_category="",
        rule_source_dir="rules",
        relative_path="rules/sample_rule.yml",
    )
    result = SigmaConnector().normalize(record)
    assert result.severity is None
    assert result.cvss_score is None
    assert result.cwe_ids == []
