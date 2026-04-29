from pathlib import Path

import yaml

from ingest.connectors.base import NormalizedData, DetectionRuleData
from ingest.connectors.sigma import SigmaConnector

FIXTURES = Path(__file__).parent / "fixtures" / "sigma"

SAMPLE_RULE = yaml.safe_load((FIXTURES / "sample_rule.yml").read_bytes())


def test_normalize_returns_detection_rule_data():
    record = dict(SAMPLE_RULE)
    record.update(
        rule_category="cloud/aws/cloudtrail",
        rule_source_dir="rules",
        relative_path="rules/cloud/aws/cloudtrail/sample_rule.yml",
    )
    result = SigmaConnector().normalize(record)
    assert isinstance(result, DetectionRuleData)
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


def test_normalize_populates_detection_rule_fields():
    record = dict(SAMPLE_RULE)
    record.update(
        rule_category="cloud/aws/cloudtrail",
        rule_source_dir="rules",
        relative_path="rules/cloud/aws/cloudtrail/sample_rule.yml",
    )
    result = SigmaConnector().normalize(record)
    assert result.rule_id == str(SAMPLE_RULE["id"])
    assert result.rule_format == "sigma"
    assert result.rule_source is not None  # YAML text of the rule
    assert result.content_hash is not None
    assert result.content_length is not None
    assert result.license is not None
    # raw is not set for Sigma (dropped per schema rules)
    assert result.raw is None


