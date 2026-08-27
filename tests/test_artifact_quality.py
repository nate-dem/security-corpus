from classify.artifact_quality import score_cloudtrail_row, score_sigma_row


def test_score_sigma_row_flags_valid_rule_as_clean():
    row = {
        "source_id": "sigma",
        "record_id": "sigma:rule-1",
        "content_hash": "h",
        "content_length": 120,
        "rule_id": "rule-1",
        "title": "Suspicious Login",
        "rule_source": """
id: rule-1
title: Suspicious Login
logsource:
  product: aws
detection:
  selection:
    eventName: ConsoleLogin
  condition: selection
""",
    }

    scored = score_sigma_row(row)

    assert scored["artifact_structural_should_review"] is False
    assert scored["sigma_yaml_parse_status"] == "ok"
    assert scored["sigma_missing_detection"] is False
    assert scored["sigma_empty_or_trivial_detection"] is False


def test_score_sigma_row_flags_malformed_and_missing_detection():
    row = {
        "source_id": "sigma",
        "record_id": "sigma:rule-2",
        "content_hash": "h",
        "content_length": 10,
        "rule_source": "id: [broken",
    }

    scored = score_sigma_row(row, duplicate_count=2, min_content_length=20)

    assert scored["artifact_structural_should_review"] is True
    assert scored["sigma_malformed_yaml"] is True
    assert scored["sigma_missing_detection"] is True
    assert scored["sigma_content_length_outlier"] is True
    assert scored["sigma_exact_duplicate_rule"] is True


def test_score_cloudtrail_row_computes_diversity_and_repetition():
    row = {
        "source_id": "cloudtrail-flaws",
        "record_id": "cloudtrail-flaws:1",
        "content_hash": "h",
        "content_length": 500,
        "event_count": 4,
        "session_duration_seconds": 60,
        "actions": ["RunInstances"],
        "aws_services": ["ec2.amazonaws.com"],
        "principals": ["root"],
    }

    scored = score_cloudtrail_row(row, max_action_repetition_ratio=0.7)

    assert scored["cloudtrail_event_count"] == 4
    assert scored["cloudtrail_action_count"] == 1
    assert scored["cloudtrail_action_repetition_ratio"] == 0.75
    assert scored["cloudtrail_action_repetition_outlier"] is True
    assert "cloudtrail_action_repetition_outlier" in scored["artifact_quality_flags"]
