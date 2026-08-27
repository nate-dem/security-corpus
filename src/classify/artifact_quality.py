"""Structural quality checks for non-prose artifact sources."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any


ARTIFACT_QUALITY_MODEL = "structural_checks"
ARTIFACT_QUALITY_VERSION = "structural-v1-2026-08-26"

# RESEARCHER: tune thresholds from data audits before using these as filters.
DEFAULT_MIN_CONTENT_LENGTH: int | None = None
DEFAULT_MAX_CONTENT_LENGTH: int | None = None
DEFAULT_MIN_EVENT_COUNT: int | None = None
DEFAULT_MAX_ACTION_REPETITION_RATIO: float | None = None


def score_sigma_row(
    row: Mapping[str, Any],
    *,
    duplicate_count: int = 1,
    min_content_length: int | None = DEFAULT_MIN_CONTENT_LENGTH,
    max_content_length: int | None = DEFAULT_MAX_CONTENT_LENGTH,
    scored_at: str | None = None,
) -> dict[str, Any]:
    """Return structural Sigma artifact-quality fields for one normalized row."""
    parsed_rule, yaml_status = _parse_yaml_rule(row.get("rule_source"))
    flags: list[str] = []

    missing_rule_source = not _nonempty_string(row.get("rule_source"))
    malformed_yaml = yaml_status == "malformed_yaml"
    if missing_rule_source:
        flags.append("missing_rule_source")
    if malformed_yaml:
        flags.append("malformed_yaml")

    rule_id = row.get("rule_id") or _dict_get(parsed_rule, "id")
    title = row.get("title") or _dict_get(parsed_rule, "title")
    logsource = _dict_get(parsed_rule, "logsource")
    detection = _dict_get(parsed_rule, "detection")

    checks = {
        "sigma_missing_id": not _nonempty_string(rule_id),
        "sigma_missing_title": not _nonempty_string(title),
        "sigma_missing_logsource": not isinstance(logsource, dict) or not logsource,
        "sigma_missing_detection": not isinstance(detection, dict) or not detection,
        "sigma_empty_or_trivial_detection": _is_trivial_sigma_detection(detection),
        "sigma_incomplete_rule_source": missing_rule_source,
        "sigma_malformed_yaml": malformed_yaml,
        "sigma_content_length_outlier": _content_length_outlier(
            row.get("content_length"),
            min_content_length=min_content_length,
            max_content_length=max_content_length,
        ),
        "sigma_exact_duplicate_rule": duplicate_count > 1,
    }
    flags.extend(name for name, value in checks.items() if value and name not in flags)

    return {
        **_base_artifact_row(row, "sigma", duplicate_count, flags, scored_at),
        "sigma_yaml_parse_status": yaml_status,
        **checks,
    }


def score_cloudtrail_row(
    row: Mapping[str, Any],
    *,
    duplicate_count: int = 1,
    min_content_length: int | None = DEFAULT_MIN_CONTENT_LENGTH,
    max_content_length: int | None = DEFAULT_MAX_CONTENT_LENGTH,
    min_event_count: int | None = DEFAULT_MIN_EVENT_COUNT,
    max_action_repetition_ratio: float | None = DEFAULT_MAX_ACTION_REPETITION_RATIO,
    scored_at: str | None = None,
) -> dict[str, Any]:
    """Return structural CloudTrail session-quality fields for one row."""
    event_count = _int_or_none(row.get("event_count"))
    duration = _int_or_none(row.get("session_duration_seconds"))
    actions = _as_list(row.get("actions"))
    services = _as_list(row.get("aws_services"))
    principals = _as_list(row.get("principals"))
    action_repetition_ratio = _repetition_ratio(event_count, actions)

    checks = {
        "cloudtrail_missing_event_count": event_count is None,
        "cloudtrail_missing_duration": duration is None,
        "cloudtrail_no_services": len(services) == 0,
        "cloudtrail_no_actions": len(actions) == 0,
        "cloudtrail_no_principals": len(principals) == 0,
        "cloudtrail_insufficient_context": (
            min_event_count is not None
            and event_count is not None
            and event_count < min_event_count
        ),
        "cloudtrail_action_repetition_outlier": (
            max_action_repetition_ratio is not None
            and action_repetition_ratio is not None
            and action_repetition_ratio > max_action_repetition_ratio
        ),
        "cloudtrail_content_length_outlier": _content_length_outlier(
            row.get("content_length"),
            min_content_length=min_content_length,
            max_content_length=max_content_length,
        ),
        "cloudtrail_exact_duplicate_session": duplicate_count > 1,
    }
    flags = [name for name, value in checks.items() if value]

    return {
        **_base_artifact_row(row, "cloudtrail", duplicate_count, flags, scored_at),
        "cloudtrail_event_count": event_count,
        "cloudtrail_session_duration_seconds": duration,
        "cloudtrail_action_count": len(actions),
        "cloudtrail_service_count": len(services),
        "cloudtrail_principal_count": len(principals),
        "cloudtrail_action_repetition_ratio": action_repetition_ratio,
        **checks,
    }


def _base_artifact_row(
    row: Mapping[str, Any],
    family: str,
    duplicate_count: int,
    flags: list[str],
    scored_at: str | None,
) -> dict[str, Any]:
    return {
        "source_id": row.get("source_id"),
        "record_id": row.get("record_id"),
        "content_hash": row.get("content_hash"),
        "artifact_family": family,
        "artifact_quality_model": ARTIFACT_QUALITY_MODEL,
        "artifact_quality_version": ARTIFACT_QUALITY_VERSION,
        "artifact_quality_scored_at": scored_at or datetime.now(timezone.utc).isoformat(),
        "artifact_duplicate_content_hash_count": duplicate_count,
        "artifact_structural_should_review": bool(flags),
        "artifact_quality_flags": sorted(set(flags)),
    }


def _parse_yaml_rule(rule_source: Any) -> tuple[dict[str, Any] | None, str]:
    if not _nonempty_string(rule_source):
        return None, "missing_yaml"
    try:
        import yaml
    except ImportError:
        raise RuntimeError(
            "PyYAML is required for Sigma artifact-quality scoring. "
            'Install classify extras with: pip install -e ".[classify]"'
        ) from None
    try:
        parsed = yaml.safe_load(str(rule_source))
    except Exception:
        return None, "malformed_yaml"
    if not isinstance(parsed, dict):
        return None, "yaml_not_mapping"
    return parsed, "ok"


def _is_trivial_sigma_detection(detection: Any) -> bool:
    if not isinstance(detection, dict) or not detection:
        return True
    condition = detection.get("condition")
    if not _nonempty_string(condition):
        return True
    selection_keys = [key for key in detection if key != "condition"]
    if not selection_keys:
        return True
    return all(_is_empty_detection_value(detection.get(key)) for key in selection_keys)


def _is_empty_detection_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, Mapping):
        return len(value) == 0 or all(_is_empty_detection_value(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0 or all(_is_empty_detection_value(item) for item in value)
    return False


def _content_length_outlier(
    value: Any,
    *,
    min_content_length: int | None,
    max_content_length: int | None,
) -> bool:
    length = _int_or_none(value)
    if length is None:
        return True
    if min_content_length is not None and length < min_content_length:
        return True
    if max_content_length is not None and length > max_content_length:
        return True
    return False


def _repetition_ratio(event_count: int | None, unique_actions: list[Any]) -> float | None:
    if event_count is None or event_count <= 0:
        return None
    return max(0.0, 1.0 - (len(unique_actions) / event_count))


def _dict_get(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return None


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return [value]
