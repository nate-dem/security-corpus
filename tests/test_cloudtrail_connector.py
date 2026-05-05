"""Tests for the CloudTrail session connector."""

import gzip
import json
from pathlib import Path

import pytest

from ingest.connectors.cloudtrail import (
    CloudTrailSessionConnector,
    _extract_identity_short,
    _assemble_session_content,
    _EXCLUDED_IPS,
)


def _write_cloudtrail_gz(path: Path, filename: str, events: list[dict]):
    """Write a CloudTrail .json.gz fixture file."""
    filepath = path / filename
    data = {"Records": events}
    with gzip.open(filepath, "wt", encoding="utf-8") as f:
        json.dump(data, f)


def _make_event(
    ip: str = "10.0.0.1",
    event_time: str = "2020-01-01T00:00:00Z",
    event_name: str = "ListBuckets",
    event_source: str = "s3.amazonaws.com",
    arn: str = "arn:aws:iam::123456789012:root",
    error_code: str | None = None,
    request_params: dict | None = None,
) -> dict:
    """Create a minimal CloudTrail event dict."""
    event = {
        "sourceIPAddress": ip,
        "eventTime": event_time,
        "eventName": event_name,
        "eventSource": event_source,
        "eventID": "test-event-id",
        "eventType": "AwsApiCall",
        "awsRegion": "us-east-1",
        "userIdentity": {
            "type": "Root",
            "arn": arn,
        },
    }
    if error_code:
        event["errorCode"] = error_code
    if request_params:
        event["requestParameters"] = request_params
    return event


# --- Identity extraction tests ---


class TestExtractIdentityShort:
    def test_root(self):
        event = _make_event(arn="arn:aws:iam::123456789012:root")
        assert _extract_identity_short(event) == "root"

    def test_user(self):
        event = _make_event(arn="arn:aws:iam::123456789012:user/backup")
        assert _extract_identity_short(event) == "backup"

    def test_assumed_role(self):
        event = _make_event(arn="arn:aws:sts::123456789012:assumed-role/MyRole/session1")
        assert _extract_identity_short(event) == "MyRole/session1"

    def test_no_arn_fallback(self):
        event = {"userIdentity": {"type": "AWSService"}}
        assert _extract_identity_short(event) == "AWSService"

    def test_empty_identity(self):
        event = {}
        assert _extract_identity_short(event) == "unknown"




# --- Session splitting tests ---


class TestIterRecords:
    def test_basic_session_split(self, tmp_path):
        """Two IPs produce two sessions."""
        events = [
            _make_event(ip="10.0.0.1", event_time="2020-01-01T00:00:00Z"),
            _make_event(ip="10.0.0.1", event_time="2020-01-01T00:05:00Z"),
            _make_event(ip="10.0.0.2", event_time="2020-01-01T00:00:00Z"),
        ]
        _write_cloudtrail_gz(tmp_path, "flaws_cloudtrail00.json.gz", events)

        connector = CloudTrailSessionConnector()
        sessions = list(connector.iter_records(tmp_path))
        assert len(sessions) == 2

    def test_time_gap_splits_session(self, tmp_path):
        """Same IP with >30min gap produces two sessions."""
        events = [
            _make_event(ip="10.0.0.1", event_time="2020-01-01T00:00:00Z"),
            _make_event(ip="10.0.0.1", event_time="2020-01-01T00:10:00Z"),
            _make_event(ip="10.0.0.1", event_time="2020-01-01T01:00:00Z"),  # 50min gap
            _make_event(ip="10.0.0.1", event_time="2020-01-01T01:05:00Z"),
        ]
        _write_cloudtrail_gz(tmp_path, "flaws_cloudtrail00.json.gz", events)

        connector = CloudTrailSessionConnector()
        sessions = list(connector.iter_records(tmp_path))
        assert len(sessions) == 2
        assert len(sessions[0]["events"]) == 2
        assert len(sessions[1]["events"]) == 2

    def test_excluded_ip_filtered(self, tmp_path):
        """Bot IP events are excluded."""
        bot_ip = next(iter(_EXCLUDED_IPS))
        events = [
            _make_event(ip=bot_ip, event_time="2020-01-01T00:00:00Z"),
            _make_event(ip=bot_ip, event_time="2020-01-01T00:01:00Z"),
            _make_event(ip="10.0.0.1", event_time="2020-01-01T00:00:00Z"),
        ]
        _write_cloudtrail_gz(tmp_path, "flaws_cloudtrail00.json.gz", events)

        connector = CloudTrailSessionConnector()
        sessions = list(connector.iter_records(tmp_path))
        assert len(sessions) == 1
        assert sessions[0]["source_ip"] == "10.0.0.1"

    def test_multiple_files(self, tmp_path):
        """Events across multiple .json.gz files are combined."""
        _write_cloudtrail_gz(tmp_path, "flaws_cloudtrail00.json.gz", [
            _make_event(ip="10.0.0.1", event_time="2020-01-01T00:00:00Z"),
        ])
        _write_cloudtrail_gz(tmp_path, "flaws_cloudtrail01.json.gz", [
            _make_event(ip="10.0.0.1", event_time="2020-01-01T00:05:00Z"),
        ])

        connector = CloudTrailSessionConnector()
        sessions = list(connector.iter_records(tmp_path))
        assert len(sessions) == 1
        assert len(sessions[0]["events"]) == 2


# --- Normalize tests ---


class TestNormalize:
    def _make_session_record(self):
        events = [
            _make_event(
                ip="10.0.0.1",
                event_time="2020-01-01T00:00:00Z",
                event_name="ListBuckets",
                event_source="s3.amazonaws.com",
                arn="arn:aws:iam::123456789012:root",
            ),
            _make_event(
                ip="10.0.0.1",
                event_time="2020-01-01T00:05:00Z",
                event_name="GetBucketAcl",
                event_source="s3.amazonaws.com",
                arn="arn:aws:iam::123456789012:user/backup",
                request_params={"bucketName": "flaws.cloud"},
            ),
            _make_event(
                ip="10.0.0.1",
                event_time="2020-01-01T00:06:00Z",
                event_name="GetObject",
                event_source="s3.amazonaws.com",
                arn="arn:aws:iam::123456789012:user/backup",
                error_code="AccessDenied",
                request_params={"bucketName": "flaws.cloud", "key": "secret.txt"},
            ),
        ]
        from datetime import datetime, timezone
        return {
            "events": events,
            "source_ip": "10.0.0.1",
            "session_start": datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            "session_end": datetime(2020, 1, 1, 0, 6, 0, tzinfo=timezone.utc),
        }

    def test_event_count(self):
        connector = CloudTrailSessionConnector()
        record = self._make_session_record()
        result = connector.normalize(record)
        assert result.event_count == 3

    def test_duration(self):
        connector = CloudTrailSessionConnector()
        record = self._make_session_record()
        result = connector.normalize(record)
        assert result.session_duration_seconds == 360  # 6 minutes

    def test_principals(self):
        connector = CloudTrailSessionConnector()
        record = self._make_session_record()
        result = connector.normalize(record)
        assert "root" in result.principals
        assert "backup" in result.principals

    def test_actions(self):
        connector = CloudTrailSessionConnector()
        record = self._make_session_record()
        result = connector.normalize(record)
        assert "ListBuckets" in result.actions
        assert "GetBucketAcl" in result.actions
        assert "GetObject" in result.actions

    def test_has_errors(self):
        connector = CloudTrailSessionConnector()
        record = self._make_session_record()
        result = connector.normalize(record)
        assert result.has_errors is True

    def test_content_has_header(self):
        connector = CloudTrailSessionConnector()
        record = self._make_session_record()
        result = connector.normalize(record)
        assert "# CloudTrail Session" in result.content
        assert "Source IP: 10.0.0.1" in result.content
        assert "## Events" in result.content

    def test_content_has_events_as_json(self):
        connector = CloudTrailSessionConnector()
        record = self._make_session_record()
        result = connector.normalize(record)
        assert '"eventName":"ListBuckets"' in result.content
        assert '"eventName":"GetBucketAcl"' in result.content
        assert '"errorCode":"AccessDenied"' in result.content
        assert '"bucketName":"flaws.cloud"' in result.content

    def test_content_hash_populated(self):
        connector = CloudTrailSessionConnector()
        record = self._make_session_record()
        result = connector.normalize(record)
        assert result.content_hash is not None
        assert len(result.content_hash) == 64  # SHA-256 hex

    def test_content_length_populated(self):
        connector = CloudTrailSessionConnector()
        record = self._make_session_record()
        result = connector.normalize(record)
        assert result.content_length > 0

    def test_license(self):
        connector = CloudTrailSessionConnector()
        record = self._make_session_record()
        result = connector.normalize(record)
        assert "flaws.cloud" in result.license


# --- Content assembly tests ---


class TestContentAssembly:
    def test_all_events_included(self):
        """All events are rendered in content without truncation."""
        events = [
            _make_event(event_time=f"2020-01-01T00:{i:02d}:00Z")
            for i in range(50)
        ]

        content = _assemble_session_content(
            source_ip="10.0.0.1",
            duration=3600,
            event_count=len(events),
            principals=["root"],
            aws_services=["s3.amazonaws.com"],
            regions=["us-east-1"],
            events=events,
        )
        # Each event becomes one JSON line after the header
        json_lines = [l for l in content.split("\n") if l.startswith("{")]
        assert len(json_lines) == 50


# --- End-to-end test ---


class TestEndToEnd:
    def test_round_trip(self, tmp_path):
        """Full iter_records -> normalize round trip."""
        events = [
            _make_event(ip="192.168.1.1", event_time="2020-06-15T10:00:00Z",
                        event_name="GetCallerIdentity", event_source="sts.amazonaws.com"),
            _make_event(ip="192.168.1.1", event_time="2020-06-15T10:01:00Z",
                        event_name="ListBuckets", event_source="s3.amazonaws.com"),
            _make_event(ip="192.168.1.1", event_time="2020-06-15T10:02:00Z",
                        event_name="GetBucketAcl", event_source="s3.amazonaws.com",
                        request_params={"bucketName": "target-bucket"}),
        ]
        _write_cloudtrail_gz(tmp_path, "flaws_cloudtrail00.json.gz", events)

        connector = CloudTrailSessionConnector()
        records = list(connector.iter_records(tmp_path))
        assert len(records) == 1

        normalized = connector.normalize(records[0])
        assert normalized.source_id == "cloudtrail-flaws"
        assert normalized.event_count == 3
        assert normalized.source_ip == "192.168.1.1"
        assert '"eventName":"GetCallerIdentity"' in normalized.content
        assert '"eventName":"ListBuckets"' in normalized.content
        assert '"bucketName":"target-bucket"' in normalized.content
