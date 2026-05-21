"""CloudTrail session connector for flaws.cloud dataset.

Groups CloudTrail events into sessions by source IP + time gap.
Single-pass in-memory approach — all filtered events fit in memory.

Scope filters:
  - Exclude repeated IP (1.3M repetitive RunInstances)
  - AWS service IPs are kept (normal background activity)

Session grouping:
  - Events sorted by (sourceIPAddress, eventTime)
  - New session starts when IP changes or time gap exceeds _SESSION_GAP_SECONDS
  - Sessions below _MIN_SESSION_EVENTS are dropped
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ingest.connectors.base import CloudTrailSessionData
from ingest.readers import read
from ingest.utils import FLAWS_CLOUD_PUBLIC, compute_content_hash, compute_token_count


_EXCLUDED_IPS: set[str] = {
    "5.205.62.253",  # 1.3M repetitive RunInstances — unhelpful
}

_MIN_SESSION_EVENTS = 1
# gap threshold (seconds) that splits sessions
_SESSION_GAP_SECONDS = 1800  # 30 minutes


class CloudTrailSessionConnector:
    """Groups CloudTrail events into session-level records.

    Each session is defined by a unique source IP + a 30-minute time gap boundary.
    Designed for the flaws.cloud public dataset.
    """

    def __init__(self):
        self.source_id = "cloudtrail-flaws"

    def iter_records(self, path: Path) -> Iterator[dict]:
        """Yield one session dict per IP+time-window grouping.

        ``path`` is the directory containing flaws_cloudtrail*.json.gz files.
        """
        # Load all events, filtering excluded IPs
        events = _load_and_filter_events(path)

        # Sort by IP then time for session splitting
        events.sort(key=lambda e: (e.get("sourceIPAddress", ""), e.get("eventTime", "")))

        # Split into sessions
        if not events:
            return

        current_ip = events[0].get("sourceIPAddress", "")
        current_session = [events[0]]

        for event in events[1:]:
            ip = event.get("sourceIPAddress", "")
            if ip != current_ip or _exceeds_gap(current_session[-1], event):
                if len(current_session) >= _MIN_SESSION_EVENTS:
                    yield _build_session_record(current_ip, current_session)
                current_ip = ip
                current_session = [event]
            else:
                current_session.append(event)

        # Final session
        if len(current_session) >= _MIN_SESSION_EVENTS:
            yield _build_session_record(current_ip, current_session)

    def normalize(self, record: dict) -> CloudTrailSessionData:
        """Convert a session record dict into CloudTrailSessionData."""
        events = record["events"]
        source_ip = record["source_ip"]
        session_start = record["session_start"]
        session_end = record["session_end"]

        principals = sorted(set(
            _extract_identity_short(e) for e in events
        ))
        actions = sorted(set(e.get("eventName", "") for e in events))
        aws_services = sorted(set(e.get("eventSource", "") for e in events))
        regions = sorted(set(e.get("awsRegion", "") for e in events if e.get("awsRegion")))
        has_errors = any("errorCode" in e for e in events)

        duration = int((session_end - session_start).total_seconds())
        event_count = len(events)

        content = _assemble_session_content(
            source_ip=source_ip,
            duration=duration,
            event_count=event_count,
            principals=principals,
            aws_services=aws_services,
            regions=regions,
            events=events,
        )

        record_id_suffix = f"{source_ip}:{session_start.strftime('%Y%m%dT%H%M%SZ')}"

        return CloudTrailSessionData(
            record_id=f"{self.source_id}:{record_id_suffix}",
            source_id=self.source_id,
            source_record_id=record_id_suffix,
            content=content,
            content_length=compute_token_count(content),
            content_hash=compute_content_hash(content),
            ingested_at=datetime.now(timezone.utc),
            published_at=session_start,
            license=FLAWS_CLOUD_PUBLIC,
            source_ip=source_ip,
            event_count=event_count,
            session_duration_seconds=duration,
            principals=principals,
            actions=actions,
            aws_services=aws_services,
            regions=regions,
            has_errors=has_errors,
        )


def _load_and_filter_events(path: Path) -> list[dict]:
    """Load all CloudTrail events from .json.gz files, excluding bot IPs."""
    events = []
    for gz_file in sorted(path.glob("flaws_cloudtrail*.json.gz")):
        for event in read(gz_file, json_path="Records.item"):
            if event.get("sourceIPAddress", "") not in _EXCLUDED_IPS:
                events.append(event)
    return events


def _exceeds_gap(prev_event: dict, curr_event: dict) -> bool:
    """Check if the time gap between two events exceeds the session threshold."""
    prev_time = _parse_event_time(prev_event)
    curr_time = _parse_event_time(curr_event)
    if prev_time is None or curr_time is None:
        return True
    return (curr_time - prev_time).total_seconds() > _SESSION_GAP_SECONDS


def _parse_event_time(event: dict) -> datetime | None:
    """Parse CloudTrail eventTime string to datetime."""
    time_str = event.get("eventTime")
    if not time_str:
        return None
    try:
        return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _build_session_record(source_ip: str, events: list[dict]) -> dict:
    """Package a list of events into a session record dict."""
    session_start = _parse_event_time(events[0])
    session_end = _parse_event_time(events[-1])
    # Fallback if parsing fails
    if session_start is None:
        session_start = datetime(2017, 1, 1, tzinfo=timezone.utc)
    if session_end is None:
        session_end = session_start

    return {
        "events": events,
        "source_ip": source_ip,
        "session_start": session_start,
        "session_end": session_end,
    }


def _extract_identity_short(event: dict) -> str:
    """Extract short identity name from a CloudTrail event.

    Examples:
        arn:aws:iam::811596193553:root -> root
        arn:aws:iam::811596193553:user/backup -> backup
        arn:aws:sts::811596193553:assumed-role/MyRole/session -> MyRole/session
        Root -> Root
    """
    identity = event.get("userIdentity", {})
    arn = identity.get("arn", "")
    if arn:
        # ARN format: arn:aws:iam::account:type/name or arn:aws:sts::account:type/name
        parts = arn.split(":")
        if len(parts) >= 6:
            resource = parts[5]  # e.g., "root", "user/backup", "assumed-role/X/Y"
            if "/" in resource:
                # user/backup -> backup, assumed-role/X/Y -> X/Y
                return resource.split("/", 1)[1]
            return resource
    # Fallback to identity type
    return identity.get("type", "unknown")


def _assemble_session_content(
    source_ip: str,
    duration: int,
    event_count: int,
    principals: list[str],
    aws_services: list[str],
    regions: list[str],
    events: list[dict],
) -> str:
    """Render a session as a header + raw JSON events.

    Includes full JSON for each event so the model learns the CloudTrail schema.
    """
    import json

    parts = []

    # Header
    parts.append("# CloudTrail Session")
    parts.append(f"Source IP: {source_ip}")
    parts.append(f"Duration: {duration}s | Events: {event_count}")
    parts.append(f"Principals: {', '.join(principals)}")
    parts.append(f"Services: {', '.join(aws_services)}")
    if regions:
        parts.append(f"Regions: {', '.join(regions)}")

    parts.append("")
    parts.append("## Events")
    parts.append("")

    # Full JSON events
    for event in events:
        parts.append(json.dumps(event, separators=(",", ":")))

    return "\n".join(parts)
