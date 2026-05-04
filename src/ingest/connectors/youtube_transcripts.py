from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from ingest.connectors.base import TranscriptData
from ingest.utils import CC_BY_4_0, compute_content_hash, compute_token_count

# Path to the model-classified channel allowlist produced by
# scripts/classify_youtube_channels.py.  Each line is a channel_id.
_CHANNEL_ALLOWLIST_PATH = Path("data/youtube-transcripts/security_channels.txt")

MIN_WORD_COUNT = 100  # drop shorts, music videos, trailers


def _load_channel_allowlist() -> frozenset[str]:
    """Load security channel IDs from the classified allowlist file.

    Returns an empty frozenset if the file doesn't exist yet — in that case
    iter_records with filter_security=True will pass nothing until the
    classify_youtube_channels script has been run.
    """
    if not _CHANNEL_ALLOWLIST_PATH.exists():
        return frozenset()
    ids = {
        line.strip()
        for line in _CHANNEL_ALLOWLIST_PATH.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    return frozenset(ids)


def _is_security_relevant(record: dict, channel_allowlist: frozenset[str]) -> bool:
    """Return True if this record is from a classified security channel.

    Stage 1: English only, minimum word count.
    Stage 2: channel_id present in the model-classified allowlist.
    """
    if record.get("transcription_language") != "en":
        return False
    if (record.get("word_count") or 0) < MIN_WORD_COUNT:
        return False
    return record.get("channel_id") in channel_allowlist


class YouTubeTranscriptsConnector:
    source_id = "youtube-transcripts"

    def iter_records(self, path: Path, *, filter_security: bool = True) -> Iterator[dict]:
        """Yield one row per transcript from cctube_*.parquet shards under path.

        path should be the directory containing the downloaded Parquet shards
        (data/youtube-transcripts/raw/).  Files are walked in sorted order so
        runs are deterministic.  Missing or empty shards are skipped silently.

        filter_security=True (default): only yields English records whose
        channel_id appears in the model-classified allowlist, with word_count >= 100.
        Set filter_security=False to yield all records (full corpus mode).
        """
        allowlist = _load_channel_allowlist() if filter_security else frozenset()
        for parquet_file in sorted(path.glob("cctube_*.parquet")):
            try:
                table = pq.read_table(parquet_file)
            except Exception:
                continue
            for batch in table.to_batches():
                cols = {name: batch.column(name).to_pylist() for name in batch.schema.names}
                n = batch.num_rows
                for i in range(n):
                    record = {col: cols[col][i] for col in cols}
                    if filter_security and not _is_security_relevant(record, allowlist):
                        continue
                    yield record

    def normalize(self, record: dict) -> TranscriptData:
        """Convert one YouTube-Commons row into the canonical schema."""
        video_id = record.get("video_id") or ""
        text = record.get("text") or ""
        title = record.get("title") or None
        lang = record.get("transcription_language") or record.get("source_language") or None

        # record_id must be unique across languages for the same video
        lang_suffix = f":{lang}" if lang else ""
        record_id = f"youtube-transcripts:{video_id}{lang_suffix}"

        row_license = record.get("license")
        license_str = row_license if row_license else CC_BY_4_0

        return TranscriptData(
            record_id=record_id,
            source_id=self.source_id,
            source_record_id=f"{video_id}{lang_suffix}",
            content=text,
            title=title,
            content_length=compute_token_count(text),
            content_hash=compute_content_hash(text),
            raw=None,  # rows are large; skip raw to keep Parquet output lean
            ingested_at=datetime.now(timezone.utc),
            published_at=_parse_date(record.get("date")),
            source_url=record.get("video_link") or (f"https://www.youtube.com/watch?v={video_id}" if video_id else None),
            license=license_str,
            video_id=video_id or None,
            channel=record.get("channel") or None,
            channel_id=record.get("channel_id") or None,
            language=lang,
            word_count=record.get("word_count"),
        )


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None
