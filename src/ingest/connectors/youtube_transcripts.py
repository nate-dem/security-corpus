from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from ingest.connectors.base import TranscriptData
from ingest.utils import CC_BY_4_0, compute_content_hash, compute_token_count


class YouTubeTranscriptsConnector:
    source_id = "youtube-transcripts"

    def iter_records(self, path: Path) -> Iterator[dict]:
        """Yield one row per transcript from cctube_*.parquet shards under path.

        path should be the directory containing the downloaded Parquet shards
        (data/youtube-transcripts/raw/).  Files are walked in sorted order so
        runs are deterministic.  Missing or empty shards are skipped silently.
        """
        for parquet_file in sorted(path.glob("cctube_*.parquet")):
            try:
                table = pq.read_table(parquet_file)
            except Exception:
                continue
            for batch in table.to_batches():
                cols = {name: batch.column(name).to_pylist() for name in batch.schema.names}
                n = batch.num_rows
                for i in range(n):
                    yield {col: cols[col][i] for col in cols}

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
