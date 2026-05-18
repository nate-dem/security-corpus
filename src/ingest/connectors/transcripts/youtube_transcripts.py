from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from ingest.connectors.base import TranscriptData
from ingest.utils import CC_BY_4_0, compute_content_hash, compute_token_count


_CHANNEL_ALLOWLIST_PATH = Path("data/youtube-transcripts/security_channels.txt")

MIN_WORD_COUNT = 50

# Columns needed to decide whether a row passes all gates; everything except
# the large transcript text field. Read these first across all row groups, and
# only fetch "text" for row groups that have at least one passing row.
_GATE_COLS = [
    "video_id",
    "channel_id",
    "channel",
    "title",
    "transcription_language",
    "original_language",
    "language_id_method",
    "word_count",
    "character_count",
    "date",
    "video_link",
    "license",
]


def _load_channel_allowlist() -> frozenset[str]:
    if not _CHANNEL_ALLOWLIST_PATH.exists():
        return frozenset()
    ids = {
        line.strip()
        for line in _CHANNEL_ALLOWLIST_PATH.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    return frozenset(ids)


def _is_english(record: dict) -> bool:
    """Return True if the transcript is English.

    Checks transcription_language with prefix matching to catch en-GB, en-US,
    etc. Falls back to original_language when transcription_language is absent.
    Prefers records where language_id_method is "metadata" (YouTube-declared)
    over "detection" (algorithmic), but does not hard-reject detected records.
    """
    t_lang = record.get("transcription_language") or ""
    o_lang = record.get("original_language") or ""
    method = record.get("language_id_method") or ""

    is_en = t_lang.startswith("en") or (not t_lang and o_lang == "en")

    # Reject only if language was declared (metadata) as non-English. If the
    # method is detection/unknown, give the benefit of the doubt when the code
    # starts with "en".
    if method == "metadata" and not is_en:
        return False

    return is_en


def _passes_gate(record: dict, channel_allowlist: frozenset[str]) -> bool:
    """Stage 1 gate: language, word count, and channel allowlist."""
    if not _is_english(record):
        return False
    if (record.get("word_count") or 0) < MIN_WORD_COUNT:
        return False
    return record.get("channel_id") in channel_allowlist


# Backward-compatible alias used by existing tests.
_is_security_relevant = _passes_gate


class YouTubeTranscriptsConnector:
    source_id = "youtube-transcripts"

    def iter_records(self, path: Path, *, filter_security: bool = True) -> Iterator[dict]:
        """Yield one record per transcript from cctube_*.parquet shards.

        Two-pass column projection per row group:
        - Pass 1 reads only _GATE_COLS, skipping the large "text" column.
        - Pass 2 reads "text" only for row groups with at least one passing row,
          then combines it with the gate-column data.

        This avoids loading transcript text for the vast majority of rows that
        fail the channel-allowlist or language gate.
        """
        allowlist = _load_channel_allowlist() if filter_security else frozenset()

        for parquet_file in sorted(path.glob("cctube_*.parquet")):
            try:
                pf = pq.ParquetFile(parquet_file)
            except Exception:
                continue

            for row_group_index in range(pf.metadata.num_row_groups):
                try:
                    meta_table = pf.read_row_group(row_group_index, columns=_GATE_COLS)
                except Exception:
                    continue

                meta_cols = {
                    name: meta_table.column(name).to_pylist()
                    for name in meta_table.schema.names
                }
                meta_records = [
                    {col: meta_cols[col][i] for col in meta_cols}
                    for i in range(meta_table.num_rows)
                ]

                if filter_security:
                    passing_indices = [
                        i
                        for i, record in enumerate(meta_records)
                        if _passes_gate(record, allowlist)
                    ]
                else:
                    passing_indices = list(range(meta_table.num_rows))

                if not passing_indices:
                    continue

                try:
                    text_col = (
                        pf.read_row_group(row_group_index, columns=["text"])
                        .column("text")
                        .to_pylist()
                    )
                except Exception:
                    continue

                for i in passing_indices:
                    record = meta_records[i]
                    record["text"] = text_col[i]
                    yield record

    def normalize(self, record: dict) -> TranscriptData:
        """Convert one YouTube-Commons row into the canonical schema."""
        video_id = record.get("video_id") or ""
        text = record.get("text") or ""
        title = record.get("title") or None
        lang = (
            record.get("transcription_language")
            or record.get("source_language")
            or record.get("original_language")
            or None
        )

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
            raw=None,
            ingested_at=datetime.now(timezone.utc),
            published_at=_parse_date(record.get("date")),
            source_url=record.get("video_link")
            or (f"https://www.youtube.com/watch?v={video_id}" if video_id else None),
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
