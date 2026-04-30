"""
Comprehensive tests for YouTubeTranscriptsConnector.

Organized into six classes:
  TestIterRecords           — shard walking, ordering, edge cases
  TestNormalizeIdentification — record_id / source_id / source_record_id / source_url
  TestNormalizeContent      — text→content, title, hash, token count
  TestNormalizeTranscriptFields — video_id, channel, language, word_count
  TestNormalizeMetadata     — ingested_at, published_at, license, raw=None
  TestNormalizeEndToEnd     — full fixture round-trip, uniqueness invariants
"""
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datetime import timezone
from pathlib import Path

from ingest.connectors.base import NormalizedData, TranscriptData
from ingest.connectors.youtube_transcripts import YouTubeTranscriptsConnector

FIXTURES = Path(__file__).parent / "fixtures" / "youtube-transcripts"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_record(
    *,
    video_id: str = "abc123xyz",
    title: str = "Test Video Title",
    text: str = "This is a test transcript for a security video.",
    channel: str = "TestChannel",
    channel_id: str = "UCtest001",
    date: str = "2023-06-15",
    license: str = "CC-BY 4.0",
    original_language: str = "en",
    source_language: str = "en",
    transcription_language: str = "en",
    word_count: int = 9,
    character_count: int = 48,
) -> dict:
    return {
        "video_id": video_id,
        "video_link": f"https://www.youtube.com/watch?v={video_id}",
        "title": title,
        "text": text,
        "channel": channel,
        "channel_id": channel_id,
        "date": date,
        "license": license,
        "original_language": original_language,
        "source_language": source_language,
        "transcription_language": transcription_language,
        "word_count": word_count,
        "character_count": character_count,
    }


def _normalize(record: dict) -> TranscriptData:
    return YouTubeTranscriptsConnector().normalize(record)


def _write_parquet(tmp_path: Path, rows: list[dict], filename: str = "cctube_0.parquet") -> Path:
    """Write a list of row dicts to a Parquet shard in tmp_path."""
    if not rows:
        table = pa.table({k: pa.array([], type=pa.string()) for k in _make_record()})
    else:
        keys = list(rows[0].keys())
        table = pa.table({k: [r[k] for r in rows] for k in keys})
    dest = tmp_path / filename
    pq.write_table(table, dest)
    return dest


def _all_fixture_records() -> list[dict]:
    return list(YouTubeTranscriptsConnector().iter_records(FIXTURES))


# ---------------------------------------------------------------------------
# TestIterRecords
# ---------------------------------------------------------------------------

class TestIterRecords:
    def test_yields_all_fixture_rows(self):
        """Fixture file has 3 rows."""
        records = _all_fixture_records()
        assert len(records) == 3

    def test_yields_nothing_from_empty_directory(self, tmp_path):
        records = list(YouTubeTranscriptsConnector().iter_records(tmp_path))
        assert records == []

    def test_ignores_non_cctube_parquet_files(self, tmp_path):
        """Files not matching cctube_*.parquet are not read."""
        _write_parquet(tmp_path, [_make_record()], filename="other_data.parquet")
        records = list(YouTubeTranscriptsConnector().iter_records(tmp_path))
        assert records == []

    def test_skips_malformed_parquet_silently(self, tmp_path):
        (tmp_path / "cctube_0.parquet").write_bytes(b"not a parquet file")
        records = list(YouTubeTranscriptsConnector().iter_records(tmp_path))
        assert records == []

    def test_multiple_shards_all_yielded(self, tmp_path):
        _write_parquet(tmp_path, [_make_record(video_id="vid_a")], "cctube_0.parquet")
        _write_parquet(tmp_path, [_make_record(video_id="vid_b")], "cctube_1.parquet")
        records = list(YouTubeTranscriptsConnector().iter_records(tmp_path))
        assert len(records) == 2

    def test_shard_ordering_is_alphabetical(self, tmp_path):
        """Shards are walked in sorted (alphabetical) order."""
        _write_parquet(tmp_path, [_make_record(video_id="second")], "cctube_1.parquet")
        _write_parquet(tmp_path, [_make_record(video_id="first")], "cctube_0.parquet")
        records = list(YouTubeTranscriptsConnector().iter_records(tmp_path))
        assert records[0]["video_id"] == "first"
        assert records[1]["video_id"] == "second"

    def test_all_yielded_records_have_video_id_key(self):
        for record in _all_fixture_records():
            assert "video_id" in record

    def test_handles_empty_parquet_shard(self, tmp_path):
        """A shard with zero rows yields nothing but does not raise."""
        _write_parquet(tmp_path, [], "cctube_0.parquet")
        records = list(YouTubeTranscriptsConnector().iter_records(tmp_path))
        assert records == []


# ---------------------------------------------------------------------------
# TestNormalizeIdentification
# ---------------------------------------------------------------------------

class TestNormalizeIdentification:
    def test_record_id_format(self):
        result = _normalize(_make_record(video_id="abc123xyz", transcription_language="en"))
        assert result.record_id == "youtube-transcripts:abc123xyz:en"

    def test_source_id(self):
        result = _normalize(_make_record())
        assert result.source_id == "youtube-transcripts"

    def test_source_record_id(self):
        result = _normalize(_make_record(video_id="abc123xyz", transcription_language="en"))
        assert result.source_record_id == "abc123xyz:en"

    def test_source_url_from_video_link(self):
        result = _normalize(_make_record(video_id="abc123xyz"))
        assert result.source_url == "https://www.youtube.com/watch?v=abc123xyz"

    def test_source_url_constructed_when_link_absent(self):
        record = _make_record(video_id="abc123xyz")
        del record["video_link"]
        result = _normalize(record)
        assert result.source_url == "https://www.youtube.com/watch?v=abc123xyz"

    def test_source_url_is_none_when_both_link_and_id_absent(self):
        record = _make_record()
        record["video_id"] = ""
        record["video_link"] = ""
        result = _normalize(record)
        assert result.source_url is None

    def test_record_id_includes_language(self):
        result_en = _normalize(_make_record(video_id="v1", transcription_language="en"))
        result_fr = _normalize(_make_record(video_id="v1", transcription_language="fr"))
        assert result_en.record_id != result_fr.record_id


# ---------------------------------------------------------------------------
# TestNormalizeContent
# ---------------------------------------------------------------------------

class TestNormalizeContent:
    def test_text_maps_to_content(self):
        result = _normalize(_make_record(text="Specific transcript text here."))
        assert result.content == "Specific transcript text here."

    def test_title_maps_to_title(self):
        result = _normalize(_make_record(title="My Video Title"))
        assert result.title == "My Video Title"

    def test_title_is_none_when_empty(self):
        record = _make_record(title="")
        result = _normalize(record)
        assert result.title is None

    def test_content_hash_is_sha256(self):
        result = _normalize(_make_record(text="hello world"))
        assert result.content_hash is not None
        assert len(result.content_hash) == 64

    def test_content_length_is_positive_for_nonempty_text(self):
        result = _normalize(_make_record(text="hello world"))
        assert result.content_length is not None
        assert result.content_length > 0

    def test_content_length_is_zero_for_empty_text(self):
        result = _normalize(_make_record(text=""))
        assert result.content_length == 0

    def test_same_text_produces_same_hash(self):
        r1 = _normalize(_make_record(video_id="v1", text="identical"))
        r2 = _normalize(_make_record(video_id="v2", text="identical"))
        assert r1.content_hash == r2.content_hash

    def test_different_text_produces_different_hash(self):
        r1 = _normalize(_make_record(text="text one"))
        r2 = _normalize(_make_record(text="text two"))
        assert r1.content_hash != r2.content_hash


# ---------------------------------------------------------------------------
# TestNormalizeTranscriptFields
# ---------------------------------------------------------------------------

class TestNormalizeTranscriptFields:
    def test_video_id_mapped(self):
        result = _normalize(_make_record(video_id="abc123xyz"))
        assert result.video_id == "abc123xyz"

    def test_channel_mapped(self):
        result = _normalize(_make_record(channel="SecurityAcademy"))
        assert result.channel == "SecurityAcademy"

    def test_channel_id_mapped(self):
        result = _normalize(_make_record(channel_id="UCsec001"))
        assert result.channel_id == "UCsec001"

    def test_language_from_transcription_language(self):
        result = _normalize(_make_record(transcription_language="fr"))
        assert result.language == "fr"

    def test_language_falls_back_to_source_language(self):
        record = _make_record(source_language="de")
        del record["transcription_language"]
        result = _normalize(record)
        assert result.language == "de"

    def test_word_count_mapped(self):
        result = _normalize(_make_record(word_count=42))
        assert result.word_count == 42

    def test_video_id_none_when_empty(self):
        record = _make_record()
        record["video_id"] = ""
        result = _normalize(record)
        assert result.video_id is None

    def test_channel_none_when_empty(self):
        record = _make_record()
        record["channel"] = ""
        result = _normalize(record)
        assert result.channel is None


# ---------------------------------------------------------------------------
# TestNormalizeMetadata
# ---------------------------------------------------------------------------

class TestNormalizeMetadata:
    def test_ingested_at_is_set(self):
        result = _normalize(_make_record())
        assert result.ingested_at is not None

    def test_ingested_at_is_utc(self):
        result = _normalize(_make_record())
        assert result.ingested_at.tzinfo == timezone.utc

    def test_published_at_parsed_from_date(self):
        result = _normalize(_make_record(date="2023-06-15"))
        assert result.published_at is not None
        assert result.published_at.year == 2023
        assert result.published_at.month == 6
        assert result.published_at.day == 15

    def test_published_at_is_none_when_date_absent(self):
        record = _make_record()
        record["date"] = ""
        result = _normalize(record)
        assert result.published_at is None

    def test_license_from_row(self):
        result = _normalize(_make_record(license="CC-BY 4.0"))
        assert result.license == "CC-BY 4.0"

    def test_license_falls_back_to_cc_by_40(self):
        record = _make_record()
        record["license"] = ""
        result = _normalize(record)
        assert result.license == "CC-BY-4.0"

    def test_raw_is_none(self):
        """Rows are large; raw is intentionally not stored."""
        result = _normalize(_make_record())
        assert result.raw is None

    def test_returns_transcript_data_subclass(self):
        result = _normalize(_make_record())
        assert isinstance(result, TranscriptData)
        assert type(result) is TranscriptData


# ---------------------------------------------------------------------------
# TestNormalizeEndToEnd
# ---------------------------------------------------------------------------

class TestNormalizeEndToEnd:
    def test_all_fixture_records_normalize_without_error(self):
        connector = YouTubeTranscriptsConnector()
        for record in connector.iter_records(FIXTURES):
            result = connector.normalize(record)
            assert isinstance(result, TranscriptData)

    def test_all_normalized_record_ids_are_unique(self):
        connector = YouTubeTranscriptsConnector()
        ids = [connector.normalize(r).record_id for r in connector.iter_records(FIXTURES)]
        assert len(ids) == len(set(ids)), f"Duplicate record_ids: {ids}"

    def test_all_normalized_records_have_source_id(self):
        connector = YouTubeTranscriptsConnector()
        for record in connector.iter_records(FIXTURES):
            assert connector.normalize(record).source_id == "youtube-transcripts"

    def test_all_fixture_records_have_non_empty_content(self):
        connector = YouTubeTranscriptsConnector()
        for record in connector.iter_records(FIXTURES):
            result = connector.normalize(record)
            assert len(result.content) > 0, f"Empty content for {record.get('video_id')}"

    def test_all_fixture_records_have_language(self):
        connector = YouTubeTranscriptsConnector()
        for record in connector.iter_records(FIXTURES):
            result = connector.normalize(record)
            assert result.language is not None

    def test_fixture_sql_injection_video(self):
        connector = YouTubeTranscriptsConnector()
        records = list(connector.iter_records(FIXTURES))
        target = next(r for r in records if r["video_id"] == "abc123xyz")
        result = connector.normalize(target)
        assert result.channel == "SecurityAcademy"
        assert result.published_at.year == 2023
        assert type(result) is TranscriptData  # not VulnerabilityData — no cve_id field

    def test_multi_language_same_video_distinct_ids(self, tmp_path):
        """Two transcripts of the same video in different languages get distinct record_ids."""
        rows = [
            _make_record(video_id="shared_video", transcription_language="en"),
            _make_record(video_id="shared_video", transcription_language="fr"),
        ]
        _write_parquet(tmp_path, rows, "cctube_0.parquet")
        connector = YouTubeTranscriptsConnector()
        ids = [connector.normalize(r).record_id for r in connector.iter_records(tmp_path)]
        assert len(ids) == 2
        assert len(set(ids)) == 2
