"""YouTube-Commons ingest connector.

Flow overview
-------------
Raw data comes from the PleIAs/YouTube-Commons HuggingFace dataset, downloaded
as cctube_N.parquet shards by scripts/export_youtube_transcripts.py.

Before ingesting, scripts/classify_youtube_videos.py runs on Marlowe HPC to
classify every English-language record using a vLLM metadata classifier.
It writes a per-shard JSONL cache that is merged into a single file by
scripts/merge_classifier_caches.py.

At ingest time, iter_records() applies three gates in sequence:

  Gate 1 — Language        : keep English transcripts only
  Gate 2 — Word count      : keep records with >= MIN_WORD_COUNT words
  Gate 3 — Classifier cache: keep records whose video_id appears in the
                             keep set produced by the classifier

Gates 1 and 2 run cheaply on pass-1 metadata (no text column loaded).
The text column is only fetched in pass 2 for records that survive all gates,
which avoids loading gigabytes of transcript text that would be discarded.

Gate 3 is skipped when no cache_path is provided.  This lets the connector
run without a pre-built cache (producing a broader, unfiltered result) or
with a cache (producing the classifier-filtered result).
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from ingest.connectors.base import TranscriptData
from ingest.connectors.youtube_classifier import load_keep_set
from ingest.utils import CC_BY_4_0, compute_content_hash, compute_token_count

# Minimum transcript length accepted.  Transcripts shorter than this are
# almost always auto-caption fragments or upload artifacts.
MIN_WORD_COUNT = 50

# Columns read in pass 1 (the cheap metadata pass).
# The large "text" column is deliberately excluded here and fetched only in
# pass 2 for records that survive all three gates.
#
# Language fields (all exist in the official HuggingFace dataset YAML):
#   transcription_language — language of the text field (original or translated)
#   source_language        — language of the source audio before any translation
#   original_language      — language the video was originally spoken in
#   language_id_method     — how the language was detected (present in cctube_*
#                            shards; may be absent in official train-* split)
#
# Schema resilience: not all shards expose identical columns (cctube_* vs
# train-* differ slightly — e.g. cctube_0.parquet lacks source_language while
# the official schema includes it).  iter_records() intersects _GATE_COLS with
# each file's actual schema before reading, so a missing column is skipped
# rather than raising an error or silently returning all-None values.
_GATE_COLS = [
    "video_id",
    "channel_id",
    "channel",
    "title",
    "transcription_language",
    "source_language",
    "original_language",
    "language_id_method",
    "word_count",
    "character_count",
    "date",
    "video_link",
    "license",
]

# Matches valid BCP-47 English codes found in the actual PleIAs/YouTube-Commons
# parquet schema (verified against cctube_0.parquet):
#   - bare 'en'                      — the dominant value (7,201/shard)
#   - 'en-GB', 'en-US', 'en-AU' etc — ISO 3166-1 alpha-2/3 region suffixes
#                                      (always uppercase in BCP-47)
#
# Does NOT match (intentionally rejected):
#   - 'en-fr', 'en-de', etc.  — YouTube's detection method tags some bilingual
#     videos (e.g. French/English) with a lowercase language suffix.  These have
#     original_language='fr'/'de' and are not English-primary content.
#   - 'en-GHEw9DUond8'        — garbage value (looks like a YouTube ID appended
#     by mistake); handled via the metadata fallback in _is_english() below.
#
# Note: 'source_language' is referenced in the HuggingFace README but does NOT
# exist in the actual parquet schema.  It is absent from _GATE_COLS and from
# _is_english() intentionally.
_EN_CODE_RE = re.compile(r"^en(-[A-Z]{2,3})?$")


# ---------------------------------------------------------------------------
# Language check
# ---------------------------------------------------------------------------

def _is_english(record: dict) -> bool:
    """Return True if the transcript is English.

    Three acceptance paths, tried in order:

    1. transcription_language matches a valid BCP-47 English code
       (en, en-GB, en-US, en-AU, …).  This is the primary path — 99%+ of
       English records in the dataset have transcription_language='en'.

    2. transcription_language is completely absent, AND source_language='en'.
       source_language captures the audio language before any translation step.
       Present in the official dataset schema but absent from some cctube_*
       shards; record.get() returns None safely when the column was not loaded.
       This path is ONLY active when transcription_language is absent — not when
       it is a non-English code like 'fr'.

    3. transcription_language is absent OR a non-standard value, AND YouTube's
       own metadata declared the source as English (original_language='en',
       language_id_method='metadata').  This catches the rare 'en-GHEw9DUond8'
       style corruption where a valid English video got a mangled language code.
       The metadata fallback correctly discriminates between garbage codes
       (method='metadata', o_lang='en') and valid non-English codes like 'en-fr'
       (method='detection').

    Mixed-language detection codes ('en-fr', 'en-de') are intentionally
    rejected: they originate from YouTube's algorithmic detection on bilingual
    videos whose primary language is not English (original_language='fr', etc.).
    """
    t_lang = (record.get("transcription_language") or "").strip()
    s_lang = (record.get("source_language") or "").strip()
    o_lang = (record.get("original_language") or "").strip()
    method = (record.get("language_id_method") or "").strip()

    # Path 1: clean BCP-47 English code.
    if _EN_CODE_RE.match(t_lang):
        return True

    if not t_lang:
        # transcription_language is absent — use audio-track and metadata signals.
        # Path 2: source_language (audio language before any translation).
        if s_lang == "en":
            return True
        # Path 3: YouTube metadata declares the source video as English.
        if method == "metadata" and o_lang == "en":
            return True
    else:
        # transcription_language is non-empty but failed the BCP-47 regex.
        # Could be a valid non-English code ('fr', 'de', 'en-fr') or a garbage
        # value ('en-GHEw9DUond8').  source_language is not consulted here
        # because it cannot distinguish between the two cases.  Only the metadata
        # fallback can: garbage English codes have method='metadata' and
        # o_lang='en'; valid non-English codes have method='detection'.
        if method == "metadata" and o_lang == "en":
            return True

    return False


# ---------------------------------------------------------------------------
# Gate function
# ---------------------------------------------------------------------------

def _passes_gate(record: dict, video_keep_set: frozenset[str] | None) -> bool:
    """Return True if this record should pass through to text fetching.

    Three checks are applied in order:

    1. Language gate  : drop non-English transcripts
    2. Word count gate: drop very short transcripts (< MIN_WORD_COUNT words)
    3. Classifier gate: drop video_ids not in the keep set

    video_keep_set=None means "no classifier cache was loaded".  In that case
    the classifier gate is skipped and any English record with enough words
    passes.  This is intentionally lenient — it lets the pipeline run before
    the classifier cache is built, producing a broader candidate set.
    """
    # Gate 1 — language
    if not _is_english(record):
        return False

    # Gate 2 — word count
    if (record.get("word_count") or 0) < MIN_WORD_COUNT:
        return False

    # Gate 3 — classifier cache (skipped if no cache was provided)
    if video_keep_set is not None:
        return record.get("video_id") in video_keep_set

    return True


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class YouTubeTranscriptsConnector:
    source_id = "youtube-transcripts"

    def iter_records(
        self,
        path: Path,
        *,
        filter_security: bool = True,
        cache_path: Path | None = None,
    ) -> Iterator[dict]:
        """Yield one record per transcript from cctube_*.parquet shards.

        Parameters
        ----------
        path:
            Directory containing cctube_*.parquet files.
        filter_security:
            When True (default), apply the language + word count gates and,
            if cache_path is provided, the classifier cache gate.
            When False, yield every record in every shard unfiltered — useful
            for inspection, test fixtures, and building the channel CSV.
        cache_path:
            Path to the merged classifier cache JSONL file produced by
            scripts/merge_classifier_caches.py.  When provided and
            filter_security=True, only records whose video_id appears in the
            cache with should_keep=True pass gate 3.
            When None and filter_security=True, gate 3 is skipped (language
            and word count gates still apply).

        Two-pass column projection per row group
        ----------------------------------------
        Pass 1: Read only _GATE_COLS (no transcript text).  Evaluate gates for
                every row in the row group.
        Pass 2: Read the "text" column only for the row group if at least one
                row passed all gates.  Then slice out only the passing rows.

        This avoids materialising gigabytes of transcript text for records that
        will be dropped by the language or classifier gate.
        """
        # Build the keep set once before iterating shards.
        # None means "gate 3 is inactive — pass everything that meets gates 1 and 2."
        # A frozenset (even empty) means gate 3 is active; only listed video_ids pass.
        # Gate 3 is only activated when the cache file actually exists on disk.
        # If cache_path is given but the file is missing (not yet built), gate 3
        # stays inactive rather than rejecting everything.
        video_keep_set: frozenset[str] | None = None
        if filter_security and cache_path is not None and cache_path.exists():
            video_keep_set = load_keep_set(cache_path)

        for parquet_file in sorted(path.glob("cctube_*.parquet")):
            try:
                pf = pq.ParquetFile(parquet_file)
            except Exception:
                continue

            # Intersect the requested columns with what this file actually has.
            # cctube_* and train-* shards differ slightly (e.g. cctube_0 lacks
            # source_language; train-* lacks language_id_method).  Reading only
            # available columns avoids ArrowInvalid errors on missing fields.
            available_cols = [
                c for c in _GATE_COLS
                if c in set(pf.schema_arrow.names)
            ]

            for rg in range(pf.metadata.num_row_groups):
                # Pass 1: cheap metadata columns only.
                try:
                    meta_table = pf.read_row_group(rg, columns=available_cols)
                except Exception:
                    continue

                n = meta_table.num_rows
                meta_cols = {
                    name: meta_table.column(name).to_pylist()
                    for name in meta_table.schema.names
                }
                meta_records = [
                    {col: meta_cols[col][i] for col in meta_cols}
                    for i in range(n)
                ]

                if filter_security:
                    passing_indices = [
                        i for i, rec in enumerate(meta_records)
                        if _passes_gate(rec, video_keep_set)
                    ]
                else:
                    passing_indices = list(range(n))

                # Skip the expensive text read entirely if no rows passed.
                if not passing_indices:
                    continue

                # Pass 2: fetch transcript text only for this row group, then
                # attach it to the records that made it through the gates.
                try:
                    text_col = (
                        pf.read_row_group(rg, columns=["text"])
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
        """Convert a raw record dict into a normalized TranscriptData object."""
        video_id = record.get("video_id") or ""
        text = record.get("text") or ""
        title = record.get("title") or None

        # Prefer transcription_language (language of the text column), then
        # source_language (audio-track language before translation), then
        # original_language (as declared by the video uploader/YouTube metadata).
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
            source_url=(
                record.get("video_link")
                or (f"https://www.youtube.com/watch?v={video_id}" if video_id else None)
            ),
            license=license_str,
            video_id=video_id or None,
            channel=record.get("channel") or None,
            channel_id=record.get("channel_id") or None,
            language=lang,
            word_count=record.get("word_count"),
            character_count=record.get("character_count"),
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
