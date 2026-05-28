"""FineWeb connector — schema and record normalization."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from ingest.connectors.base import NormalizedData
from ingest.connectors.fineweb.dsir import DSIRScorer, DEFAULT_CLIP
from ingest.utils import compute_content_hash, compute_token_count

# ---------------------------------------------------------------------------
# Quality gate defaults
# ---------------------------------------------------------------------------

MIN_TOKEN_COUNT: int = 40
MIN_LANGUAGE_SCORE: float = 0.65

# DSIR routing defaults — match ingest_fineweb.py
HIGH_THRESHOLD: float = 0.0
MEDIUM_THRESHOLD: float = -0.3


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class FineWebData(NormalizedData):
    """FineWeb / Common Crawl web document, confidence-stratified via DSIR scoring."""

    language_score: float | None = None
    dump: str | None = None           # CommonCrawl dump name (e.g. CC-MAIN-2025-26)
    confidence: str | None = None     # "high" | "medium" | "background"
    dsir_score: float | None = None   # per-n-gram log importance weight (DSIRScorer)


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class FineWebConnector:
    """
    Connector for FineWeb (HuggingFaceFW/fineweb) parquet shards.

    Expects shards already downloaded locally or already-scored output from
    the Datatrove pipeline (scripts/ingest_fineweb.py).

    If a DSIRScorer is supplied via scorer=, documents are scored on the fly
    (useful for re-scoring raw shards against an updated scorer). If scorer=None,
    documents are yielded without scoring and confidence will be None.

    For streaming directly from HuggingFace on Marlowe, use ingest_fineweb.py
    with the Datatrove backend — the scorer is loaded there from scorer.pkl.
    """

    source_id = "fineweb"

    def iter_records(
        self,
        path: Path,
        *,
        files: list[Path] | None = None,
        scorer: DSIRScorer | None = None,
        min_token_count: int = MIN_TOKEN_COUNT,
        min_language_score: float = MIN_LANGUAGE_SCORE,
        high_threshold: float = HIGH_THRESHOLD,
        medium_threshold: float = MEDIUM_THRESHOLD,
        confidence_filter: str | None = None,
    ) -> Iterator[dict]:
        """
        Yield FineWeb records, optionally DSIR-scored.

        Args:
            path:               directory containing *.parquet shards
            files:              explicit shard list; if None, globs path/*.parquet
            scorer:             fitted DSIRScorer; if None, scoring is skipped
            min_token_count:    drop documents with fewer tokens
            min_language_score: drop documents below this fastText confidence
            high_threshold:     dsir_score >= this → "high"
            medium_threshold:   dsir_score >= this → "medium"
            confidence_filter:  if set, only yield records matching this tier
                                (requires scorer to be provided)
        """
        shard_list = sorted(files) if files is not None else sorted(path.glob("*.parquet"))

        for shard in shard_list:
            try:
                table = pq.ParquetFile(str(shard)).read()
            except Exception:
                continue

            for record in table.to_pylist():
                if (record.get("token_count") or 0) < min_token_count:
                    continue
                if (record.get("language_score") or 0.0) < min_language_score:
                    continue

                if scorer is not None:
                    text = record.get("text") or ""
                    score = scorer.score(text)
                    if score >= high_threshold:
                        confidence = "high"
                    elif score >= medium_threshold:
                        confidence = "medium"
                    else:
                        confidence = "background"
                    record["_dsir_score"] = score
                    record["_confidence"] = confidence
                else:
                    record["_dsir_score"] = None
                    record["_confidence"] = None

                if confidence_filter and record["_confidence"] != confidence_filter:
                    continue

                yield record

    def normalize(self, record: dict) -> FineWebData:
        doc_id = record.get("id") or ""
        text = record.get("text") or ""
        url = record.get("url") or ""

        published_at = None
        crawl_date = record.get("date")
        if crawl_date:
            try:
                published_at = datetime.strptime(crawl_date[:10], "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                pass

        return FineWebData(
            source_id=self.source_id,
            source_record_id=doc_id,
            record_id=f"{self.source_id}:{doc_id}",
            content=text,
            content_length=compute_token_count(text),
            content_hash=compute_content_hash(text),
            ingested_at=datetime.now(timezone.utc),
            published_at=published_at,
            source_url=url,
            license="https://commoncrawl.org/terms-of-use",
            raw=None,
            language_score=record.get("language_score"),
            dump=record.get("dump"),
            confidence=record.get("_confidence"),
            dsir_score=record.get("_dsir_score"),
        )
