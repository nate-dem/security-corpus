from typing import Protocol, Iterator, Literal
from pydantic import BaseModel, Field
from pathlib import Path
from datetime import datetime


class NormalizedData(BaseModel):
    # -- REQUIRED --

    # identification
    source_id: str
    source_record_id: str
    record_id: str  # convention: f"{source_id}:{source_record_id}"

    # primary content
    content: str
    title: str | None = None
    content_length: int | None = None       # tokens, cl100k_base
    content_hash: str | None = None         # SHA-256 of content for dedup

    # metadata
    ingested_at: datetime                    # always UTC
    published_at: datetime | None = None
    source_url: str | None = None
    license: str | None = None

    # raw source preservation (optional, only for small structured sources)
    raw: dict | None = None


class VulnerabilityData(NormalizedData):
    """NVD CVEs, CISA KEV, GitHub Advisory DB."""
    cve_id: str | None = None
    severity: str | None = None
    cvss_score: float | None = None
    cwe_ids: list[str] = Field(default_factory=list)
    exploited_in_wild: bool | None = None


class MitreData(NormalizedData):
    """MITRE ATT&CK, CWE, CAPEC."""
    framework: str | None = None              # "attack" | "cwe" | "capec"
    category_id: str | None = None            # T1055, CWE-79, CAPEC-100


class DetectionRuleData(NormalizedData):
    """Sigma, YARA."""
    rule_id: str | None = None
    rule_format: str | None = None            # "sigma" | "yara"
    rule_level: str | None = None             # severity/confidence
    rule_source: str | None = None            # raw rule text


class QAThreadData(NormalizedData):
    """Stack Exchange sites and similar Q&A formats."""
    score: int | None = None
    answer_count: int | None = None
    has_accepted_answer: bool | None = None
    closed: bool | None = None
    tags: list[str] = Field(default_factory=list)


class TranscriptData(NormalizedData):
    """YouTube transcripts and similar spoken-word video corpora."""
    video_id: str | None = None
    channel: str | None = None
    channel_id: str | None = None
    language: str | None = None          # transcription_language (BCP-47 code)
    word_count: int | None = None


class CloudTrailSessionData(NormalizedData):
    """CloudTrail log sessions (grouped by source IP + time gap)."""
    event_count: int | None = None
    session_duration_seconds: int | None = None
    source_ip: str | None = None
    principals: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    aws_services: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    has_errors: bool | None = None


class AcademicPaperData(NormalizedData):
    """arXiv academic papers."""
    arxiv_id: str | None = None
    source_format: Literal["latex", "pdf"] | None = None
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    categories: list[str] = Field(default_factory=list)
    primary_category: str | None = None
    doi: str | None = None
    journal_ref: str | None = None


class Connector(Protocol):
    source_id: str

    def iter_records(self, path: Path) -> Iterator[dict]: ...
    def normalize(self, record: dict) -> NormalizedData: ...
