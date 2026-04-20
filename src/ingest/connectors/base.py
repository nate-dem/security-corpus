from typing import Protocol, Iterator
from pydantic import BaseModel, Field
from pathlib import Path
from datetime import datetime


class NormalizedData(BaseModel):
    # -- REQUIRED --

    # identification
    record_id: str
    source_id: str
    source_record_id: str

    # main data
    content: str
    
    # Preserved raw source record for re-normalization without re-ingestion.
    # Structured sources (NVD, MITRE CVE List, CWE) → inline dict with raw data.
    # Larger sources (Common Crawl, arXiv PDFs) → reference to source.
    raw: dict

    # metadata
    ingested_at: datetime

    # -- OPTIONAL --

    # data info
    # necessary for arXiv papers, web pages, etc but data such as NVD CVEs will have set to None
    title: str | None = None

    # security-specific fields
    severity: str | None = None
    cvss_score: float | None = None
    cwe_ids: list[str] = Field(default_factory=list)

    # metadata
    published_at: datetime | None = None
    modified_at: datetime | None = None
    source_url: str | None = None
    language: str | None = None


class Connector(Protocol):
    source_id: str

    def iter_records(self, path: Path) -> Iterator[dict]: ...
    def normalize(self, record: dict) -> NormalizedData: ... 