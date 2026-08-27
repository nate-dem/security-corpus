"""Shared data structures for the Security Scope index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IndexedDocument:
    doc_id: str
    virtual_dir: str
    root: str
    source_id: str
    source_record_id: str
    record_id: str
    title: str | None
    source_url: str | None
    license: str | None
    content_length: int | None
    content_hash: str | None
    meta: dict[str, Any]
    content_relpath: str
    line_count: int

    @property
    def meta_path(self) -> str:
        return f"{self.virtual_dir}/meta.json"

    @property
    def content_path(self) -> str:
        return f"{self.virtual_dir}/content.lines"

    def content_file(self, index_dir: Path) -> Path:
        return index_dir / self.content_relpath
