"""Virtual path helpers."""

from __future__ import annotations

import re
from typing import Any


_SEGMENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def normalize_virtual_path(path: str) -> str:
    if not path:
        return "/"
    normalized = "/" + path.strip()
    normalized = re.sub(r"/+", "/", normalized)
    if len(normalized) > 1:
        normalized = normalized.rstrip("/")
    return normalized


def safe_segment(value: Any, *, fallback: str = "doc") -> str:
    text = str(value or fallback).strip()
    text = text.replace("/", "_").replace(":", "_")
    text = _SEGMENT_RE.sub("_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or fallback


def root_for_source(source_id: str) -> tuple[str, str | None]:
    if source_id == "arxiv":
        return "/papers", None
    if source_id in {"nvd", "cisa-kev"}:
        return "/vulns", source_id
    if source_id in {"mitre-attack", "mitre-cwe", "capec"}:
        return "/knowledge", source_id
    if source_id == "sigma":
        return "/rules", "sigma"
    if source_id == "cloudtrail-flaws":
        return "/logs", "cloudtrail-flaws"
    if source_id == "youtube-transcripts":
        return "/transcripts", "youtube"
    if source_id == "fineweb-security":
        return "/web", "fineweb-security"
    if source_id == "stackoverflow" or source_id.startswith("stackexchange-") or source_id.startswith("reddit-"):
        return "/qa", source_id
    return "/docs", source_id


def base_doc_name(row: dict[str, Any]) -> str:
    source_id = str(row.get("source_id") or "unknown")
    if source_id == "arxiv":
        return "arx_" + safe_segment(row.get("arxiv_id") or row.get("source_record_id"))
    if source_id in {"nvd", "cisa-kev"}:
        return safe_segment(row.get("cve_id") or row.get("source_record_id"))
    if source_id == "sigma":
        return safe_segment(row.get("rule_id") or row.get("source_record_id"))
    if source_id == "cloudtrail-flaws":
        return safe_segment(row.get("source_record_id"))
    if source_id == "youtube-transcripts":
        return safe_segment(row.get("video_id") or row.get("source_record_id"))
    return safe_segment(row.get("source_record_id") or row.get("record_id"))


def virtual_dir_for_row(row: dict[str, Any], *, suffix: int | None = None) -> str:
    root, source_segment = root_for_source(str(row.get("source_id") or "unknown"))
    name = base_doc_name(row)
    if suffix is not None:
        name = f"{name}_{suffix}"
    parts = [root]
    if source_segment:
        parts.append(safe_segment(source_segment))
    parts.append(name)
    return "/".join(parts)
