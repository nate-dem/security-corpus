"""Index builder for the Paperclip-style security corpus interface."""

from __future__ import annotations

import glob
import json
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb

from securityclip.config import DEFAULT_INDEX_DIR, FINAL_SOURCE_SPECS, SourceSpec
from securityclip.paths import root_for_source, safe_segment, virtual_dir_for_row


INDEX_DB_NAME = "securityclip.sqlite"
DEFAULT_FTS_MAX_CHARS = 500_000

_COLUMN_DEFAULTS: dict[str, str] = {
    "source_id": "NULL::VARCHAR",
    "source_record_id": "NULL::VARCHAR",
    "record_id": "NULL::VARCHAR",
    "content": "NULL::VARCHAR",
    "title": "NULL::VARCHAR",
    "content_length": "NULL::BIGINT",
    "content_hash": "NULL::VARCHAR",
    "ingested_at": "NULL::VARCHAR",
    "published_at": "NULL::VARCHAR",
    "source_url": "NULL::VARCHAR",
    "license": "NULL::VARCHAR",
    "raw": "NULL::VARCHAR",
    "arxiv_id": "NULL::VARCHAR",
    "source_format": "NULL::VARCHAR",
    "authors": "NULL::VARCHAR[]",
    "abstract": "NULL::VARCHAR",
    "categories": "NULL::VARCHAR[]",
    "primary_category": "NULL::VARCHAR",
    "doi": "NULL::VARCHAR",
    "journal_ref": "NULL::VARCHAR",
    "score": "NULL::BIGINT",
    "answer_count": "NULL::BIGINT",
    "has_accepted_answer": "NULL::BOOLEAN",
    "closed": "NULL::BOOLEAN",
    "tags": "NULL::VARCHAR[]",
    "cve_id": "NULL::VARCHAR",
    "severity": "NULL::VARCHAR",
    "cvss_score": "NULL::DOUBLE",
    "cwe_ids": "NULL::VARCHAR[]",
    "exploited_in_wild": "NULL::BOOLEAN",
    "framework": "NULL::VARCHAR",
    "category_id": "NULL::VARCHAR",
    "rule_id": "NULL::VARCHAR",
    "rule_format": "NULL::VARCHAR",
    "rule_level": "NULL::VARCHAR",
    "rule_source": "NULL::VARCHAR",
    "event_count": "NULL::BIGINT",
    "session_duration_seconds": "NULL::BIGINT",
    "source_ip": "NULL::VARCHAR",
    "principals": "NULL::VARCHAR[]",
    "actions": "NULL::VARCHAR[]",
    "aws_services": "NULL::VARCHAR[]",
    "regions": "NULL::VARCHAR[]",
    "has_errors": "NULL::BOOLEAN",
    "video_id": "NULL::VARCHAR",
    "channel": "NULL::VARCHAR",
    "channel_id": "NULL::VARCHAR",
    "language": "NULL::VARCHAR",
    "word_count": "NULL::BIGINT",
    "dsir_score": "NULL::DOUBLE",
    "qwen_security_relevance": "NULL::SMALLINT",
    "qwen_quality": "NULL::SMALLINT",
    "qwen_model_should_keep": "NULL::BOOLEAN",
    "qwen_should_keep": "NULL::BOOLEAN",
    "qwen_reason": "NULL::VARCHAR",
    "qwen_parse_status": "NULL::VARCHAR",
    "qwen_keep_policy": "NULL::VARCHAR",
    "qwen_keep_policy_passed": "NULL::BOOLEAN",
    "qwen_keep_policy_reason": "NULL::VARCHAR",
    "qwen_model": "NULL::VARCHAR",
    "qwen_prompt_version": "NULL::VARCHAR",
    "qwen_scored_at": "NULL::VARCHAR",
    "qwen_shard_id": "NULL::VARCHAR",
    "qwen_task": "NULL::VARCHAR",
    "qwen_input_kind": "NULL::VARCHAR",
}

_METADATA_COLUMNS = tuple(_COLUMN_DEFAULTS)
_EXCLUDED_META = {"content", "raw", "qwen_raw_response", "rule_source"}


@dataclass(frozen=True)
class BuildSummary:
    index_dir: Path
    documents: int
    skipped_missing_patterns: tuple[str, ...]


def build_index(
    index_dir: Path = DEFAULT_INDEX_DIR,
    *,
    source_specs: Sequence[SourceSpec] = FINAL_SOURCE_SPECS,
    overwrite: bool = False,
    fts_max_chars: int = DEFAULT_FTS_MAX_CHARS,
    batch_size: int = 500,
) -> BuildSummary:
    """Build a Security Scope index from the configured Parquet sources."""

    index_dir = Path(index_dir)
    if index_dir.exists() and overwrite:
        shutil.rmtree(index_dir)
    if index_dir.exists() and any(index_dir.iterdir()):
        raise FileExistsError(f"Index directory is not empty: {index_dir}")

    files_dir = index_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(index_dir / INDEX_DB_NAME)
    _init_sqlite(con)

    duck = duckdb.connect()
    seen_dirs: dict[str, int] = {}
    documents = 0
    skipped: list[str] = []

    for spec in source_specs:
        files = sorted(glob.glob(spec.pattern))
        if not files:
            skipped.append(spec.pattern)
            continue
        relation = _select_relation_sql(duck, files)
        for rows in _iter_rows(duck, relation, batch_size=batch_size):
            for row in rows:
                row = _normalize_row(row)
                content = (row.get("content") or "").strip()
                if not content:
                    continue
                base_dir = virtual_dir_for_row(row)
                suffix = seen_dirs.get(base_dir)
                if suffix is None:
                    seen_dirs[base_dir] = 1
                    virtual_dir = base_dir
                else:
                    virtual_dir = virtual_dir_for_row(row, suffix=suffix + 1)
                    seen_dirs[base_dir] = suffix + 1
                doc_id = virtual_dir.lstrip("/").replace("/", "__")
                rel_dir = Path(*virtual_dir.lstrip("/").split("/"))
                content_relpath = (Path("files") / rel_dir / "content.lines").as_posix()
                content_path = index_dir / content_relpath
                meta_path = index_dir / "files" / rel_dir / "meta.json"
                content_path.parent.mkdir(parents=True, exist_ok=True)
                meta = _build_meta(row, virtual_dir=virtual_dir, doc_id=doc_id)
                line_count = _write_content_lines(content_path, content)
                _write_json(meta_path, meta)
                if row.get("rule_source"):
                    (content_path.parent / "rule.yml").write_text(str(row["rule_source"]), encoding="utf-8")
                _insert_document(
                    con,
                    doc_id=doc_id,
                    virtual_dir=virtual_dir,
                    row=row,
                    meta=meta,
                    content_relpath=content_relpath,
                    line_count=line_count,
                    fts_content=_fts_text(row, content, max_chars=fts_max_chars),
                )
                documents += 1

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "documents": documents,
        "source_specs": [spec.__dict__ for spec in source_specs],
        "skipped_missing_patterns": skipped,
        "fts_max_chars": fts_max_chars,
        "format": "securityclip-index-v1",
    }
    _write_json(index_dir / "manifest.json", manifest)
    con.commit()
    con.close()
    duck.close()
    return BuildSummary(index_dir=index_dir, documents=documents, skipped_missing_patterns=tuple(skipped))


def _init_sqlite(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        PRAGMA journal_mode = WAL;
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            virtual_dir TEXT NOT NULL UNIQUE,
            root TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_record_id TEXT,
            record_id TEXT,
            title TEXT,
            source_url TEXT,
            license TEXT,
            content_length INTEGER,
            content_hash TEXT,
            meta_json TEXT NOT NULL,
            content_relpath TEXT NOT NULL,
            line_count INTEGER NOT NULL
        );
        CREATE VIRTUAL TABLE documents_fts USING fts5(
            doc_id UNINDEXED,
            title,
            content,
            meta
        );
        CREATE TABLE handles (
            handle_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            command TEXT NOT NULL,
            query TEXT NOT NULL
        );
        CREATE TABLE handle_items (
            handle_id TEXT NOT NULL,
            rank INTEGER NOT NULL,
            doc_id TEXT NOT NULL,
            PRIMARY KEY (handle_id, rank)
        );
        """
    )


def _select_relation_sql(duck: duckdb.DuckDBPyConnection, files: Sequence[str]) -> str:
    file_list = ", ".join(_sql_string(path) for path in files)
    expr = f"read_parquet([{file_list}], union_by_name=true, hive_partitioning=false)"
    columns = {row[0] for row in duck.sql(f"DESCRIBE SELECT * FROM {expr}").fetchall()}
    select_exprs = []
    for column in _METADATA_COLUMNS:
        if column in columns:
            select_exprs.append(f"{_quote_ident(column)} AS {_quote_ident(column)}")
        else:
            select_exprs.append(f"{_COLUMN_DEFAULTS[column]} AS {_quote_ident(column)}")
    return f"SELECT {', '.join(select_exprs)} FROM {expr}"


def _iter_rows(
    duck: duckdb.DuckDBPyConnection,
    relation_sql: str,
    *,
    batch_size: int,
) -> Iterable[list[dict[str, Any]]]:
    reader = duck.sql(relation_sql).to_arrow_reader(batch_size=batch_size)
    for batch in reader:
        yield batch.to_pylist()


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in row.items():
        if hasattr(value, "as_py"):
            value = value.as_py()
        normalized[key] = value
    source_id = str(normalized.get("source_id") or "unknown")
    normalized["source_id"] = source_id
    normalized["source_record_id"] = str(normalized.get("source_record_id") or normalized.get("record_id") or "")
    normalized["record_id"] = str(normalized.get("record_id") or f"{source_id}:{normalized['source_record_id']}")
    return normalized


def _build_meta(row: dict[str, Any], *, virtual_dir: str, doc_id: str) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "doc_id": doc_id,
        "virtual_dir": virtual_dir,
        "meta_path": f"{virtual_dir}/meta.json",
        "content_path": f"{virtual_dir}/content.lines",
    }
    for key, value in row.items():
        if key in _EXCLUDED_META or value is None:
            continue
        meta[key] = _jsonable(value)
    return meta


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_content_lines(path: Path, content: str) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for count, line in enumerate(content.splitlines(), start=1):
            f.write(f"L{count}: {line}\n")
        if count == 0:
            f.write("L1: \n")
            return 1
    return count


def _fts_text(row: dict[str, Any], content: str, *, max_chars: int) -> str:
    if max_chars > 0 and len(content) > max_chars:
        content = content[:max_chars]
    meta_bits = [
        str(row.get(key) or "")
        for key in (
            "source_id",
            "source_record_id",
            "record_id",
            "arxiv_id",
            "cve_id",
            "category_id",
            "rule_id",
            "tags",
            "qwen_reason",
        )
    ]
    return "\n".join([*meta_bits, content])


def _insert_document(
    con: sqlite3.Connection,
    *,
    doc_id: str,
    virtual_dir: str,
    row: dict[str, Any],
    meta: dict[str, Any],
    content_relpath: str,
    line_count: int,
    fts_content: str,
) -> None:
    root, _ = root_for_source(row["source_id"])
    con.execute(
        """
        INSERT INTO documents (
            doc_id, virtual_dir, root, source_id, source_record_id, record_id,
            title, source_url, license, content_length, content_hash, meta_json,
            content_relpath, line_count
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc_id,
            virtual_dir,
            root,
            row["source_id"],
            row.get("source_record_id"),
            row.get("record_id"),
            row.get("title"),
            row.get("source_url"),
            row.get("license"),
            _safe_int(row.get("content_length")),
            row.get("content_hash"),
            json.dumps(meta, sort_keys=True),
            content_relpath,
            line_count,
        ),
    )
    con.execute(
        "INSERT INTO documents_fts (doc_id, title, content, meta) VALUES (?, ?, ?, ?)",
        (
            doc_id,
            row.get("title") or "",
            fts_content,
            " ".join(str(meta.get(key, "")) for key in ("source_id", "source_record_id", "record_id")),
        ),
    )


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
