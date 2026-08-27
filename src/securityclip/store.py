"""Query and virtual filesystem service for Security Scope."""

from __future__ import annotations

import json
import os
import re
import secrets
import shlex
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from securityclip.config import DEFAULT_INDEX_DIR
from securityclip.indexer import INDEX_DB_NAME
from securityclip.model import IndexedDocument
from securityclip.paths import normalize_virtual_path


DEFAULT_PREVIEW_LINES = 100
DEFAULT_LIMIT = 100

# Canonical corpus roots (document types). Used to fan out per-root searches.
ROOTS = ("/papers", "/qa", "/vulns", "/rules", "/logs", "/knowledge", "/transcripts", "/web")


@dataclass(frozen=True)
class SearchResult:
    rank: int
    doc: IndexedDocument
    snippet: str
    score: float | None = None


@dataclass(frozen=True)
class GrepMatch:
    doc: IndexedDocument
    line_number: int
    line: str


class SecurityClipStore:
    def __init__(self, index_dir: Path = DEFAULT_INDEX_DIR):
        self.index_dir = Path(index_dir)
        self.db_path = self.index_dir / INDEX_DB_NAME
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Security Scope index not found at {self.db_path}; run `security-scope build` first"
            )
        self.con = sqlite3.connect(self.db_path)
        self.con.row_factory = sqlite3.Row

    def close(self) -> None:
        self.con.close()

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        roots: Sequence[str] | None = None,
        per_root_limit: int | None = None,
    ) -> tuple[str, list[SearchResult]]:
        """Full-text search.

        With ``per_root_limit`` set, returns up to that many documents *per corpus root*
        (one ranked query per in-scope root, merged by relevance) instead of a single
        global ``limit``. In-scope roots are ``roots`` when given, else all ``ROOTS``.
        The CLI leaves ``per_root_limit`` unset and keeps the global-``limit`` behavior.
        """
        fts_query = _to_fts_query(query)
        select = """
            SELECT d.*, snippet(documents_fts, 2, '[', ']', ' ... ', 20) AS snippet,
                   bm25(documents_fts) AS score
            FROM documents_fts
            JOIN documents d ON d.doc_id = documents_fts.doc_id
            WHERE documents_fts MATCH ?
            """
        if per_root_limit is None:
            sql = select
            params: list[object] = [fts_query]
            if roots:
                sql += f" AND d.root IN ({', '.join('?' for _ in roots)})"
                params.extend(roots)
            sql += " ORDER BY score LIMIT ?"
            params.append(limit)
            rows = list(self.con.execute(sql, params).fetchall())
        else:
            scope = list(roots) if roots else list(ROOTS)
            rows = []
            for root in scope:
                rows.extend(
                    self.con.execute(
                        select + " AND d.root = ? ORDER BY score LIMIT ?",
                        (fts_query, root, per_root_limit),
                    ).fetchall()
                )
            # Re-rank the merged per-root results by relevance (bm25: lower is better).
            rows.sort(key=lambda row: row["score"] if row["score"] is not None else 0.0)
        results = [
            SearchResult(rank=idx, doc=_doc_from_row(row, self.index_dir), snippet=row["snippet"] or "", score=row["score"])
            for idx, row in enumerate(rows, start=1)
        ]
        handle = self._save_handle("search", query, [result.doc.doc_id for result in results])
        return handle, results

    def cat(self, virtual_path: str, *, all_lines: bool = False, preview_lines: int = DEFAULT_PREVIEW_LINES) -> str:
        path = normalize_virtual_path(virtual_path)
        if path.endswith("/meta.json"):
            doc = self.doc_for_dir(path.removesuffix("/meta.json"))
            return json.dumps(doc.meta, indent=2, sort_keys=True) + "\n"
        if path.endswith("/content.lines"):
            doc = self.doc_for_dir(path.removesuffix("/content.lines"))
            if all_lines:
                return doc.content_file(self.index_dir).read_text(encoding="utf-8")
            return self._read_head(doc, preview_lines, include_truncation_note=True)
        if path.endswith("/rule.yml"):
            doc = self.doc_for_dir(path.removesuffix("/rule.yml"))
            rule_path = doc.content_file(self.index_dir).with_name("rule.yml")
            if not rule_path.exists():
                raise FileNotFoundError(path)
            return rule_path.read_text(encoding="utf-8")
        return self.ls(path)

    def head(self, count: int, virtual_path: str) -> str:
        path = normalize_virtual_path(virtual_path)
        if not path.endswith("/content.lines"):
            return "\n".join(self.cat(path).splitlines()[:count]) + "\n"
        doc = self.doc_for_dir(path.removesuffix("/content.lines"))
        return self._read_head(doc, count, include_truncation_note=False)

    def grep(
        self,
        pattern: str,
        virtual_path: str | None = None,
        *,
        from_handle: str | None = None,
        ignore_case: bool = False,
        limit: int = DEFAULT_LIMIT,
        roots: Sequence[str] | None = None,
    ) -> tuple[str | None, list[GrepMatch]]:
        regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        docs: Iterable[IndexedDocument] = self._docs_for_grep_scope(pattern, virtual_path, from_handle=from_handle)
        if roots:
            allowed = set(roots)
            docs = (doc for doc in docs if doc.root in allowed)
        matches: list[GrepMatch] = []
        matched_doc_ids: list[str] = []
        seen_doc_ids: set[str] = set()
        for doc in docs:
            for line_number, line in _iter_line_file(doc.content_file(self.index_dir)):
                if regex.search(line):
                    matches.append(GrepMatch(doc=doc, line_number=line_number, line=line.rstrip("\n")))
                    if doc.doc_id not in seen_doc_ids:
                        matched_doc_ids.append(doc.doc_id)
                        seen_doc_ids.add(doc.doc_id)
                    if len(matches) >= limit:
                        handle = self._save_handle("grep", pattern, matched_doc_ids) if matched_doc_ids else None
                        return handle, matches
        handle = self._save_handle("grep", pattern, matched_doc_ids) if matched_doc_ids else None
        return handle, matches

    def ls(self, virtual_path: str = "/", *, limit: int = 200) -> str:
        path = normalize_virtual_path(virtual_path)
        doc = self._doc_for_dir_optional(path)
        if doc is not None:
            names = ["meta.json", "content.lines"]
            if doc.content_file(self.index_dir).with_name("rule.yml").exists():
                names.append("rule.yml")
            return "\n".join(names) + "\n"
        if path == "/":
            return "\n".join(["papers/", "qa/", "vulns/", "knowledge/", "rules/", "logs/", "transcripts/", "web/"]) + "\n"
        prefix = path.rstrip("/") + "/"
        depth = prefix.count("/")
        rows = self.con.execute(
            "SELECT virtual_dir FROM documents WHERE virtual_dir LIKE ? ORDER BY virtual_dir LIMIT ?",
            (prefix + "%", limit),
        ).fetchall()
        children: list[str] = []
        seen: set[str] = set()
        for row in rows:
            remainder = row["virtual_dir"][len(prefix):]
            child = remainder.split("/", 1)[0]
            if child not in seen:
                suffix = "/" if "/" in remainder else "/"
                children.append(child + suffix)
                seen.add(child)
        if not children and depth > 1:
            raise FileNotFoundError(path)
        return "\n".join(children) + ("\n" if children else "")

    def map(self, from_handle: str, prompt: str, *, limit: int | None = None) -> str:
        command = os.environ.get("SECURITYCLIP_MAP_COMMAND")
        if not command:
            raise RuntimeError("SECURITYCLIP_MAP_COMMAND is not set; map is optional/provider-configured")
        docs = self.docs_for_handle(from_handle)
        if limit is not None:
            docs = docs[:limit]
        output_parts: list[str] = []
        for doc in docs:
            content = self._read_head(doc, DEFAULT_PREVIEW_LINES, include_truncation_note=True)
            payload = f"Path: {doc.content_path}\nTitle: {doc.title or ''}\nPrompt: {prompt}\n\n{content}"
            proc = subprocess.run(
                shlex.split(command),
                input=payload,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip() or f"map command exited {proc.returncode}")
            output_parts.append(f"## {doc.content_path}\n\n{proc.stdout.strip()}\n")
        return "\n".join(output_parts)

    def docs_for_handle(self, handle_id: str) -> list[IndexedDocument]:
        rows = self.con.execute(
            """
            SELECT d.*
            FROM handle_items h
            JOIN documents d ON d.doc_id = h.doc_id
            WHERE h.handle_id = ?
            ORDER BY h.rank
            """,
            (handle_id,),
        ).fetchall()
        if not rows:
            raise KeyError(f"unknown or empty handle: {handle_id}")
        return [_doc_from_row(row, self.index_dir) for row in rows]

    def doc_for_dir(self, virtual_dir: str) -> IndexedDocument:
        doc = self._doc_for_dir_optional(virtual_dir)
        if doc is None:
            raise FileNotFoundError(virtual_dir)
        return doc

    def _doc_for_dir_optional(self, virtual_dir: str) -> IndexedDocument | None:
        path = normalize_virtual_path(virtual_dir)
        row = self.con.execute("SELECT * FROM documents WHERE virtual_dir = ?", (path,)).fetchone()
        return _doc_from_row(row, self.index_dir) if row else None

    def _read_head(self, doc: IndexedDocument, count: int, *, include_truncation_note: bool) -> str:
        lines: list[str] = []
        with doc.content_file(self.index_dir).open("r", encoding="utf-8") as f:
            for _, line in zip(range(count), f):
                lines.append(line.rstrip("\n"))
        if include_truncation_note and doc.line_count > count:
            lines.append(f"... ({doc.line_count - count} more lines; use `head -N` or grep to inspect)")
        return "\n".join(lines) + ("\n" if lines else "")

    def _docs_for_scope(self, virtual_path: str | None, *, from_handle: str | None) -> Iterable[IndexedDocument]:
        if from_handle:
            yield from self.docs_for_handle(from_handle)
            return
        path = normalize_virtual_path(virtual_path or "/")
        if path.endswith("/content.lines"):
            yield self.doc_for_dir(path.removesuffix("/content.lines"))
            return
        if path.endswith("/meta.json"):
            yield self.doc_for_dir(path.removesuffix("/meta.json"))
            return
        doc = self._doc_for_dir_optional(path)
        if doc is not None:
            yield doc
            return
        prefix = path.rstrip("/") + "/"
        rows = self.con.execute(
            "SELECT * FROM documents WHERE virtual_dir LIKE ? ORDER BY virtual_dir",
            (prefix + "%",),
        ).fetchall()
        for row in rows:
            yield _doc_from_row(row, self.index_dir)

    def _docs_for_grep_scope(
        self,
        pattern: str,
        virtual_path: str | None,
        *,
        from_handle: str | None,
    ) -> Iterable[IndexedDocument]:
        if from_handle:
            yield from self.docs_for_handle(from_handle)
            return

        path = normalize_virtual_path(virtual_path or "/")
        if path.endswith("/content.lines"):
            yield self.doc_for_dir(path.removesuffix("/content.lines"))
            return
        if path.endswith("/meta.json"):
            yield self.doc_for_dir(path.removesuffix("/meta.json"))
            return
        doc = self._doc_for_dir_optional(path)
        if doc is not None:
            yield doc
            return

        fts_query = _simple_fts_query_for_regex(pattern)
        if not fts_query:
            yield from self._docs_for_scope(path, from_handle=None)
            return

        prefix = path.rstrip("/") + "/"
        try:
            rows = self.con.execute(
                """
                SELECT d.*
                FROM documents_fts
                JOIN documents d ON d.doc_id = documents_fts.doc_id
                WHERE documents_fts MATCH ?
                  AND d.virtual_dir LIKE ?
                ORDER BY bm25(documents_fts)
                """,
                (fts_query, prefix + "%"),
            )
            for row in rows:
                yield _doc_from_row(row, self.index_dir)
        except sqlite3.OperationalError:
            yield from self._docs_for_scope(path, from_handle=None)

    def _save_handle(self, command: str, query: str, doc_ids: Sequence[str]) -> str:
        handle = "s_" + secrets.token_hex(4)
        self.con.execute(
            "INSERT INTO handles (handle_id, created_at, command, query) VALUES (?, ?, ?, ?)",
            (handle, datetime.now(timezone.utc).isoformat(), command, query),
        )
        self.con.executemany(
            "INSERT INTO handle_items (handle_id, rank, doc_id) VALUES (?, ?, ?)",
            [(handle, idx, doc_id) for idx, doc_id in enumerate(doc_ids, start=1)],
        )
        self.con.commit()
        return handle


def _doc_from_row(row: sqlite3.Row, index_dir: Path) -> IndexedDocument:
    meta = json.loads(row["meta_json"])
    return IndexedDocument(
        doc_id=row["doc_id"],
        virtual_dir=row["virtual_dir"],
        root=row["root"],
        source_id=row["source_id"],
        source_record_id=row["source_record_id"] or "",
        record_id=row["record_id"] or "",
        title=row["title"],
        source_url=row["source_url"],
        license=row["license"],
        content_length=row["content_length"],
        content_hash=row["content_hash"],
        meta=meta,
        content_relpath=row["content_relpath"],
        line_count=row["line_count"],
    )


def _to_fts_query(query: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9_+#.-]+", query)
    cleaned = [token.strip(".-") for token in tokens if token.strip(".-")]
    if not cleaned:
        return json.dumps(query)
    return " AND ".join(json.dumps(token) for token in cleaned)


def _simple_fts_query_for_regex(pattern: str) -> str | None:
    if re.search(r"[\\.^$*+?{}\[\]|()]", pattern):
        return None
    tokens = re.findall(r"[A-Za-z0-9_+#-]+", pattern)
    cleaned = [token.strip("-") for token in tokens if len(token.strip("-")) >= 2]
    if not cleaned:
        return None
    return " AND ".join(json.dumps(token) for token in cleaned)


def _iter_line_file(path: Path) -> Iterable[tuple[int, str]]:
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            match = re.match(r"L(\d+):\s?(.*)", raw)
            if match:
                yield int(match.group(1)), match.group(2)
            else:
                yield 0, raw
