"""Execute validated Security Scope web operations through SecurityClipStore."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from securityclip.model import IndexedDocument
from securityclip.store import GrepMatch, SearchResult, SecurityClipStore
from securityclip.web.operations import ValidatedOperation, render_command
from securityclip.web.settings import WebSettings


CITATION_RE = re.compile(r"(?P<path>/(?:papers|qa|vulns|rules|logs|knowledge|transcripts|web)/\S+?/content\.lines):L(?P<line>\d+):?(?P<snippet>.*)")


@dataclass
class OperationOutput:
    operation: dict[str, Any]
    command: str
    ok: bool
    output_text: str
    data: dict[str, Any] = field(default_factory=dict)
    citations: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    latency_ms: int = 0
    truncated: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "command": self.command,
            "ok": self.ok,
            "output_text": self.output_text,
            "data": self.data,
            "citations": self.citations,
            "sources": self.sources,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "truncated": self.truncated,
        }


class OperationExecutor:
    def __init__(self, index_dir: Path, settings: WebSettings, *, roots: list[str] | None = None):
        self.index_dir = Path(index_dir)
        self.settings = settings
        self.roots = list(roots) if roots else None

    def execute_many(self, operations: list[ValidatedOperation]) -> list[OperationOutput]:
        outputs: list[OperationOutput] = []
        for operation in operations:
            outputs.append(self.execute(operation))
        return outputs

    def execute(self, operation: ValidatedOperation) -> OperationOutput:
        command = render_command(operation)
        started = time.monotonic()
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(self._execute_unbounded, operation, command)
        try:
            output = future.result(timeout=self.settings.command_timeout_seconds)
        except TimeoutError:
            future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            return OperationOutput(
                operation=operation.to_json(),
                command=command,
                ok=False,
                output_text="",
                error=f"operation timed out after {self.settings.command_timeout_seconds:g}s",
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        finally:
            if future.done():
                pool.shutdown(wait=True, cancel_futures=True)
        output.latency_ms = int((time.monotonic() - started) * 1000)
        if len(output.output_text) > self.settings.max_output_chars:
            output.output_text = output.output_text[: self.settings.max_output_chars] + "\n... [truncated]"
            output.truncated = True
        return output

    def _execute_unbounded(self, operation: ValidatedOperation, command: str) -> OperationOutput:
        try:
            with StoreSession(self.index_dir) as store:
                if operation.tool == "search":
                    return _execute_search(store, operation, command, roots=self.roots)
                if operation.tool == "grep":
                    return _execute_grep(store, operation, command, roots=self.roots)
                if operation.tool == "grep_from":
                    return _execute_grep_from(store, operation, command, roots=self.roots)
                if operation.tool == "cat":
                    return _execute_cat(store, operation, command)
                if operation.tool == "head":
                    return _execute_head(store, operation, command)
                if operation.tool == "ls":
                    return _execute_ls(store, operation, command, roots=self.roots)
        except Exception as exc:  # noqa: BLE001 - surfaced to UI as operation error
            return OperationOutput(
                operation=operation.to_json(),
                command=command,
                ok=False,
                output_text="",
                error=str(exc),
            )
        raise AssertionError(f"unhandled operation: {operation.tool}")


class StoreSession:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.store: SecurityClipStore | None = None

    def __enter__(self) -> SecurityClipStore:
        self.store = SecurityClipStore(self.index_dir)
        return self.store

    def __exit__(self, *args: object) -> None:
        if self.store is not None:
            self.store.close()


def _execute_search(
    store: SecurityClipStore, operation: ValidatedOperation, command: str, *, roots: list[str] | None = None
) -> OperationOutput:
    # The web UI shows results grouped by document type, so cap per root: up to `limit`
    # documents from each in-scope root rather than `limit` total across all of them.
    limit = operation.params["limit"]
    handle, results = store.search(operation.params["query"], limit=limit, roots=roots, per_root_limit=limit)
    data = {
        "handle": handle,
        "results": [_search_result_json(result) for result in results],
    }
    output = _search_text(handle, results)
    return OperationOutput(
        operation=operation.to_json(),
        command=command,
        ok=True,
        output_text=output,
        data=data,
        sources=[_doc_source(result.doc) for result in results],
    )


def _execute_grep(
    store: SecurityClipStore, operation: ValidatedOperation, command: str, *, roots: list[str] | None = None
) -> OperationOutput:
    handle, matches = store.grep(
        operation.params["pattern"],
        operation.params["path"],
        ignore_case=operation.params["ignore_case"],
        limit=operation.params["limit"],
        roots=roots,
    )
    return _grep_output(operation, command, handle, matches)


def _execute_grep_from(
    store: SecurityClipStore, operation: ValidatedOperation, command: str, *, roots: list[str] | None = None
) -> OperationOutput:
    handle, matches = store.grep(
        operation.params["pattern"],
        from_handle=operation.params["handle"],
        ignore_case=operation.params["ignore_case"],
        limit=operation.params["limit"],
        roots=roots,
    )
    return _grep_output(operation, command, handle, matches)


def _execute_cat(store: SecurityClipStore, operation: ValidatedOperation, command: str) -> OperationOutput:
    text = store.cat(operation.params["path"])
    data: dict[str, Any] = {"path": operation.params["path"], "text": text}
    if operation.params["path"].endswith("/meta.json"):
        try:
            data["meta"] = json.loads(text)
        except json.JSONDecodeError:
            pass
    return OperationOutput(operation=operation.to_json(), command=command, ok=True, output_text=text, data=data)


def _execute_head(store: SecurityClipStore, operation: ValidatedOperation, command: str) -> OperationOutput:
    text = store.head(operation.params["count"], operation.params["path"])
    citations = citations_from_head(operation.params["path"], text)
    return OperationOutput(
        operation=operation.to_json(),
        command=command,
        ok=True,
        output_text=text,
        data={"path": operation.params["path"], "count": operation.params["count"], "lines": citations},
        citations=citations,
    )


def _execute_ls(
    store: SecurityClipStore, operation: ValidatedOperation, command: str, *, roots: list[str] | None = None
) -> OperationOutput:
    text = store.ls(operation.params["path"], limit=operation.params["limit"])
    children = [line for line in text.splitlines() if line]
    if roots and operation.params["path"] == "/":
        allowed = {root.lstrip("/") + "/" for root in roots}
        children = [child for child in children if child in allowed]
        text = "\n".join(children) + ("\n" if children else "")
    return OperationOutput(
        operation=operation.to_json(),
        command=command,
        ok=True,
        output_text=text,
        data={"path": operation.params["path"], "children": children},
    )


def _grep_output(operation: ValidatedOperation, command: str, handle: str | None, matches: list[GrepMatch]) -> OperationOutput:
    lines = [f"{match.doc.content_path}:L{match.line_number}:{match.line}" for match in matches]
    if handle:
        lines.append(f"Search results: {handle}")
    citations = [_match_citation(match) for match in matches]
    return OperationOutput(
        operation=operation.to_json(),
        command=command,
        ok=True,
        output_text="\n".join(lines) + ("\n" if lines else ""),
        data={
            "handle": handle,
            "matches": [_grep_match_json(match) for match in matches],
        },
        citations=citations,
        sources=[_doc_source(match.doc) for match in matches],
    )


def citations_from_text(text: str) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = CITATION_RE.match(line)
        if match:
            citations.append(
                {
                    "path": match.group("path"),
                    "line_number": int(match.group("line")),
                    "citation": f"{match.group('path')}:L{match.group('line')}",
                    "snippet": match.group("snippet").strip(),
                }
            )
    return citations


def citations_from_head(path: str, text: str) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for raw in text.splitlines():
        match = re.match(r"L(\d+):\s?(.*)", raw)
        if match:
            citations.append(
                {
                    "path": path,
                    "line_number": int(match.group(1)),
                    "citation": f"{path}:L{match.group(1)}",
                    "snippet": match.group(2),
                }
            )
    return citations


def collect_citations(outputs: list[OperationOutput]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    citations: list[dict[str, Any]] = []
    for output in outputs:
        for citation in [*output.citations, *citations_from_text(output.output_text)]:
            key = citation.get("citation") or f"{citation.get('path')}:{citation.get('line_number')}"
            if key in seen:
                continue
            seen.add(key)
            citations.append(citation)
    return citations


def collect_sources(outputs: list[OperationOutput]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    sources: list[dict[str, Any]] = []
    for output in outputs:
        for source in output.sources:
            key = source["path"]
            if key in seen:
                continue
            seen.add(key)
            sources.append(source)
    return sources


def _search_text(handle: str, results: list[SearchResult]) -> str:
    lines = [f"Search results: {handle}"]
    for result in results:
        doc = result.doc
        lines.append(f"{result.rank}. {doc.content_path}")
        lines.append(f"   Title: {doc.title or doc.record_id}")
        lines.append(f"   Source: {doc.source_id} | Tokens: {doc.content_length or 'unknown'}")
        if result.snippet:
            lines.append(f"   Snippet: {' '.join(result.snippet.split())}")
    return "\n".join(lines) + "\n"


def _search_result_json(result: SearchResult) -> dict[str, Any]:
    doc = result.doc
    return {
        "rank": result.rank,
        "path": doc.content_path,
        "meta_path": doc.meta_path,
        "title": doc.title,
        "source_id": doc.source_id,
        "record_id": doc.record_id,
        "snippet": result.snippet,
        "score": result.score,
    }


def _grep_match_json(match: GrepMatch) -> dict[str, Any]:
    return {
        "path": match.doc.content_path,
        "line_number": match.line_number,
        "line": match.line,
        "citation": f"{match.doc.content_path}:L{match.line_number}",
        "source_id": match.doc.source_id,
        "title": match.doc.title,
    }


def _match_citation(match: GrepMatch) -> dict[str, Any]:
    return {
        "path": match.doc.content_path,
        "line_number": match.line_number,
        "citation": f"{match.doc.content_path}:L{match.line_number}",
        "snippet": match.line,
        "source_id": match.doc.source_id,
        "title": match.doc.title,
    }


def _doc_source(doc: IndexedDocument) -> dict[str, Any]:
    return {
        "path": doc.content_path,
        "meta_path": doc.meta_path,
        "virtual_dir": doc.virtual_dir,
        "source_id": doc.source_id,
        "title": doc.title or doc.record_id,
        "record_id": doc.record_id,
        "source_url": doc.source_url,
        "license": doc.license,
        "tokens": doc.content_length,
    }
