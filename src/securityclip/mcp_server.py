"""Optional MCP server wrapper for Security Scope.

This module intentionally imports MCP lazily so the deterministic CLI works
without extra dependencies. Install the optional MCP dependency before running
``security-scope-mcp``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from securityclip.config import DEFAULT_INDEX_DIR, INDEX_ENV_VAR, default_index_dir
from securityclip.store import SecurityClipStore


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="security-scope-mcp")
    parser.add_argument(
        "--index",
        type=Path,
        default=default_index_dir(),
        help=f"Index directory (default: ${INDEX_ENV_VAR} if set, otherwise {DEFAULT_INDEX_DIR})",
    )
    args = parser.parse_args(argv)
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except ImportError as exc:
        raise SystemExit("security-scope-mcp requires the optional `mcp` package") from exc

    mcp = FastMCP("security-scope")

    @mcp.tool()
    def search(query: str, limit: int = 20) -> dict:
        store = SecurityClipStore(args.index)
        try:
            handle, results = store.search(query, limit=limit)
            return {
                "handle": handle,
                "results": [
                    {
                        "rank": result.rank,
                        "path": result.doc.content_path,
                        "meta_path": result.doc.meta_path,
                        "title": result.doc.title,
                        "source_id": result.doc.source_id,
                        "snippet": result.snippet,
                    }
                    for result in results
                ],
            }
        finally:
            store.close()

    @mcp.tool()
    def cat(path: str) -> str:
        store = SecurityClipStore(args.index)
        try:
            return store.cat(path)
        finally:
            store.close()

    @mcp.tool()
    def head(path: str, count: int = 50) -> str:
        store = SecurityClipStore(args.index)
        try:
            return store.head(count, path)
        finally:
            store.close()

    @mcp.tool()
    def grep(pattern: str, path: str = "/", ignore_case: bool = False, limit: int = 100) -> dict:
        store = SecurityClipStore(args.index)
        try:
            handle, matches = store.grep(pattern, path, ignore_case=ignore_case, limit=limit)
            return {
                "handle": handle,
                "matches": [
                    {"path": match.doc.content_path, "line_number": match.line_number, "line": match.line}
                    for match in matches
                ],
            }
        finally:
            store.close()

    @mcp.tool()
    def ls(path: str = "/") -> str:
        store = SecurityClipStore(args.index)
        try:
            return store.ls(path)
        finally:
            store.close()

    mcp.run()
    return 0
