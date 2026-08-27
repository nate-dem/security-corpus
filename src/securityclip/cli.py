"""Command-line interface for Security Scope."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from securityclip.config import DEFAULT_INDEX_DIR, INDEX_ENV_VAR, SOURCE_LAYOUTS, SourceSpec, default_index_dir
from securityclip.indexer import DEFAULT_FTS_MAX_CHARS, build_index
from securityclip.store import SecurityClipStore


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            specs = tuple(SourceSpec(f"custom-{idx}", pattern) for idx, pattern in enumerate(args.source, start=1))
            if not specs:
                specs = SOURCE_LAYOUTS[args.source_layout]
            summary = build_index(
                args.index,
                source_specs=specs,
                overwrite=args.overwrite,
                fts_max_chars=args.fts_max_chars,
                batch_size=args.batch_size,
            )
            print(f"Indexed {summary.documents:,} documents at {summary.index_dir}")
            if summary.skipped_missing_patterns:
                print("Skipped missing patterns:")
                for pattern in summary.skipped_missing_patterns:
                    print(f"  {pattern}")
            return 0

        store = SecurityClipStore(args.index)
        try:
            return _run_store_command(store, args)
        finally:
            store.close()
    except Exception as exc:  # noqa: BLE001 - CLI should turn failures into plain text
        print(f"securityclip: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="security-scope")
    parser.add_argument(
        "--index",
        type=Path,
        default=default_index_dir(),
        help=f"Index directory (default: ${INDEX_ENV_VAR} if set, otherwise {DEFAULT_INDEX_DIR})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build the local Security Scope index.")
    build.add_argument("--source", action="append", default=[], help="Override input Parquet pattern; can be repeated.")
    build.add_argument(
        "--source-layout",
        choices=sorted(SOURCE_LAYOUTS),
        default="repo",
        help="Built-in source path layout to use when --source is not provided.",
    )
    build.add_argument("--overwrite", action="store_true", help="Replace an existing index directory.")
    build.add_argument("--batch-size", type=int, default=500)
    build.add_argument("--fts-max-chars", type=int, default=DEFAULT_FTS_MAX_CHARS)

    search = subparsers.add_parser("search", help="Search the corpus.")
    search.add_argument("query")
    search.add_argument("-n", "--limit", type=int, default=20)
    search.add_argument("--json", action="store_true")

    cat = subparsers.add_parser("cat", help="Read a virtual file.")
    cat.add_argument("path")
    cat.add_argument("--all", action="store_true", help="Print all lines for content.lines files.")

    head = subparsers.add_parser("head", help="Preview a virtual file.")
    head.add_argument("count_or_path")
    head.add_argument("path", nargs="?")
    head.add_argument("--json", action="store_true")

    grep = subparsers.add_parser("grep", help="Regex search virtual files.")
    grep.add_argument("-i", "--ignore-case", action="store_true")
    grep.add_argument("--from", dest="from_handle")
    grep.add_argument("--limit", type=int, default=100)
    grep.add_argument("--json", action="store_true")
    grep.add_argument("pattern")
    grep.add_argument("path", nargs="?")

    ls = subparsers.add_parser("ls", help="List virtual directories.")
    ls.add_argument("path", nargs="?", default="/")
    ls.add_argument("--limit", type=int, default=200)
    ls.add_argument("--json", action="store_true")

    map_cmd = subparsers.add_parser("map", help="Run optional configured reader over a result handle.")
    map_cmd.add_argument("--from", dest="from_handle", required=True)
    map_cmd.add_argument("--limit", type=int)
    map_cmd.add_argument("prompt")

    return parser


def _run_store_command(store: SecurityClipStore, args: argparse.Namespace) -> int:
    if args.command == "search":
        handle, results = store.search(args.query, limit=args.limit)
        if args.json:
            print(json.dumps({
                "handle": handle,
                "results": [_result_json(result) for result in results],
            }, indent=2))
            return 0
        print(f"Search results: {handle}")
        for result in results:
            doc = result.doc
            title = doc.title or doc.record_id
            print(f"{result.rank}. {doc.content_path}")
            print(f"   Title: {title}")
            print(f"   Source: {doc.source_id} | Tokens: {doc.content_length or 'unknown'}")
            if result.snippet:
                print(f"   Snippet: {_single_line(result.snippet)}")
        return 0
    if args.command == "cat":
        print(store.cat(args.path, all_lines=args.all), end="")
        return 0
    if args.command == "head":
        count, path = _parse_head_args(args.count_or_path, args.path)
        text = store.head(count, path)
        if args.json:
            print(json.dumps({"path": path, "count": count, "text": text, "lines": _line_json(text)}, indent=2))
            return 0
        print(text, end="")
        return 0
    if args.command == "grep":
        handle, matches = store.grep(
            args.pattern,
            args.path,
            from_handle=args.from_handle,
            ignore_case=args.ignore_case,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps({
                "handle": handle,
                "matches": [_grep_match_json(match) for match in matches],
            }, indent=2))
            return 0
        for match in matches:
            print(f"{match.doc.content_path}:L{match.line_number}:{match.line}")
        if handle:
            print(f"Search results: {handle}")
        return 0
    if args.command == "ls":
        text = store.ls(args.path, limit=args.limit)
        if args.json:
            print(json.dumps({"path": args.path, "children": [line for line in text.splitlines() if line]}, indent=2))
            return 0
        print(text, end="")
        return 0
    if args.command == "map":
        print(store.map(args.from_handle, args.prompt, limit=args.limit), end="")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def _parse_head_args(count_or_path: str, path: str | None) -> tuple[int, str]:
    if count_or_path.startswith("-") and count_or_path[1:].isdigit():
        if path is None:
            raise ValueError("head count provided without a path")
        return int(count_or_path[1:]), path
    if path is not None:
        raise ValueError("head accepts either `head PATH` or `head -N PATH`")
    return 10, count_or_path


def _result_json(result) -> dict:
    doc = result.doc
    return {
        "rank": result.rank,
        "path": doc.content_path,
        "meta_path": doc.meta_path,
        "title": doc.title,
        "source_id": doc.source_id,
        "record_id": doc.record_id,
        "snippet": result.snippet,
    }


def _grep_match_json(match) -> dict:
    return {
        "path": match.doc.content_path,
        "line_number": match.line_number,
        "line": match.line,
        "citation": f"{match.doc.content_path}:L{match.line_number}",
        "source_id": match.doc.source_id,
        "title": match.doc.title,
    }


def _line_json(text: str) -> list[dict]:
    lines = []
    for raw in text.splitlines():
        if raw.startswith("L") and ":" in raw:
            maybe_number, value = raw[1:].split(":", 1)
            if maybe_number.isdigit():
                lines.append({"line_number": int(maybe_number), "line": value.lstrip()})
                continue
        lines.append({"line_number": None, "line": raw})
    return lines


def _single_line(value: str) -> str:
    return " ".join(value.split())
