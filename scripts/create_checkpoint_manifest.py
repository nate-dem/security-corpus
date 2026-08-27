#!/usr/bin/env python3
"""Create a checksummed inventory of local corpus checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Sequence

import pyarrow.parquet as pq


HASH_CHUNK_BYTES = 8 * 1024 * 1024


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    excluded = tuple((repo_root / value).resolve() for value in args.exclude)
    paths = list(_iter_files(repo_root, args.roots, excluded))
    if not paths:
        raise SystemExit("No checkpoint files found")

    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "git": _git_state(repo_root),
        "roots": args.roots,
        "excluded": args.exclude,
        "files": 0,
        "bytes": 0,
        "parquet_files": 0,
        "parquet_rows": 0,
        "parquet_tokens": 0,
        "errors": [],
    }

    manifest_path = output_dir / "files.jsonl.gz"
    with gzip.open(manifest_path, "wt", encoding="utf-8") as output:
        for index, path in enumerate(paths, start=1):
            entry = _file_entry(repo_root, path)
            output.write(json.dumps(entry, sort_keys=True) + "\n")
            summary["files"] += 1
            summary["bytes"] += entry["size_bytes"]

            parquet = entry.get("parquet")
            if parquet:
                summary["parquet_files"] += 1
                summary["parquet_rows"] += parquet["rows"]
                summary["parquet_tokens"] += parquet.get("tokens", 0)
            if entry.get("error"):
                summary["errors"].append(
                    {"path": entry["path"], "error": entry["error"]}
                )

            if index % 1_000 == 0 or index == len(paths):
                print(f"Manifested {index:,}/{len(paths):,} files", flush=True)

    summary["files_manifest"] = manifest_path.name
    summary["files_manifest_sha256"] = _sha256(manifest_path)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {summary_path}")
    print(f"Wrote {manifest_path}")
    return 1 if summary["errors"] else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="+",
        help="Files or directories to inventory, relative to --repo-root.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Relative file or directory to exclude. Repeat as needed.",
    )
    return parser


def _iter_files(
    repo_root: Path,
    roots: Iterable[str],
    excluded: tuple[Path, ...],
) -> Iterable[Path]:
    found: set[Path] = set()
    for value in roots:
        root = (repo_root / value).resolve()
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if not path.is_file() or _is_excluded(path, excluded):
                continue
            if path in found:
                continue
            found.add(path)
            yield path


def _is_excluded(path: Path, excluded: tuple[Path, ...]) -> bool:
    return any(path == item or item in path.parents for item in excluded)


def _file_entry(repo_root: Path, path: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": path.relative_to(repo_root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix != ".parquet":
        return entry

    try:
        parquet = pq.ParquetFile(path)
        columns = parquet.schema_arrow.names
        details: dict[str, Any] = {
            "rows": parquet.metadata.num_rows,
            "row_groups": parquet.metadata.num_row_groups,
            "columns": columns,
        }
        if "content_length" in columns:
            details["tokens"] = sum(
                value or 0
                for batch in parquet.iter_batches(columns=["content_length"])
                for value in batch.column(0).to_pylist()
            )
        entry["parquet"] = details
    except Exception as exc:  # Record corruption without losing the inventory.
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(repo_root: Path) -> dict[str, str]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "status": run("status", "--short"),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    raise SystemExit(main())
