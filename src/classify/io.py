"""Shared I/O helpers for downstream classifier workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_parent(path: Path) -> None:
    """Create the parent directory for a file path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def find_parquet_files(path: Path) -> list[Path]:
    """Return Parquet files from a file path or recursively from a directory."""
    if path.is_file():
        return [path]

    files = sorted(path.glob("**/*.parquet"))
    return files


def read_json(path: Path) -> dict[str, Any]:
    """Read a small JSON metadata file."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a small JSON metadata file."""
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
