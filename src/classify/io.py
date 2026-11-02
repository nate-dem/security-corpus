"""Shared I/O helpers for downstream filtering workflows."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4


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


def publish_outputs(outputs: dict[Path, Path]) -> None:
    """Replace completed artifacts, rolling back replacements on an I/O error.

    Staging and destinations must share a filesystem. This protects prior
    checkpoints during computation and ordinary failures; multiple renames are
    not a transaction against power loss or SIGKILL.
    """
    backups: dict[Path, Path] = {}
    installed = []
    try:
        for staged, destination in outputs.items():
            if destination.exists():
                backup = destination.with_name(f".{destination.name}.{uuid4().hex}.backup")
                os.replace(destination, backup)
                backups[destination] = backup
            os.replace(staged, destination)
            installed.append((staged, destination))
    except BaseException:
        for staged, destination in reversed(installed):
            os.replace(destination, staged)
        for destination, backup in backups.items():
            os.replace(backup, destination)
        raise
    for backup in backups.values():
        if backup.is_dir():
            shutil.rmtree(backup)
        else:
            backup.unlink()
