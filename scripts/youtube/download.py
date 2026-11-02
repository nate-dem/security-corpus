#!/usr/bin/env python3
"""Download every YouTube-Commons Parquet shard without filtering any records.

This directory is self-contained and can be copied to Marlowe on its own.
Hugging Face handles partial HTTP transfers; this script pins the inventory,
checks every file against upstream hashes, and reports incomplete runs as errors.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import time


REPO_ID = "PleIAs/YouTube-Commons"
# Verified against the public Hub inventory on 2026-09-10. All files at this
# snapshot are enumerated; neither a shard count nor a filename range is assumed.
DEFAULT_REVISION = "9addbabbfcd7409acbcd11a3b59ec2aef6da7eb0"
LOG = logging.getLogger("youtube-download")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _directory_lock(root: Path):
    # An OS lock is released on exit, Ctrl-C, or job termination. The empty
    # lock file may remain; it must not be deleted to "unlock" a running job.
    with (root / ".download.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another downloader is using {root}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _hub_api():
    from huggingface_hub import HfApi

    return HfApi(token=False)


def _hub_download(**kwargs) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(token=False, **kwargs)


def _validate_manifest(manifest: dict, *, repo_id: str = REPO_ID,
                       suffixes: tuple[str, ...] = (".parquet",)) -> None:
    if manifest.get("format_version") != 1 or manifest.get("repo_id") != repo_id:
        raise ValueError("Manifest is not a supported dataset inventory")
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("revision", ""))):
        raise ValueError("Manifest must pin an immutable dataset commit")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Manifest contains no Parquet shards")
    names = set()
    for entry in files:
        name = entry["path"]
        path = PurePosixPath(name)
        if (
            not isinstance(name, str) or "\\" in name
            or path.is_absolute() or ".." in path.parts
            or path.as_posix() != name or name.startswith(".")
            or not name.endswith(suffixes) or name in names
        ):
            raise ValueError(f"Unsafe or duplicate shard path: {name}")
        names.add(name)
        if type(entry["size"]) is not int or entry["size"] <= 0:
            raise ValueError(f"Invalid expected size: {name}")
        algorithm = entry.get("hash_algorithm")
        digest_length = {"sha256": 64, "git_blob_sha1": 40}.get(algorithm)
        if digest_length is None or not re.fullmatch(
            rf"[0-9a-f]{{{digest_length}}}", str(entry.get("hash", "")),
        ):
            raise ValueError(f"Missing/invalid upstream checksum: {name}")
    if manifest.get("total_bytes") != sum(entry["size"] for entry in files):
        raise ValueError("Manifest byte total does not match its shard inventory")


def _load_or_create_manifest(root: Path, revision: str | None, *, offline: bool) -> dict:
    destination = root / "manifest.json"
    if destination.exists():
        manifest = json.loads(destination.read_text(encoding="utf-8"))
        _validate_manifest(manifest)
        if revision is not None and revision != manifest["revision"]:
            raise ValueError("Dataset revision changed; use a different --output-dir")
        return manifest
    if offline:
        raise FileNotFoundError(f"Cannot verify without {destination}")
    requested_revision = revision or DEFAULT_REVISION
    if not re.fullmatch(r"[0-9a-f]{40}", requested_revision):
        raise ValueError("--revision must be a full immutable 40-character commit SHA")
    info = _hub_api().dataset_info(
        REPO_ID, revision=requested_revision, files_metadata=True, timeout=60,
    )
    if info.sha != requested_revision:
        raise ValueError("Hub response does not match the requested dataset revision")
    files = []
    for sibling in sorted(info.siblings, key=lambda item: item.rfilename):
        if not sibling.rfilename.endswith(".parquet"):
            continue
        lfs = sibling.lfs
        files.append({
            "path": sibling.rfilename,
            "size": sibling.size,
            "hash_algorithm": "sha256" if lfs else "git_blob_sha1",
            "hash": lfs["sha256"] if lfs else sibling.blob_id,
        })
    manifest = {
        "format_version": 1,
        "repo_id": REPO_ID,
        "revision": info.sha,
        "created_at": _utc_now(),
        "selection": "Every .parquet file in the pinned repository; no row filters",
        "total_bytes": sum(entry["size"] for entry in files),
        "files": files,
    }
    _validate_manifest(manifest)
    _write_json(destination, manifest)
    return manifest


def _shard_path(raw_dir: Path, entry: dict) -> Path:
    path = raw_dir / entry["path"]
    if not path.resolve().is_relative_to(raw_dir.resolve()) or path.is_symlink():
        raise ValueError(f"Shard path points outside the download directory: {path}")
    return path


def _verify_file(path: Path, entry: dict) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing shard: {entry['path']}")
    before = path.stat()
    if before.st_size != entry["size"]:
        raise ValueError(f"Size mismatch for {entry['path']}: {before.st_size} != {entry['size']}")
    sha256 = hashlib.sha256()
    expected_digest = sha256
    if entry["hash_algorithm"] == "git_blob_sha1":
        expected_digest = hashlib.sha1(f"blob {before.st_size}\0".encode())
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            sha256.update(chunk)
            if expected_digest is not sha256:
                expected_digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed during verification: {entry['path']}")
    if expected_digest.hexdigest() != entry["hash"]:
        raise ValueError(f"Checksum mismatch for {entry['path']}")
    return {"path": entry["path"], "bytes": after.st_size, "sha256": sha256.hexdigest()}


def _download_one(entry: dict, manifest: dict, root: Path, retries: int, verify_only: bool,
                  *, download_fn=None) -> dict:
    raw_dir = root / "raw"
    path = _shard_path(raw_dir, entry)
    if verify_only:
        return {**_verify_file(path, entry), "action": "verified"}
    if path.exists():
        try:
            return {**_verify_file(path, entry), "action": "already_verified"}
        except ValueError as exc:
            LOG.warning("%s; replacing this invalid shard", exc)
            # Only a shard that failed its upstream integrity check is removed.
            # Ordinary interrupted Hub transfers remain in raw/.cache for resume.
            path.unlink()
    for attempt in range(1, retries + 1):
        try:
            downloaded = Path((download_fn or _hub_download)(
                repo_id=manifest["repo_id"], repo_type="dataset",
                revision=manifest["revision"], filename=entry["path"],
                local_dir=raw_dir, cache_dir=root / ".hub-cache",
            ))
            if downloaded.resolve() != path.resolve():
                raise ValueError(f"Unexpected download destination: {downloaded}")
            try:
                verified = _verify_file(path, entry)
            except ValueError:
                path.unlink(missing_ok=True)
                raise
            return {**verified, "action": "downloaded"}
        except Exception as exc:
            if attempt == retries:
                raise
            delay = min(2 ** attempt, 30)
            LOG.warning("%s: attempt %d/%d failed (%s); retrying in %ds",
                        entry["path"], attempt, retries, type(exc).__name__, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _configure_hub(root: Path) -> None:
    # Set before importing the Hub. HTTP resume is sufficient; keep Xet's
    # internal fan-out and cache off for this small CPU allocation.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ.setdefault("HF_HOME", str(root / ".hf-home"))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")


def _run(args: argparse.Namespace, root: Path) -> int:
    _configure_hub(root)
    manifest = _load_or_create_manifest(root, args.revision, offline=args.verify_only)
    return _transfer_manifest(args, root, manifest)


def _transfer_manifest(args: argparse.Namespace, root: Path, manifest: dict, *, download_fn=None) -> int:
    files = manifest["files"]
    LOG.info("Snapshot %s: %d shards, %.2f GB (%.2f GiB)", manifest["revision"],
             len(files), manifest["total_bytes"] / 1e9, manifest["total_bytes"] / 2**30)
    LOG.info("Data destination: %s", root / "raw")
    if args.plan_only:
        LOG.info("Inventory saved to %s; no shards downloaded", root / "manifest.json")
        return 0

    raw_dir = root / "raw"
    raw_dir.mkdir(exist_ok=True)
    if not args.verify_only:
        remaining = sum(
            max(0, entry["size"] - (path.stat().st_size if path.is_file() else 0))
            for entry in files for path in [_shard_path(raw_dir, entry)]
        )
        free = shutil.disk_usage(root).free
        # A small buffer allows metadata and an in-flight replacement shard.
        needed = remaining + max(entry["size"] for entry in files)
        LOG.info("Filesystem free: %.2f GiB; estimated additional need: %.2f GiB. "
                 "Filesystem free space does not include your project quota.", free / 2**30, needed / 2**30)
        if free < needed:
            raise OSError("Insufficient filesystem space for this inventory")

    report = {
        "repo_id": manifest["repo_id"], "revision": manifest["revision"],
        "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "started_at": _utc_now(), "status": "running", "complete": False,
        "expected_shards": len(files), "expected_bytes": manifest["total_bytes"],
        "verified_shards": 0, "verified_bytes": 0, "verified": [], "failed": [],
    }
    summary_path = root / "download-summary.json"
    _write_json(summary_path, report)
    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = {}
    try:
        for entry in files:
            kwargs = {"download_fn": download_fn} if download_fn is not None else {}
            futures[executor.submit(_download_one, entry, manifest, root, args.retries,
                                    args.verify_only, **kwargs)] = entry
        for future in as_completed(futures):
            entry = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                # Avoid persisting request URLs, which can contain signed tokens.
                report["failed"].append({"path": entry["path"], "error_type": type(exc).__name__})
                LOG.error("FAILED %s (%s)", entry["path"], type(exc).__name__)
            else:
                report["verified"].append(result)
                report["verified_shards"] += 1
                report["verified_bytes"] += result["bytes"]
                LOG.info("[%d/%d] %s: %s | verified %.2f/%.2f GiB",
                         report["verified_shards"], len(files), result["action"], entry["path"],
                         report["verified_bytes"] / 2**30, manifest["total_bytes"] / 2**30)
            _write_json(summary_path, report)
    except BaseException:
        report["status"] = "interrupted"
        _write_json(summary_path, report)
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    report["complete"] = report["verified_shards"] == len(files) and not report["failed"]
    report["status"] = "complete" if report["complete"] else "incomplete"
    report["finished_at"] = _utc_now()
    report["verified"].sort(key=lambda item: item["path"])
    report["failed"].sort(key=lambda item: item["path"])
    _write_json(summary_path, report)
    LOG.info("%s: %d/%d shards verified; %d failures. Report: %s",
             report["status"].upper(), report["verified_shards"], len(files),
             len(report["failed"]), summary_path)
    return 0 if report["complete"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Dedicated dataset directory containing raw/ and manifests.")
    parser.add_argument("--revision", help=f"Immutable commit; new directories default to {DEFAULT_REVISION}.")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent shard transfers (default: 2).")
    parser.add_argument("--retries", type=int, default=4, help="Attempts per shard in this run (default: 4).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true", help="Freeze inventory without downloading shards.")
    mode.add_argument("--verify-only", action="store_true", help="Recheck all files offline; never download/repair.")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.retries < 1:
        parser.error("--workers and --retries must be positive")
    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(root / "download.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(file_handler)
    try:
        with _directory_lock(root):
            return _run(args, root)
    except KeyboardInterrupt:
        LOG.warning("Interrupted. Run the same command again to resume.")
        return 130
    except Exception as exc:
        LOG.error("Stopped (%s): %s", type(exc).__name__,
                  str(exc) if not type(exc).__module__.startswith(("requests", "huggingface_hub"))
                  else "Hub request failed; check network access and retry.")
        return 1
    finally:
        LOG.removeHandler(file_handler)
        file_handler.close()


if __name__ == "__main__":
    raise SystemExit(main())
