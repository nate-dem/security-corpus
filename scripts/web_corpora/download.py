"""Download pinned RedSage/Primus snapshots with authenticated HTTP resume.

Uses the tested YouTube transfer/verification engine, with explicit Hugging
Face authentication. No token is accepted on the command line or written to
reports. Access conditions must first be accepted on the dataset website.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from scripts.youtube.download import (
    LOG, _directory_lock, _transfer_manifest, _validate_manifest, _write_json,
)
from .sources import SOURCES, data_files


def validate_manifest(manifest: dict, source: str) -> None:
    spec = SOURCES[source]
    _validate_manifest(manifest, repo_id=spec["repo_id"], suffixes=(spec["suffix"], ".md"))
    if manifest.get("source") != source or manifest["revision"] != spec["revision"]:
        raise ValueError("Manifest does not match the approved source snapshot")
    if not data_files(manifest):
        raise ValueError("No data files in manifest")


def load_manifest(root: Path, source: str, *, offline: bool = False) -> dict:
    bundled = Path(__file__).parent / "manifests" / f"{source}.json"
    expected = json.loads(bundled.read_text())
    validate_manifest(expected, source)
    destination = root / "manifest.json"
    if destination.exists():
        actual = json.loads(destination.read_text())
        validate_manifest(actual, source)
        if actual != expected:
            raise ValueError("Snapshot inventory changed; choose a new data directory")
        return actual
    if offline:
        raise FileNotFoundError(f"No download manifest at {destination}")
    _write_json(destination, expected)
    return expected


def check_access(manifest: dict) -> None:
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    # A HEAD request verifies gated file access without downloading the payload.
    get_hf_file_metadata(hf_hub_url(
        manifest["repo_id"], data_files(manifest)[0]["path"],
        repo_type="dataset", revision=manifest["revision"],
    ), token=True, timeout=30)


def authenticated_download(**kwargs) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(token=True, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Parent directory holding a separate directory per source.")
    parser.add_argument("--source", choices=[*SOURCES, "all"], default="all")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=4)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan-only", action="store_true", help="Offline pinned inventory; no login needed.")
    modes.add_argument("--check-access", action="store_true", help="Small authenticated HEAD requests only.")
    modes.add_argument("--verify-only", action="store_true", help="Offline verification, never repairs.")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.retries < 1:
        parser.error("workers and retries must be positive")
    # Preserve HF_HOME/HF_TOKEN_PATH: the login command and job must share the
    # same credential location. No credentials or tokenizer assets go in raw/.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sources = list(SOURCES) if args.source == "all" else [args.source]
    failed = False
    for source in sources:
        root = args.data_dir.expanduser().resolve() / source
        root.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(root / "download.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOG.addHandler(handler)
        try:
            with _directory_lock(root):
                manifest = load_manifest(root, source, offline=args.verify_only)
                LOG.info("%s: %d data files, %.2f GB plus small source documentation",
                         source, len(data_files(manifest)), manifest["total_bytes"] / 1e9)
                if not args.plan_only and not args.verify_only:
                    check_access(manifest)
                    LOG.info("Access OK: %s", manifest["repo_id"])
                if not args.check_access:
                    failed |= bool(_transfer_manifest(args, root, manifest,
                                                      download_fn=authenticated_download))
        except KeyboardInterrupt:
            LOG.warning("Interrupted; rerun the same command to resume")
            return 130
        except Exception as exc:
            failed = True
            # Never echo a Hub exception/request URL; it may contain signed credentials.
            if type(exc).__module__.startswith(("huggingface_hub", "requests", "httpx")):
                LOG.error("%s failed (%s). Accept access conditions at https://huggingface.co/datasets/%s, "
                          "run hf auth login, then retry --check-access. Check network if already authorized.",
                          source, type(exc).__name__, SOURCES[source]["repo_id"])
            else:
                LOG.error("%s failed (%s): %s", source, type(exc).__name__, exc)
        finally:
            LOG.removeHandler(handler)
            handler.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
