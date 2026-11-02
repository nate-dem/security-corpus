"""Run existing CPU finalization with bounded, byte-equivalent checksum I/O.

The frozen auditor and its checkpoints remain unchanged. Every file is read on
every fingerprint call; this wrapper never substitutes timestamps or cached
digests for file bytes. Its own code identity is recorded in the Slurm log.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import sys
import time

from scripts.curation_audit import run as audit

LOG = logging.getLogger(__name__)
TARGETS = {
    "pipeline": "scripts.youtube_finalize.pipeline",
    "finalize": "scripts.youtube_finalize.finalize",
    "full-audit": "scripts.youtube_resume.full_audit",
}


def part_files(directory):
    """Match rglob/is_file selection, using scandir's cached entry metadata.

    Like the frozen auditor, follow file symlinks but do not recurse through
    directory symlinks. Sort Path objects exactly as the original function does.
    Only advisory lock files are excluded; hidden and extra files remain bound.
    """
    pending, files = [directory], []
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file() and entry.name != ".download.lock":
                    files.append(path)
    return sorted(files)


def fingerprint(plan, entry, threads):
    """Compute the original digest, retaining its exact file/path ordering."""
    slot = entry["slot"]
    directory = audit.run.folder(plan, slot)
    started = time.monotonic()
    LOG.info("Checksum slot %s: checking summary and prepared input", slot)
    if audit._sha256(directory / "summary.json") != entry["summary_sha256"]:
        raise ValueError(f"Frozen worker summary changed: slot {slot}")
    part = plan["parts"][slot]
    source = Path(plan["input_dir"]) / part["path"]
    if audit._sha256(source) != part["sha256"]:
        raise ValueError(f"Prepared input changed: slot {slot}")
    files = part_files(directory)
    LOG.info("Checksum slot %s: reading %s worker files", slot, len(files))
    result = hashlib.sha256(part["sha256"].encode())
    last_log = time.monotonic()
    # map() eagerly submits on Python 3.10. Batches bound the queued futures and
    # completed digests as well as the number of simultaneously open files.
    batch_size = threads * 8
    # Close all threads before returning: the next stage may fork dedup workers.
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for start in range(0, len(files), batch_size):
            batch = files[start : start + batch_size]
            for file, digest in zip(batch, pool.map(audit._sha256, batch)):
                result.update(str(file.relative_to(directory)).encode() + b"\0")
                result.update(digest.encode() + b"\n")
            if time.monotonic() - last_log >= 30:
                LOG.info(
                    "Checksum slot %s: %s/%s files; %.1fs elapsed",
                    slot,
                    start + len(batch),
                    len(files),
                    time.monotonic() - started,
                )
                last_log = time.monotonic()
    LOG.info(
        "Checksum slot %s: complete; %s files; %.1fs elapsed",
        slot,
        len(files),
        time.monotonic() - started,
    )
    return result.hexdigest()


@contextmanager
def parallel_fingerprints(threads):
    if threads < 1:
        raise ValueError("Checksum thread count must be positive")
    original = audit.fingerprint

    def parallel(plan, entry):
        return fingerprint(plan, entry, threads)

    audit.fingerprint = parallel
    try:
        yield
    finally:
        audit.fingerprint = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("stage", choices=TARGETS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    allocation = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocation and args.threads > int(allocation):
        parser.error("--threads exceeds SLURM_CPUS_PER_TASK")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    target = importlib.import_module(TARGETS[args.stage])
    LOG.info(
        "Checksum execution: %s",
        json.dumps(
            {
                "wrapper": str(Path(__file__).resolve()),
                "wrapper_sha256": audit._sha256(Path(__file__)),
                "auditor_code": audit.auditor_code(),
                "target": TARGETS[args.stage],
                "target_sha256": audit._sha256(Path(target.__file__)),
                "threads": args.threads,
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "digest_policy": "original bytes and sorted relative paths; no digest cache",
            },
            sort_keys=True,
        ),
    )
    previous = sys.argv
    sys.argv = [TARGETS[args.stage], *args.arguments]
    try:
        with parallel_fingerprints(args.threads):
            target.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
