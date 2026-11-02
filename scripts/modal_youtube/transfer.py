"""Prepare, transfer and recover private Modal work without rewriting its plan."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import hashlib
import subprocess

from . import common as c
from .inspect_run import inspect


def queue():
    result = subprocess.run(
        ["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%i|%j|%T|%r"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.split("|", 3) for line in result.stdout.splitlines() if line.strip()]


def check_held(plan, hold=False):
    state = c.read(Path(plan["output_dir"]) / "pool/controller.json")
    active = str(state.get("active_gpu", ""))
    rows = [r for r in queue() if r[1] == "youtube-pool"]
    if (
        not active.isdigit()
        or not rows
        or any(r[0].split("_")[0] != active or r[2] != "PENDING" for r in rows)
    ):
        raise RuntimeError(
            "Handoff requires the controller's pending YouTube array only. A worker is running or queue state changed; nothing will be cancelled."
        )
    if hold:
        subprocess.run(["scontrol", "hold", active], check=True)
        rows = [r for r in queue() if r[1] == "youtube-pool"]
    if not rows or any(
        r[0].split("_")[0] != active or r[2] != "PENDING" or r[3] != "JobHeldUser"
        for r in rows
    ):
        raise RuntimeError(
            "All YouTube workers must remain PENDING / JobHeldUser during transfer and import"
        )
    return active


def prepare(plan_path, directory):
    from scripts.curation_single import run
    from scripts.youtube_filter.score import _versions

    plan = run.load_plan(plan_path)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "manifest.json"
    if target.exists():
        manifest, _, plan = c.validate_manifest(target)
        check_held(plan)
        c.verify_files(Path(manifest["mount"]), manifest["files"])
        print(f"Reusing frozen handoff: {target}")
        return
    active = check_held(plan, hold=True)
    print(
        f"Held pending array {active}; completed filtering outputs remain in place.",
        flush=True,
    )
    inventory = inspect(plan_path)
    slots = [p["slot"] for p in inventory["parts"] if not p["workflow_finished"]]
    if not slots:
        raise ValueError("No unfinished YouTube work")
    mount = plan_path.parent.parent.resolve()
    prepared = Path(plan["input_dir"])
    if not prepared.is_relative_to(mount):
        raise ValueError(
            "Prepared inputs and scoring output must share the curation parent"
        )
    files = [
        plan_path,
        prepared / "summary.json",
        prepared / "run-config.json",
        prepared / "lineage.parquet",
    ]
    for slot in slots:
        source = prepared / c.safe_relative(plan["parts"][slot]["path"])
        if c.sha(source) != plan["parts"][slot]["sha256"]:
            raise ValueError(f"Prepared source changed: {slot}")
        files.append(source)
        # Preserve partially scored work and its original runtime/config binding.
        files.extend(c.part_files(run.folder(plan, slot)))
    entries = {}
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(mount):
            raise ValueError(f"Unexpected artifact outside mount: {path}")
        entries[str(path.relative_to(mount))] = {
            "sha256": c.sha(path),
            "bytes": path.stat().st_size,
        }
    runtime = _versions(False)
    for name in (
        "duckdb",
        "pyarrow",
        "ninja",
        "nvidia-cuda-cccl",
        "flashinfer-python",
        "xgrammar",
    ):
        runtime[name] = importlib.metadata.version(name)
    for version in runtime.values():
        if not re.fullmatch(r"[A-Za-z0-9.+_-]+", version):
            raise ValueError("Unsupported package version in environment")
    jobs = c.allocate(slots, plan["parts"])
    manifest = {
        "version": "modal-youtube-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mount": str(mount),
        "plan_relative": str(plan_path.relative_to(mount)),
        "plan_sha256": c.sha(plan_path),
        "code": c.code_hashes(),
        "runtime": runtime,
        "files": entries,
        "slots": slots,
        "jobs": jobs,
        "completed_slots": [
            p["slot"] for p in inventory["parts"] if p["workflow_finished"]
        ],
        "held_array": active,
        "model": plan["model"],
        "work_seconds": c.WORK_SECONDS,
        "call_seconds": c.CALL_SECONDS,
        "estimated_worker_ceiling_dollars": len(jobs)
        * (c.CALL_SECONDS + c.STARTUP_SECONDS)
        / 3600
        * c.HOURLY_DOLLARS,
        "remaining_input_tokens": inventory["remaining_input_tokens"],
        "volume": "security-corpus-youtube-modal-v1",
        "model_volume": "security-corpus-youtube-model-v1",
        "ledger": "security-corpus-youtube-budget-v1",
        "scope": "Private execution handoff; same original plan, model and policy. Runtime allocation is not a provider-enforced billing cap.",
    }
    if manifest["estimated_worker_ceiling_dollars"] > 300:
        raise ValueError("Worker allowance too large")
    check_held(plan)
    c.write(directory / "timing.json", inventory)
    c.write(target, manifest)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "remaining_input_tokens",
                    "held_array",
                    "estimated_worker_ceiling_dollars",
                )
            },
            indent=2,
        )
    )
    print(
        f"Frozen {len(slots)} unfinished parts / {sum(e['bytes'] for e in entries.values()) / 1e9:.2f} GB; {target}"
    )


def upload(manifest_path):
    import modal

    manifest, _, plan = c.validate_manifest(manifest_path)
    check_held(plan)
    ledger = modal.Dict.from_name(manifest["ledger"], create_if_missing=True)
    if ledger.get("launch", None) is not None:
        raise RuntimeError(
            "Cloud launch already recorded; do not overwrite worker checkpoints"
        )
    volume = modal.Volume.from_name(manifest["volume"], create_if_missing=True)
    root = Path(manifest["mount"])
    c.verify_files(root, manifest["files"])
    with volume.batch_upload(force=True) as batch:
        for relative in manifest["files"]:
            batch.put_file(str(root / relative), "/" + relative)
        batch.put_file(str(manifest_path), "/modal-handoff/manifest.json")
    check_held(plan)
    c.write(
        manifest_path.parent / "uploaded.json",
        {"manifest_sha256": c.sha(manifest_path)},
    )
    print("Private upload complete; no GPU has been requested.")


def status(manifest_path):
    import modal

    manifest = c.read(manifest_path)
    ledger = modal.Dict.from_name(manifest["ledger"])
    print(
        json.dumps(
            {
                "launch": ledger.get("launch", None),
                "done": ledger.get("coordinator_done", None),
                "workers": {
                    str(i): ledger.get(
                        f"worker-done-{i}", ledger.get(f"progress-{i}", None)
                    )
                    for i in range(len(manifest["jobs"]))
                },
            },
            indent=2,
        )
    )


def fetch(manifest_path):
    import modal

    manifest, _, plan = c.validate_manifest(manifest_path)
    check_held(plan)
    ledger = modal.Dict.from_name(manifest["ledger"])
    if ledger.get("coordinator_done", None) is None:
        raise RuntimeError(
            "Cloud coordinator has not finished; leave checkpoints in place"
        )
    volume = modal.Volume.from_name(manifest["volume"])
    target = manifest_path.parent / "returned"
    target.mkdir(exist_ok=True)
    inventories = []
    for worker in range(len(manifest["jobs"])):
        state = ledger.get(f"worker-done-{worker}", None)
        if state is None:
            print(
                f"Worker {worker} has no final inventory; retain its cloud checkpoints for recovery"
            )
            continue
        name = f"modal-handoff/worker-{worker}.json"
        raw = b"".join(volume.read_file(name))
        if hashlib.sha256(raw).hexdigest() != state["inventory_sha256"]:
            raise ValueError("Worker inventory checksum mismatch")
        report = json.loads(raw)
        if (
            c.sha(manifest_path) != report["manifest_sha256"]
            or report["worker"] != worker
        ):
            raise ValueError("Cloud worker provenance mismatch")
        for relative, expected in report["files"].items():
            rel = c.safe_relative(relative)
            # Downloads may only contain this worker's allocated output parts.
            allowed = [
                Path(plan["output_dir"]).relative_to(manifest["mount"])
                / "parts"
                / f"{n:05d}"
                for n in manifest["jobs"][worker]
            ]
            if not any(rel.is_relative_to(prefix) for prefix in allowed):
                raise ValueError("Cloud export escapes assigned parts")
            destination = target / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            if (
                destination.exists()
                and destination.stat().st_size == expected["bytes"]
                and c.sha(destination) == expected["sha256"]
            ):
                continue
            temporary = destination.with_suffix(destination.suffix + ".download")
            with temporary.open("wb") as handle:
                for block in volume.read_file(relative):
                    handle.write(block)
            if (
                c.sha(temporary) != expected["sha256"]
                or temporary.stat().st_size != expected["bytes"]
            ):
                raise ValueError(f"Download checksum mismatch: {relative}")
            temporary.replace(destination)
        inventories.append(report)
        c.write(target / f"worker-{worker}.json", report)
    c.write(
        target / "downloaded.json",
        {
            "manifest_sha256": c.sha(manifest_path),
            "workers": [r["worker"] for r in inventories],
        },
    )
    print(
        f"Downloaded to {target}. Run the CPU import job before releasing held GPU jobs."
    )


def merge(manifest_path):
    from scripts.curation_single import run
    from scripts.youtube.download import _directory_lock

    manifest, plan_path, plan = c.validate_manifest(manifest_path)
    check_held(plan)
    root = Path(manifest["mount"])
    returned = manifest_path.parent / "returned"
    download = c.read(returned / "downloaded.json")
    if download["manifest_sha256"] != c.sha(manifest_path):
        raise ValueError("Download belongs to another handoff")
    staged = {
        **plan,
        "output_dir": str(returned / Path(plan["output_dir"]).relative_to(root)),
    }
    report_path = manifest_path.parent / "import-summary.json"
    report = {"complete": False, "imported_slots": [], "skipped_unfinished": []}
    for worker in download["workers"]:
        exported = c.read(returned / f"worker-{worker}.json")
        for slot in manifest["jobs"][worker]:
            source, destination = run.folder(staged, slot), run.folder(plan, slot)
            if not (source / "summary.json").exists():
                if (destination / "modal-origin.json").exists() and c.read(
                    destination / "modal-origin.json"
                )["manifest_sha256"] == c.sha(manifest_path):
                    report["imported_slots"].append(slot)
                continue
            if not c.read(source / "summary.json").get("workflow_finished"):
                report["skipped_unfinished"].append(slot)
                continue
            prefix = (
                str(
                    Path(plan["output_dir"]).relative_to(root) / "parts" / f"{slot:05d}"
                )
                + "/"
            )
            c.verify_files(
                returned,
                {k: v for k, v in exported["files"].items() if k.startswith(prefix)},
            )
            with c_bound(plan_path, manifest):
                summary = run.checked_summary(plan_path, staged, slot)
                counts, _ = run.audit_part(plan_path, staged, slot)
            if any(summary[k] != v for k, v in counts.items()):
                raise ValueError(f"Cloud summary/raw-response mismatch: {slot}")
            destination.mkdir(parents=True, exist_ok=True)
            with _directory_lock(destination):
                saved = run.checked_summary(plan_path, plan, slot)
                if saved and saved.get("workflow_finished"):
                    if c.sha(destination / "summary.json") != c.sha(
                        source / "summary.json"
                    ):
                        raise ValueError(
                            f"Another completed output exists: {slot}; preserve both"
                        )
                    report["imported_slots"].append(slot)
                    continue
                backup = manifest_path.parent / "before-import" / f"{slot:05d}"
                backup.parent.mkdir(exist_ok=True)
                if backup.exists():
                    raise ValueError(
                        f"Prior import backup exists for {slot}; inspect before recovery"
                    )
                destination.rename(backup)
                source.rename(destination)
            report["imported_slots"].append(slot)
            c.write(report_path, report)
            print(f"Imported verified slot {slot}", flush=True)
    report["complete"] = True
    c.write(report_path, report)
    print(
        f"Import complete: {len(report['imported_slots'])} verified parts. Held Slurm array remains paused."
    )


@contextmanager
def c_bound(path, manifest):
    from scripts.curation_single import run

    if c.sha(path) != manifest["plan_sha256"]:
        raise ValueError("Plan changed during import")
    original = run._sha256
    run._sha256 = lambda target: (
        manifest["plan_sha256"] if Path(target) == path else original(target)
    )
    try:
        yield
    finally:
        run._sha256 = original
        if c.sha(path) != manifest["plan_sha256"]:
            raise ValueError("Plan changed during import")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "upload", "fetch", "import", "status")
    )
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if not args.plan or not args.directory:
            parser.error("prepare needs --plan and --directory")
        prepare(args.plan.resolve(), args.directory.resolve())
    else:
        if not args.manifest:
            parser.error("--manifest required")
        {"upload": upload, "fetch": fetch, "import": merge, "status": status}[
            args.command
        ](args.manifest.resolve())


if __name__ == "__main__":
    main()
