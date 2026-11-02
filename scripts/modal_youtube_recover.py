"""Bundle stopped Modal checkpoints on CPUs and fetch them directly to Marlowe.

Never launches inference, rewrites the frozen handoff, or resets the GPU ledger.
Lives outside the frozen adapter tree. The normal CPU import audits raw responses.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

PREFIX = "modal-recovery-v1"
RECOVERY_VOLUME = "security-corpus-youtube-recovery-v1"
TIMEOUT = 7200


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def relative(value):
    p = Path(value)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError(f"Unsafe artifact name: {value}")
    return p


def ledger_snapshot(ledger, count):
    return {
        k: ledger.get(k, None)
        for k in ["launch", "startup_ok", "coordinator_done"]
        + [
            f"{prefix}-{i}"
            for i in range(count)
            for prefix in ("worker-claim", "worker-done", "progress")
        ]
    }


class HashingReader:
    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        data = self.handle.read(size)
        self.digest.update(data)
        return data


def bundle_worker(root, manifest, plan, manifest_sha, worker, state, destination):
    """One archive per worker; hash bytes as archived, retaining partial work too."""
    old_path = root / "modal-handoff" / f"worker-{worker}.json"
    old_state = state.get(f"worker-done-{worker}")
    old = None
    if old_state is not None:
        if sha(old_path) != old_state["inventory_sha256"]:
            raise ValueError("Original worker inventory checksum mismatch")
        old = read(old_path)
        if old["manifest_sha256"] != manifest_sha or old["worker"] != worker:
            raise ValueError("Original worker inventory belongs to another run")
    output = Path(plan["output_dir"]).relative_to(manifest["mount"])
    inventory = {
        "manifest_sha256": manifest_sha,
        "worker": worker,
        "completed_slots": [],
        "unfinished_slots_with_files": [],
        "files": {},
        "records": 0,
        "input_tokens": 0,
        "candidate_records": 0,
        "candidate_tokens": 0,
        "unresolved_documents": 0,
        "scope": "Recovery inventory of saved files, not a successful worker exit or completed full-source audit.",
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
        for slot in manifest["jobs"][worker]:
            directory = root / output / "parts" / f"{slot:05d}"
            files = sorted(
                p
                for p in directory.rglob("*")
                if p.is_file() and p.name != ".download.lock"
            )
            summary_path = directory / "summary.json"
            summary = read(summary_path) if summary_path.exists() else {}
            if summary and summary.get("plan_sha256") != manifest["plan_sha256"]:
                raise ValueError(f"Wrong scoring plan in slot {slot}")
            finished = summary.get("workflow_finished") is True
            if finished:
                part = plan["parts"][slot]
                if (
                    summary.get("slot") != slot
                    or summary.get("source") != "youtube-commons"
                    or summary.get("records") != part["records"]
                    or summary.get("input_tokens") != part["tokens"]
                    or set(summary["files"])
                    != {
                        "documents.parquet",
                        "candidates.parquet",
                        "decisions.parquet",
                        "spans.parquet",
                    }
                ):
                    raise ValueError(f"Completed partition binding differs: {slot}")
                inventory["completed_slots"].append(slot)
                for key in (
                    "records",
                    "input_tokens",
                    "candidate_records",
                    "candidate_tokens",
                    "unresolved_documents",
                ):
                    inventory[key] += summary[key]
            elif files:
                inventory["unfinished_slots_with_files"].append(slot)
            for path in files:
                if (
                    path.is_symlink()
                    or path.resolve() != root.resolve() / path.relative_to(root)
                ):
                    raise ValueError("Unexpected symlink in worker output")
                name = str(path.relative_to(root))
                info = tarfile.TarInfo(name)
                info.size = path.stat().st_size
                info.mode = 0o600
                with path.open("rb") as f:
                    reader = HashingReader(f)
                    archive.addfile(info, reader)
                    if f.read(1):
                        raise ValueError("Output grew while archiving")
                entry = {"bytes": info.size, "sha256": reader.digest.hexdigest()}
                if old is not None and old["files"].get(name) != entry:
                    raise ValueError(
                        f"Saved file differs from final worker inventory: {name}"
                    )
                if (
                    finished
                    and path.name in summary["files"]
                    and path.parent == directory
                ):
                    if entry["sha256"] != summary["files"][path.name]:
                        raise ValueError(f"Export checksum mismatch: {name}")
                inventory["files"][name] = entry
            if finished and any(
                str((directory / name).relative_to(root)) not in inventory["files"]
                for name in summary["files"]
            ):
                raise ValueError(f"Missing completed export: {slot}")
    if old is not None and inventory["files"] != old["files"]:
        raise ValueError("Worker inventory contains missing or additional files")
    if summary_path.exists() and summary_path.is_symlink():
        raise ValueError("Unexpected summary symlink")
    temporary.replace(destination)
    return inventory


def build(root, expected_sha, volume, ledger, output_root=None):
    root = Path(root)
    manifest_path = root / "modal-handoff/manifest.json"
    if sha(manifest_path) != expected_sha:
        raise ValueError("Cloud handoff changed")
    manifest = read(manifest_path)
    plan_path = root / relative(manifest["plan_relative"])
    if sha(plan_path) != manifest["plan_sha256"]:
        raise ValueError("Original scoring plan changed")
    plan = read(plan_path)
    state = ledger_snapshot(ledger, len(manifest["jobs"]))
    if (state["launch"] or {}).get("manifest_sha256") != expected_sha:
        raise ValueError("GPU launch does not match this handoff")
    folder = Path(output_root) if output_root is not None else root / PREFIX
    folder.mkdir(exist_ok=True)
    binding = {
        "manifest_sha256": expected_sha,
        "recovery_code_sha256": sha(__file__),
        "ledger": state,
    }
    binding_path = folder / "binding.json"
    if binding_path.exists() and read(binding_path) != binding:
        raise ValueError("Recovery snapshot changed; preserve previous bundles")
    write(binding_path, binding)
    catalog = {
        "manifest_sha256": expected_sha,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workers": {},
        "complete": False,
    }
    for worker in range(len(manifest["jobs"])):
        index_path = folder / f"worker-{worker}.json"
        archive_path = folder / f"worker-{worker}.tar.gz"
        checkpoint = folder / f"worker-{worker}-bundle.json"
        if checkpoint.exists():
            entry = read(checkpoint)
            if (
                sha(index_path) != entry["inventory_sha256"]
                or sha(archive_path) != entry["archive_sha256"]
            ):
                raise ValueError("Recovery bundle changed")
            inventory = read(index_path)
        else:
            print(f"Bundling saved worker {worker} checkpoints (CPU only)", flush=True)
            inventory = bundle_worker(
                root, manifest, plan, expected_sha, worker, state, archive_path
            )
            write(index_path, inventory)
            entry = {
                "inventory_sha256": sha(index_path),
                "archive_sha256": sha(archive_path),
                "archive_bytes": archive_path.stat().st_size,
                "completed_parts": len(inventory["completed_slots"]),
                **{
                    k: inventory[k]
                    for k in (
                        "input_tokens",
                        "candidate_tokens",
                        "unresolved_documents",
                    )
                },
            }
            if (
                ledger_snapshot(ledger, len(manifest["jobs"])) != state
                or sha(manifest_path) != expected_sha
            ):
                raise ValueError("Run changed during recovery; stop and inspect")
            write(checkpoint, entry)
            volume.commit()
        catalog["workers"][str(worker)] = entry
        print(json.dumps({"worker": worker, **entry}), flush=True)
    catalog["complete"] = True
    catalog["scope"] = (
        "Stopped-run recovery only. Candidate tokens require CPU raw-response/source audit and final selection/deduplication."
    )
    write(folder / "catalog.json", catalog)
    volume.commit()
    print("Recovery bundles complete; fetch directly to Marlowe.", flush=True)


def extract_checked(archive_path, destination, inventory, allowed):
    """Extract only allocated regular files, with byte and SHA-256 verification."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    seen = set()
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            name = member.name
            path = relative(name)
            expected = inventory["files"].get(name)
            if (
                not member.isfile()
                or name in seen
                or expected is None
                or member.size != expected["bytes"]
                or not any(path.is_relative_to(p) for p in allowed)
            ):
                raise ValueError("Unexpected archive member")
            seen.add(name)
            target = destination / path
            if target.is_symlink() or not target.resolve().is_relative_to(
                destination.resolve()
            ):
                raise ValueError("Extraction path escapes destination")
            if target.exists() and (
                target.stat().st_size != expected["bytes"]
                or sha(target) != expected["sha256"]
            ):
                raise ValueError(f"Conflicting local output; preserve both: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".recovering")
            if temporary.is_symlink():
                raise ValueError("Unsafe temporary extraction path")
            digest = hashlib.sha256()
            with archive.extractfile(member) as source, temporary.open("wb") as output:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
            if digest.hexdigest() != expected["sha256"]:
                temporary.unlink()
                raise ValueError("Extracted file checksum mismatch")
            temporary.replace(target)
    if seen != set(inventory["files"]):
        raise ValueError("Incomplete recovery archive")


def fetch(manifest_path):
    import modal

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.modal_youtube import common as c, transfer

    manifest, _, plan = c.validate_manifest(manifest_path)
    transfer.check_held(plan)
    volume = modal.Volume.from_name(RECOVERY_VOLUME)
    deadline = time.monotonic() + TIMEOUT + 600
    while True:
        try:
            catalog = json.loads(b"".join(volume.read_file(f"/{PREFIX}/catalog.json")))
            break
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Recovery bundles are not ready; inspect the CPU recovery job"
                )
            print(
                "Waiting for CPU recovery bundles; no GPU is being requested.",
                flush=True,
            )
            time.sleep(60)
    if (
        catalog["complete"] is not True
        or catalog["manifest_sha256"] != sha(manifest_path)
        or set(catalog["workers"]) != {str(i) for i in range(len(manifest["jobs"]))}
    ):
        raise ValueError("Recovery catalog differs from handoff")
    target = manifest_path.parent / "returned"
    cache = manifest_path.parent / "recovery-archives"
    cache.mkdir(exist_ok=True)
    output = Path(plan["output_dir"]).relative_to(manifest["mount"])
    for worker, entry in catalog["workers"].items():
        raw = b"".join(volume.read_file(f"/{PREFIX}/worker-{worker}.json"))
        if hashlib.sha256(raw).hexdigest() != entry["inventory_sha256"]:
            raise ValueError("Recovery inventory checksum mismatch")
        inventory = json.loads(raw)
        if inventory["manifest_sha256"] != sha(manifest_path) or inventory[
            "worker"
        ] != int(worker):
            raise ValueError("Recovery worker identity changed")
        archive = cache / f"worker-{worker}.tar.gz"
        if not archive.exists() or sha(archive) != entry["archive_sha256"]:
            print(
                f"Downloading worker {worker}: {entry['archive_bytes'] / 1e9:.2f} GB",
                flush=True,
            )
            temporary = archive.with_suffix(".download")
            with temporary.open("wb") as f:
                for block in volume.read_file(f"/{PREFIX}/{archive.name}"):
                    f.write(block)
            if (
                temporary.stat().st_size != entry["archive_bytes"]
                or sha(temporary) != entry["archive_sha256"]
            ):
                raise ValueError("Recovery download checksum mismatch")
            temporary.replace(archive)
        allowed = [
            output / "parts" / f"{slot:05d}" for slot in manifest["jobs"][int(worker)]
        ]
        extract_checked(archive, target, inventory, allowed)
        write(target / f"worker-{worker}.json", inventory)
        print(
            f"Recovered worker {worker}: {len(inventory['completed_slots'])} completed parts plus partial checkpoints",
            flush=True,
        )
    transfer.check_held(plan)
    write(
        target / "downloaded.json",
        {
            "manifest_sha256": sha(manifest_path),
            "workers": sorted(map(int, catalog["workers"])),
            "recovered_after_missing_coordinator": True,
        },
    )
    write(manifest_path.parent / "recovery-summary.json", catalog)
    print(
        "Fetch complete. Run the existing CPU import; leave the GPU array held.",
        flush=True,
    )


def launch():
    import modal

    active = json.loads(
        subprocess.run(
            [sys.executable, "-m", "modal", "container", "list", "--json"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    if active:
        raise RuntimeError("Modal containers are still active; inspect before recovery")
    volume = modal.Volume.from_name("security-corpus-youtube-modal-v1")
    recovery_volume = modal.Volume.from_name(RECOVERY_VOLUME, create_if_missing=True)
    raw = b"".join(volume.read_file("/modal-handoff/manifest.json"))
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    image = modal.Image.debian_slim(python_version=python_version).add_local_file(
        __file__, "/opt/recover.py"
    )
    app = modal.App("security-corpus-youtube-recovery")

    @app.function(
        image=image,
        volumes={"/data": volume.read_only(), "/recovery": recovery_volume},
        timeout=TIMEOUT,
        cpu=(2, 2),
        memory=(8192, 8192),
        retries=0,
        max_containers=1,
        include_source=False,
        serialized=True,
    )
    def recover(expected_sha):
        import sys

        sys.path.insert(0, "/opt")
        import recover as recovery

        manifest = recovery.read("/data/modal-handoff/manifest.json")
        recovery.build(
            recovery.Path("/data"),
            expected_sha,
            modal.Volume.from_name(recovery.RECOVERY_VOLUME),
            modal.Dict.from_name(manifest["ledger"]),
            output_root=recovery.Path("/recovery") / recovery.PREFIX,
        )

    with modal.enable_output(), app.run(detach=True):
        print("Recovery CPU app:", app.app_id, flush=True)
        recover.remote(hashlib.sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bundle", "build", "fetch"))
    parser.add_argument("--expected-sha")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(os.environ.get("CYBEROPEN_WORK_DIR", Path(__file__).resolve().parents[1] / "data"))
        / "curation/modal-youtube-v1/manifest.json",
    )
    args = parser.parse_args()
    if args.command == "bundle":
        launch()
    elif args.command == "fetch":
        fetch(args.manifest)
    else:
        import modal

        manifest = read("/data/modal-handoff/manifest.json")
        build(
            Path("/data"),
            args.expected_sha,
            modal.Volume.from_name(RECOVERY_VOLUME),
            modal.Dict.from_name(manifest["ledger"]),
            output_root=Path("/recovery") / PREFIX,
        )


if __name__ == "__main__":
    main()
