"""Cloud handoff mechanics; no changes to corpus policy or frozen scorer code."""

import hashlib
import heapq
import json
from pathlib import Path
import time

WORK_SECONDS = 23400  # 6.5 hours of worker execution, including model setup.
CALL_SECONDS = 25200  # Seven-hour outer function limit, including final export.
STARTUP_SECONDS = 900
MAX_WORKERS = 8
HOURLY_DOLLARS = 3600 * (0.001097 + 8 * 0.0000131 + 96 * 0.00000222)
PROJECT = Path(__file__).resolve().parents[2]
CODE_DIRS = (
    "curation",
    "curation_single",
    "youtube",
    "youtube_filter",
    "web_corpora",
    "modal_youtube",
)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def allocate(slots, parts):
    if (
        not slots
        or len(set(slots)) != len(slots)
        or any(type(n) is not int or not 0 <= n < len(parts) for n in slots)
    ):
        raise ValueError("Need unique unfinished slots")
    count = min(MAX_WORKERS, len(slots))
    groups, queue = [[] for _ in range(count)], [(0, n) for n in range(count)]
    heapq.heapify(queue)
    for slot in sorted(slots, key=lambda n: (-parts[n]["tokens"], n)):
        if Path(parts[slot]["path"]).parts[1] != "youtube-commons":
            raise ValueError("Modal allocation must contain YouTube only")
        tokens, worker = heapq.heappop(queue)
        groups[worker].append(slot)
        heapq.heappush(queue, (tokens + parts[slot]["tokens"], worker))
    return groups


def safe_relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Unsafe artifact path")
    return path


def code_files(project=PROJECT):
    files = []
    for directory in [project / "src"] + [project / "scripts" / d for d in CODE_DIRS]:
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(directory)
            if any(p.startswith(".") or p == "__pycache__" for p in relative.parts):
                continue
            if path.is_file() and path.suffix in {
                ".py",
                ".sh",
                ".sbatch",
                ".txt",
                ".json",
                ".tiktoken",
            }:
                files.append(path)
    return files


def code_hashes():
    return {str(p.relative_to(PROJECT)): sha(p) for p in code_files()}


def validate_manifest(path):
    manifest = read(path)
    if manifest["version"] != "modal-youtube-v1":
        raise ValueError("Unknown handoff version")
    if manifest["code"] != code_hashes():
        raise ValueError("Handoff code changed; preserve this allocation")
    root = Path(manifest["mount"])
    plan_path = root / safe_relative(manifest["plan_relative"])
    if sha(plan_path) != manifest["plan_sha256"]:
        raise ValueError("Original scoring plan changed")
    plan = read(plan_path)
    if Path(plan["output_dir"]) / "plan.json" != plan_path:
        raise ValueError("Original scoring output path changed")
    if manifest["jobs"] != allocate(manifest["slots"], plan["parts"]):
        raise ValueError("Handoff allocation changed")
    if any(n in manifest["completed_slots"] for n in manifest["slots"]):
        raise ValueError("Completed work must not be allocated")
    if (
        manifest["work_seconds"] != WORK_SECONDS
        or manifest["call_seconds"] != CALL_SECONDS
    ):
        raise ValueError("Worker time allowance changed")
    expected_ceiling = (
        len(manifest["jobs"]) * (CALL_SECONDS + STARTUP_SECONDS) / 3600 * HOURLY_DOLLARS
    )
    if (
        manifest["estimated_worker_ceiling_dollars"] != expected_ceiling
        or expected_ceiling > 300
        or manifest["model"] != plan["model"]
    ):
        raise ValueError("Worker allowance exceeds $300")
    return manifest, plan_path, plan


def load_scoring_plan(manifest, plan_path):
    """Validate the unchanged plan through Modal's verified Volume mount alias.

    Modal mounts a Volume using a symlink. The frozen scorer binds the resolved
    plan filename to its original Marlowe filename. Supply that original name
    only after proving that the mount alias resolves to the exact expected file.
    This does not patch pathlib globally, rewrite the plan, or skip scorer checks.
    """
    from scripts.curation_single import run

    mount = Path(manifest["mount"])
    relative = safe_relative(manifest["plan_relative"])
    original = mount / relative
    if not mount.is_absolute() or Path(plan_path) != original:
        raise ValueError("Scoring plan is outside its original mount binding")
    physical = mount.resolve(strict=True) / relative
    if original.resolve(strict=True) != physical or original.is_symlink():
        raise ValueError("Unexpected symlink beneath the Volume mount")
    if sha(original) != manifest["plan_sha256"]:
        raise ValueError("Original scoring plan changed")

    class MountedPlanPath(type(Path())):
        def resolve(self, strict=False):
            resolved = super().resolve(strict=strict)
            if Path(self) != original or resolved != physical:
                raise ValueError("Scoring plan mount binding changed")
            return original

    # The subclass is used only for this call. All other paths and all returned
    # plan contents retain their ordinary behavior and original bytes.
    return run.load_plan(MountedPlanPath(original))


def part_files(directory):
    return sorted(
        p for p in directory.rglob("*") if p.is_file() and p.name != ".download.lock"
    )


def verify_files(root, files):
    for relative, expected in files.items():
        path = root / safe_relative(relative)
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"File escapes artifact root: {relative}")
        if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise ValueError(f"Artifact changed: {relative}")


def launch_wave(
    worker_count, ledger, spawn, manifest_sha, clock=time.monotonic, wait=time.sleep
):
    """One paid wave, with startup fan-out guarded by real completed work."""
    if not 1 <= worker_count <= MAX_WORKERS:
        raise ValueError("Worker count outside budget")
    if not ledger.put(
        "launch",
        {"manifest_sha256": manifest_sha, "workers": worker_count},
        skip_if_exists=True,
    ):
        raise RuntimeError(
            "This allowance was already launched; inspect status instead of resubmitting"
        )
    calls, results = {}, {}
    try:
        calls[0] = spawn(0)
        deadline = clock() + 1800
        while ledger.get("startup_ok", None) is not True:
            done = ledger.get("worker-done-0", None)
            if done is not None:
                raise RuntimeError(
                    f"First worker stopped before finishing a part: {done}"
                )
            if clock() >= deadline:
                raise RuntimeError(
                    "First partition did not finish within 30 minutes; other GPUs were not launched"
                )
            wait(10)
        for index in range(1, worker_count):
            calls[index] = spawn(index)
        for index, call in calls.items():
            try:
                results[str(index)] = call.get(
                    timeout=CALL_SECONDS + STARTUP_SECONDS + 60
                )
            except Exception as exc:
                results[str(index)] = {"error": str(exc)}
    except Exception:
        for call in calls.values():
            call.cancel(terminate_containers=True)
        raise
    finally:
        ledger.put(
            "coordinator_done", {"manifest_sha256": manifest_sha, "results": results}
        )
    return results
