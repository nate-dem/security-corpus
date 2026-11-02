"""Resume Marlowe YouTube filtering after the stopped Modal run is recovered.

No Modal API calls. The unchanged pool owns model execution, partition resume,
eight-GPU concurrency, and stopping on a wave with no completed-part progress.
This supervisor waits for the running CPU recovery/finalization before releasing
the handoff's held array. It continues until every YouTube partition is attempted.
"""

import argparse
import json
import logging
from pathlib import Path
import subprocess
import time

from scripts.curation_pool import run as pool
from scripts.modal_youtube import common, transfer
from scripts.youtube.download import _directory_lock, _write_json
from scripts.youtube.profile import _sha256
from scripts.youtube_finalize.pipeline import recovered_slots

HERE = Path(__file__).resolve().parent
LOG = logging.getLogger(__name__)


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def controller_busy(root):
    """Probe the original pool lock without depending on optional launchers."""
    try:
        with _directory_lock(root):
            return False
    except RuntimeError as exc:
        if isinstance(exc.__cause__, BlockingIOError):
            return True
        raise


def verify_previous(previous, manifest_path, path):
    """Read only small, checksum-bound completion evidence on the login node."""
    target = previous / "summary.json"
    if not target.exists() or read(target).get("complete") is not True:
        return False
    result = read(target)
    imported = read(previous / "import-audit.json")
    dedup_plan = read(previous / "dedup/plan.json")
    audit = read(previous / "audit/review/summary.json")
    frozen = read(previous / "audit/frozen.json")
    require(
        imported["complete"]
        and audit["audit_complete"]
        and imported["manifest_sha256"] == _sha256(manifest_path)
        and imported["plan_sha256"] == _sha256(path) == audit["plan_sha256"]
        and imported["audit_summary_sha256"]
        == _sha256(previous / "audit/review/summary.json")
        == result["audit_summary_sha256"]
        and dedup_plan["import_audit_sha256"] == _sha256(previous / "import-audit.json")
        and result["dedup_plan_sha256"] == _sha256(previous / "dedup/plan.json")
        and result["dedup_summary_sha256"] == _sha256(previous / "dedup/summary.json")
        and result["corpus_manifest_sha256"]
        == _sha256(previous / "corpus-manifest.json"),
        "Previous recovery/finalization evidence changed; do not release GPUs",
    )
    manifest = read(manifest_path)
    expected = set(manifest["completed_slots"]) | recovered_slots(manifest_path)
    require(
        set(imported["completed_slots"]) == expected
        and {p["slot"] for p in frozen["parts"]} == expected,
        "Previous audit did not include all recovered and local completions",
    )
    require(
        read(previous / "audit/review/frozen.json") == frozen,
        "Frozen audit differs from verified copy",
    )
    require(
        _sha256(previous / "audit/review/frozen.json") == audit["files"]["frozen.json"],
        "Frozen audit checksum changed",
    )
    plan = read(path)
    for entry in frozen["parts"]:
        require(
            _sha256(pool.single.folder(plan, entry["slot"]) / "summary.json")
            == entry["summary_sha256"],
            "An imported completed partition changed",
        )
    return True


def controller_state(path):
    target = path.parent / "pool/controller.json"
    state = read(target) if target.exists() else {}
    if state:
        require(
            state["plan_sha256"] == _sha256(path)
            and state["code"] == pool.code_hashes(),
            "Original worker-pool bindings changed",
        )
        require(
            not state.get("submission_in_progress"),
            "Original pool has an uncertain submission; inspect squeue/controller.json",
        )
    return state


def release_original(manifest, path, directory, state):
    """Release only the known held handoff array, with an idempotent intent log."""
    if state.get("released"):
        return
    held = str(manifest["held_array"])
    require(held.isdigit(), "Handoff must name a numeric Slurm array ID")
    old = controller_state(path)
    rows = [r for r in transfer.queue() if r[1] == "youtube-pool"]
    require(
        all(r[0].split("_")[0] == held for r in rows),
        "Unexpected YouTube GPU array exists; inspect before resuming",
    )
    if rows:
        require(
            old.get("active_gpu") == held and len(rows) <= 8,
            "Held array differs from original controller",
        )
        held_rows = [r for r in rows if r[2:] == ["PENDING", "JobHeldUser"]]
        if len(held_rows) != len(rows):
            require(
                state.get("release_intent") == held and not held_rows,
                "Original array is not wholly held; inspect its state",
            )
        else:
            state["release_intent"] = held
            _write_json(directory / "state.json", state)
            pool.command(["scontrol", "release", held])
            LOG.info(
                "Released original held Marlowe array %s; completed partitions will be skipped",
                held,
            )
    else:
        LOG.info(
            "Original held array %s is no longer queued; the pool will allocate unfinished work",
            held,
        )
    state["released"] = True
    _write_json(directory / "state.json", state)


def registered_active(root, user):
    registry = root / "submitted-jobs.txt"
    known = set(registry.read_text().splitlines()) if registry.exists() else set()
    return pool.queued(user) & known


def launch_final(directory, output, user, state):
    marker = output / "summary.json"
    if marker.exists() and read(marker).get("complete"):
        result = read(marker)
        require(
            not result["unprocessed_slots"]
            and result["completed_parts"] == result["planned_parts"],
            "Final output is only a subset; use a separate full-run output",
        )
        LOG.info("Full-run selection already finalized: %s", marker)
        return
    if state.get("final_job") in pool.queued(user):
        LOG.info("Final CPU job %s is already queued/running", state["final_job"])
        return
    require(
        not state.get("final_submission_in_progress"),
        "Final CPU submission outcome uncertain; inspect squeue and resume state",
    )
    state["final_submission_in_progress"] = True
    _write_json(directory / "state.json", state)
    # CLI limit overrides the one-hour GPU environment setting.
    job = pool.job_id(
        pool.command(
            ["sbatch", "--parsable", "--time=04:00:00", str(HERE / "final.sbatch")]
        )
    )
    state.update(
        final_job=job, final_submission_in_progress=False, status="final_cpu_submitted"
    )
    _write_json(directory / "state.json", state)
    LOG.info("Submitted full-run CPU audit/dedup job %s; final counts: %s", job, marker)


def execute(path, manifest_path, previous, output, user, poll=60):
    require(poll >= 10, "Poll interval must be at least ten seconds")
    path, manifest_path, previous, output = (
        Path(p).resolve() for p in (path, manifest_path, previous, output)
    )
    manifest, checked_path, _ = common.validate_manifest(manifest_path)
    require(checked_path == path, "Resume plan differs from recovered handoff")
    plan = pool.single.load_plan(path)
    for protected in (
        previous,
        path.parent,
        Path(plan["input_dir"]).resolve(),
        manifest_path.parent,
        HERE.parents[1],
    ):
        require(
            output != protected
            and not output.is_relative_to(protected)
            and not protected.is_relative_to(output),
            "Keep full-run output separate from existing inputs and subset",
        )
    directory = path.parent / "pool/after-modal-resume"
    directory.mkdir(parents=True, exist_ok=True)
    binding = {
        "plan_sha256": _sha256(path),
        "manifest_sha256": _sha256(manifest_path),
        "previous": str(previous),
        "output": str(output),
        "code": {
            p.name: _sha256(p)
            for p in sorted(HERE.iterdir())
            if p.suffix in {".py", ".sh", ".sbatch"}
        },
        "workers": 8,
    }
    with _directory_lock(directory):
        target = directory / "state.json"
        state = read(target) if target.exists() else {"binding": binding}
        require(
            state["binding"] == binding,
            "Resume settings/code changed; preserve the previous run",
        )
        last_wait = None
        while True:
            prior_jobs = registered_active(previous, user)
            ready = verify_previous(previous, manifest_path, path)
            if ready and not prior_jobs:
                break
            if (
                not ready
                and (previous / "submitted-jobs.txt").exists()
                and not prior_jobs
            ):
                raise RuntimeError(
                    "Previous CPU finalization did not complete. Resume youtube_finalize/finish.sh first; GPUs remain held."
                )
            message = "Waiting for existing recovery download and CPU finalization; no GPU released or submitted"
            if message != last_wait:
                LOG.info(message)
                last_wait = message
            state.update(status="waiting_for_recovery")
            _write_json(target, state)
            time.sleep(poll)
        release_original(manifest, path, directory, state)
        last_wait = None
        while True:
            if controller_busy(path.parent):
                message = "Original controller is active; waiting without submitting another GPU pool"
            else:
                old = controller_state(path)
                if not old.get("active_gpu") and old.get("stop_reason", "").startswith(
                    "no completed-part"
                ):
                    raise RuntimeError(old["stop_reason"])
                active = registered_active(path.parent, user)
                foreign = active - {old.get("active_gpu")}
                if foreign:
                    message = (
                        f"Waiting for previous scoring/audit jobs: {sorted(foreign)}"
                    )
                elif old.get("active_gpu") in active:
                    # Reconnect the original pool to its active wave, even if
                    # all summaries have appeared but the array has not exited.
                    message = None
                else:
                    pending, finished = pool.inventory(path, plan)
                    state.update(
                        finished_parts=len(finished), pending_parts=len(pending)
                    )
                    _write_json(target, state)
                    if not pending:
                        launch_final(directory, output, user, state)
                        return 0
                    message = None
                if message is None:
                    state["status"] = "filtering"
                    _write_json(target, state)
                    LOG.info(
                        "Continuing original Marlowe pool: at most 8 GPUs, unfinished YouTube only"
                    )
                    # The original pool locks its root against every other
                    # launcher and stops failed/no-progress waves. Repeat its
                    # bounded windows until all partitions have been attempted.
                    status = pool.controller(path, user, 36, 8, poll)
                    if status:
                        raise RuntimeError(
                            f"Worker pool stopped with status {status}; inspect GPU logs"
                        )
                    last_wait = None
                    continue
            if message != last_wait:
                LOG.info(message)
                last_wait = message
            time.sleep(poll)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.status:
        for target in (
            args.plan.parent / "pool/after-modal-resume/state.json",
            args.plan.parent / "pool/progress.json",
            args.output / "audit/progress.json",
            args.output / "progress.json",
            args.output / "summary.json",
        ):
            if target.exists():
                print(f"{target}\n{target.read_text()}")
        return 0
    try:
        return execute(
            args.plan, args.manifest, args.previous, args.output, args.user, args.poll
        )
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        LOG.error("Stopped: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
