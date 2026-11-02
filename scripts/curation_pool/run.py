"""Bounded Slurm submissions around the unchanged, frozen single-pass scorer.

The login-node controller only reads small manifests and polls Slurm. Model
loading, source scans, output validation and the final audit run on compute nodes.
"""

import argparse
import heapq
import json
import logging
from pathlib import Path
import subprocess
import time
import uuid

from scripts.curation_single import run as single
from scripts.youtube.download import _directory_lock, _write_json
from scripts.youtube.profile import _sha256

HERE = Path(__file__).resolve().parent


def code_hashes():
    return {
        p.name: _sha256(p)
        for p in sorted(HERE.iterdir())
        if p.suffix in {".py", ".sh", ".sbatch"}
    }


def inventory(path, plan):
    """Scheduling metadata only; workers and final audit validate file contents."""
    digest = _sha256(path)
    pending, finished = [], []
    for slot, part in enumerate(plan["parts"]):
        if part["path"].split("/")[1] != "youtube-commons":
            continue
        summary = single.folder(plan, slot) / "summary.json"
        state = json.loads(summary.read_text()) if summary.exists() else {}
        if state and state["plan_sha256"] != digest:
            raise ValueError("Checkpoint belongs to another plan")
        # run_slot already retries malformed output once. Do not endlessly retry
        # unresolved labels at the expense of previously unseen source text.
        if state.get("workflow_finished"):
            finished.append(state)
        else:
            pending.append(slot)
    return pending, finished


def allocate(slots, plan, workers):
    if not 1 <= workers <= 8 or len(slots) != len(set(slots)):
        raise ValueError("Use 1-8 workers and unique source slots")
    jobs = [[] for _ in range(min(workers, len(slots)))]
    heap = [(0, i) for i in range(len(jobs))]
    heapq.heapify(heap)
    for slot in slots:
        if (
            not 0 <= slot < len(plan["parts"])
            or "/youtube-commons/" not in plan["parts"][slot]["path"]
        ):
            raise ValueError("Pool is restricted to YouTube partitions")
        tokens, i = heapq.heappop(heap)
        jobs[i].append(slot)
        heapq.heappush(heap, (tokens + plan["parts"][slot]["tokens"], i))
    return jobs


def checked_work(path, plan, work_path):
    work = json.loads(work_path.read_text())
    if work["plan_sha256"] != _sha256(path) or work["code"] != code_hashes():
        raise ValueError("Worker allocation/code binding changed")
    jobs = work["jobs"]
    if not 1 <= len(jobs) <= 8 or any(not j for j in jobs):
        raise ValueError("Invalid pool size or empty worker")
    slots = [slot for job in jobs for slot in job]
    allocate(slots, plan, len(jobs))  # Validate identities, scope and uniqueness.
    return work


def work(path, work_path, index, seconds):
    if not 0 < seconds <= 4 * 3600:
        raise ValueError("Worker budget must fit the preempt partition")
    plan = single.load_plan(path)
    allocation = checked_work(path, plan, work_path)
    if not 0 <= index < len(allocation["jobs"]):
        raise ValueError("Worker outside allocation")
    started = time.monotonic()
    factory = single.runtime.ResidentModel()
    initialized, status = False, 0
    for slot in allocation["jobs"][index]:
        saved = single.checked_summary(path, plan, slot)
        if saved and saved.get("workflow_finished"):
            continue
        if time.monotonic() - started >= seconds:
            logging.info("Worker budget reached; remaining partitions stay pending")
            break
        if not initialized:
            single.cuda_preflight.configure_and_check()
            single.cuda_preflight.check_sampler()
            initialized = True
        logging.info("Worker %s scoring YouTube slot %s", index, slot)
        status = max(status, single.run_slot(path, plan, slot, factory))
    return status


def command(args):
    return subprocess.run(
        args, text=True, capture_output=True, check=True
    ).stdout.strip()


def queued(user):
    return {
        line.strip().split("_")[0]
        for line in command(["squeue", "-h", "-u", user, "-o", "%i"]).splitlines()
        if line.strip()
    }


def job_id(value):
    result = value.split(";")[0].strip()
    if not result.isdigit():
        raise ValueError("No unambiguous Slurm job ID; inspect squeue before retrying")
    return result


def remember(root, identifier):
    # The original launcher checks this same registry, preventing accidental
    # concurrent submissions through that older entry point.
    with (root / "submitted-jobs.txt").open("a") as f:
        f.write(identifier + "\n")
        f.flush()


def controller(path, user, hours, workers, poll):
    if not 0 < hours <= 36 or not 1 <= workers <= 8 or poll < 10:
        raise ValueError("Use hours in (0,36], 1-8 workers, poll >= 10 seconds")
    plan = single.load_plan(path)
    root = Path(plan["output_dir"])
    pool = root / "pool"
    pool.mkdir(exist_ok=True)
    # Also block the legacy launcher's plan lock while this controller owns the
    # run, including brief gaps between worker waves.
    with _directory_lock(root), _directory_lock(pool):
        state_path = pool / "controller.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state and (
            state["plan_sha256"] != _sha256(path) or state["code"] != code_hashes()
        ):
            raise ValueError("Controller binding changed; preserve previous state")
        state.update(plan_sha256=_sha256(path), code=code_hashes())
        if state.get("submission_in_progress"):
            raise ValueError(
                "Earlier submission was interrupted; inspect squeue and controller.json before resuming"
            )
        active = queued(user)
        registry = root / "submitted-jobs.txt"
        known = set(registry.read_text().splitlines()) if registry.exists() else set()
        foreign = (active & known) - {state.get("active_gpu")}
        if foreign:
            raise ValueError(
                f"Previous scoring/audit jobs still active: {sorted(foreign)}"
            )
        deadline = time.monotonic() + hours * 3600
        reason = "submission window reached"
        while True:
            if state.get("active_gpu"):
                identifier = state["active_gpu"]
                if identifier in queued(user):
                    time.sleep(poll)
                    continue
                # squeue absence alone does not establish success. Progress comes
                # from per-part scorer checkpoints, independently audited later.
                pending, finished = inventory(path, plan)
                logging.info(
                    "Array %s ended: %s finished partitions, %s unfinished",
                    identifier,
                    len(finished),
                    len(pending),
                )
                state.pop("active_gpu")
                state["last_gpu"] = identifier
                if len(finished) <= state["finished_before_wave"] and pending:
                    reason = f"no completed-part progress from array {identifier}; inspect its logs"
                    _write_json(state_path, state)
                    break
                _write_json(state_path, state)
            pending, finished = inventory(path, plan)
            progress = {
                "finished_parts": len(finished),
                "pending_parts": len(pending),
                "input_tokens": sum(p["input_tokens"] for p in finished),
                "candidate_tokens": sum(p["candidate_tokens"] for p in finished),
                "unresolved_documents": sum(
                    p["unresolved_documents"] for p in finished
                ),
                "scope": "Scheduling snapshot, not an integrity or quality audit.",
            }
            _write_json(pool / "progress.json", progress)
            if not pending:
                reason = "all YouTube partitions attempted"
                break
            if time.monotonic() >= deadline:
                break
            jobs = allocate(pending, plan, workers)
            work_path = pool / f"work-{uuid.uuid4().hex}.json"
            _write_json(
                work_path,
                {"plan_sha256": _sha256(path), "code": code_hashes(), "jobs": jobs},
            )
            state.update(
                submission_in_progress=True,
                work_list=str(work_path),
                finished_before_wave=len(finished),
            )
            _write_json(state_path, state)
            try:
                identifier = job_id(
                    command(
                        [
                            "sbatch",
                            "--parsable",
                            f"--array=0-{len(jobs) - 1}%{len(jobs)}",
                            str(HERE / "gpu.sbatch"),
                            str(path),
                            str(work_path),
                        ]
                    )
                )
            except subprocess.CalledProcessError as exc:
                # A nonzero sbatch response is an explicit rejection, so a later
                # invocation can retry. Unexpected/ambiguous responses stay locked.
                state["submission_in_progress"] = False
                _write_json(state_path, state)
                raise RuntimeError(
                    f"Slurm rejected {len(jobs)} workers: {exc.stderr.strip()}"
                ) from exc
            state.update(active_gpu=identifier, submission_in_progress=False)
            _write_json(state_path, state)
            remember(root, identifier)
            logging.info(
                "Submitted array %s: %s workers, %s YouTube partitions. GPUs <= %s.",
                identifier,
                len(jobs),
                len(pending),
                len(jobs),
            )
        state.update(stop_reason=reason)
        _write_json(state_path, state)
        logging.info("Controller stopped: %s", reason)
        identifier = job_id(
            command(["sbatch", "--parsable", str(single.HERE / "audit.sbatch")])
        )
        remember(root, identifier)
        state["audit_job"] = identifier
        _write_json(state_path, state)
        logging.info("Submitted CPU audit %s; reports: %s/review", identifier, root)
        return 2 if reason.startswith("no completed-part") else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("run")
    c.add_argument("--plan", type=Path, required=True)
    c.add_argument("--user", required=True)
    c.add_argument("--hours", type=float, default=12)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--poll", type=int, default=60)
    w = sub.add_parser("work")
    w.add_argument("--plan", type=Path, required=True)
    w.add_argument("--work-list", type=Path, required=True)
    w.add_argument("--job-index", type=int, required=True)
    w.add_argument("--seconds", type=int, default=13500)
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.command == "work":
        return work(args.plan, args.work_list, args.job_index, args.seconds)
    return controller(args.plan, args.user, args.hours, args.workers, args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
