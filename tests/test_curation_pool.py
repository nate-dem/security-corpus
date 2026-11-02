import json
from pathlib import Path
import subprocess

import pytest

from scripts.curation_pool import run as pool
from scripts.curation_single import run as single
from scripts.youtube.download import _write_json
from scripts.youtube.profile import _sha256
from test_curation_single import single_env as _single_env
from test_curation_production import production_env as _production_env

single_env = _single_env
production_env = _production_env


def work_file(path, jobs):
    target = path.parent / "pool-work.json"
    _write_json(
        target, {"plan_sha256": _sha256(path), "code": pool.code_hashes(), "jobs": jobs}
    )
    return target


def test_full_youtube_inventory_fits_eight_jobs():
    plan = {
        "parts": [
            {"path": f"shards/youtube-commons/{i}/0.parquet", "tokens": (i + 1) * 17}
            for i in range(2205)
        ]
    }
    jobs = pool.allocate(list(range(2205)), plan, workers=8)
    assert len(jobs) == 8
    assert sorted(i for job in jobs for i in job) == list(range(2205))
    assert all(len(job) > 8 for job in jobs)
    with pytest.raises(ValueError, match="unique"):
        pool.allocate([0, 0], plan, workers=8)


def test_worker_keeps_one_model_and_preserves_frozen_plan(single_env):
    path, plan, loads, calls = single_env
    frozen = path.read_bytes()
    pending, _ = pool.inventory(path, plan)
    allocation = work_file(path, [pending[:3]])
    assert pool.work(path, allocation, 0, 13500) == 0
    assert len(loads) == 1 and calls == [3, 3, 3]
    assert pool.work(path, allocation, 0, 13500) == 0
    assert len(loads) == 1 and calls == [3, 3, 3]
    assert path.read_bytes() == frozen and single.load_plan(path) == plan
    _, finished = pool.inventory(path, plan)
    assert len(finished) == 3
    _write_json(
        allocation,
        {
            "plan_sha256": _sha256(path),
            "code": pool.code_hashes(),
            "jobs": [[pending[0]], [pending[0]]],
        },
    )
    with pytest.raises(ValueError, match="unique"):
        pool.work(path, allocation, 0, 13500)


def complete_one(path, plan, slot):
    folder = single.folder(plan, slot)
    folder.mkdir(parents=True, exist_ok=True)
    _write_json(
        folder / "summary.json",
        {
            "plan_sha256": _sha256(path),
            "workflow_finished": True,
            "complete": True,
            "input_tokens": plan["parts"][slot]["tokens"],
            "candidate_tokens": 0,
            "unresolved_documents": 0,
        },
    )


def test_controller_waits_then_submits_only_unfinished_work(single_env, monkeypatch):
    path, plan, _, _ = single_env
    frozen = path.read_bytes()
    submitted, polls, slept = [], {}, []

    def command(args):
        if args[0] == "sbatch":
            submitted.append(args)
            return str(100 + len(submitted))
        active = []
        for index, job in enumerate(submitted, 101):
            if not any(a.startswith("--array=") for a in job):
                continue
            polls[index] = polls.get(index, 0) + 1
            if polls[index] == 1:
                active.append(f"{index}_[0-7]")
            elif polls[index] == 2:
                manifest = json.loads(Path(job[-1]).read_text())
                slots = [i for group in manifest["jobs"] for i in group]
                # Wave one times out after two partitions. Wave two finishes.
                for slot in slots[:2] if index == 101 else slots:
                    complete_one(path, plan, slot)
        return "\n".join(active)

    monkeypatch.setattr(pool, "command", command)
    monkeypatch.setattr(pool.time, "sleep", lambda seconds: slept.append(seconds))
    assert pool.controller(path, "fixture", hours=12, workers=8, poll=60) == 0
    assert len(submitted) == 3  # Two GPU waves then a CPU audit.
    assert any(a == "--array=0-6%7" for a in submitted[0])
    assert any(a == "--array=0-4%5" for a in submitted[1])
    assert submitted[-1][-1].endswith("audit.sbatch")
    first = json.loads(Path(submitted[0][-1]).read_text())["jobs"]
    second = json.loads(Path(submitted[1][-1]).read_text())["jobs"]
    completed = [i for group in first for i in group][:2]
    assert not set(completed) & {i for group in second for i in group}
    assert slept == [60, 60]
    assert path.read_bytes() == frozen


def test_controller_stops_failed_waves_and_handles_submit_limit(
    single_env, monkeypatch
):
    path, _, _, _ = single_env
    submissions = []

    def command(args):
        if args[0] == "squeue":
            return ""
        submissions.append(args)
        return str(300 + len(submissions))

    monkeypatch.setattr(pool, "command", command)
    assert pool.controller(path, "fixture", hours=12, workers=8, poll=60) == 2
    assert len(submissions) == 2  # Failed GPU wave and audit, no repeated failure loop.
    state_path = path.parent / "pool/controller.json"
    state = json.loads(state_path.read_text())
    assert state["stop_reason"].startswith("no completed-part progress")

    def reject(args):
        if args[0] == "squeue":
            return ""
        raise subprocess.CalledProcessError(1, args, stderr="MaxSubmitJobsPerAccount")

    monkeypatch.setattr(pool, "command", reject)
    with pytest.raises(RuntimeError, match="MaxSubmitJobsPerAccount"):
        pool.controller(path, "fixture", hours=12, workers=8, poll=60)
    assert not json.loads(state_path.read_text())["submission_in_progress"]
