from pathlib import Path
import shutil

import pytest

from scripts.curation import runtime
from scripts.curation_single import run
from scripts.modal_youtube import common as c, transfer
from test_curation_single import (
    single_env as _single_env,
    production_env as _production_env,
)

single_env, production_env = _single_env, _production_env


def test_allocation_is_disjoint_balanced_and_under_allowance():
    parts = [{"path": f"shards/youtube-commons/{i}", "tokens": 100} for i in range(17)]
    jobs = c.allocate(list(range(17)), parts)
    assert sorted(n for group in jobs for n in group) == list(range(17))
    assert len(jobs) == 8 and max(map(len, jobs)) == 3
    assert 8 * (c.CALL_SECONDS + c.STARTUP_SECONDS) / 3600 * c.HOURLY_DOLLARS < 300
    for slots in ([-1], [17], [True], [1, 1]):
        with pytest.raises(ValueError):
            c.allocate(slots, parts)
    parts[0]["path"] = "shards/primus-fineweb/a"
    with pytest.raises(ValueError, match="YouTube"):
        c.allocate([0], parts)


def test_running_jobs_cannot_be_held_or_migrated(tmp_path, monkeypatch):
    c.write(tmp_path / "pool/controller.json", {"active_gpu": "123"})
    monkeypatch.setattr(
        transfer, "queue", lambda: [["123_0", "youtube-pool", "RUNNING", "n01"]]
    )
    monkeypatch.setattr(
        transfer.subprocess,
        "run",
        lambda *a, **kw: pytest.fail("Do not hold running jobs"),
    )
    with pytest.raises(RuntimeError, match="running"):
        transfer.check_held({"output_dir": str(tmp_path)}, hold=True)


def test_pending_array_is_held_then_verified(tmp_path, monkeypatch):
    c.write(tmp_path / "pool/controller.json", {"active_gpu": "123"})
    snapshots = iter(
        [
            [["123_0", "youtube-pool", "PENDING", "Priority"]],
            [["123_0", "youtube-pool", "PENDING", "JobHeldUser"]],
        ]
    )
    monkeypatch.setattr(transfer, "queue", lambda: next(snapshots))
    calls = []
    monkeypatch.setattr(
        transfer.subprocess, "run", lambda args, **kw: calls.append(args)
    )
    assert transfer.check_held({"output_dir": str(tmp_path)}, hold=True) == "123"
    assert calls == [["scontrol", "hold", "123"]]


@pytest.fixture
def handoff(single_env, monkeypatch):
    path, plan, loads, calls = single_env
    run.run_slot(path, plan, 0, runtime.ResidentModel())
    monkeypatch.setattr(transfer, "check_held", lambda plan, hold=False: "123")
    monkeypatch.setattr(transfer.importlib.metadata, "version", lambda name: "1.0")
    directory = path.parent.parent / "modal-handoff-test"
    transfer.prepare(path, directory)
    return path, plan, directory, loads, calls


def test_handoff_excludes_completed_and_web_preserves_source_bytes(handoff):
    path, plan, directory, _, _ = handoff
    manifest, checked_path, _ = c.validate_manifest(directory / "manifest.json")
    assert path == checked_path
    assert 0 not in manifest["slots"]
    assert len(manifest["slots"]) == 6
    assert all(
        "/youtube-commons/" in plan["parts"][n]["path"] for n in manifest["slots"]
    )
    c.verify_files(Path(manifest["mount"]), manifest["files"])
    assert c.sha(path) == manifest["plan_sha256"]
    assert len(manifest["runtime"]) >= 7


def test_changed_source_prevents_reusing_handoff(handoff):
    path, plan, directory, _, _ = handoff
    manifest = c.read(directory / "manifest.json")
    source = Path(plan["input_dir"]) / plan["parts"][manifest["slots"][0]]["path"]
    source.write_bytes(source.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="Artifact changed"):
        transfer.prepare(path, directory)


def staged_part(handoff):
    path, plan, directory, loads, calls = handoff
    manifest_path = directory / "manifest.json"
    manifest = c.read(manifest_path)
    slot = manifest["jobs"][0][0]
    run.run_slot(path, plan, slot, runtime.ResidentModel())
    part = run.folder(plan, slot)
    c.write(part / "modal-origin.json", {"manifest_sha256": c.sha(manifest_path)})
    returned = directory / "returned"
    target = returned / part.relative_to(manifest["mount"])
    shutil.copytree(part, target)
    files = {
        str(p.relative_to(returned)): {"sha256": c.sha(p), "bytes": p.stat().st_size}
        for p in c.part_files(target)
    }
    c.write(returned / "worker-0.json", {"files": files})
    c.write(
        returned / "downloaded.json",
        {"manifest_sha256": c.sha(manifest_path), "workers": [0]},
    )
    shutil.rmtree(part)
    return manifest_path, part, target, slot, loads, calls


def test_import_verifies_raw_data_preserves_plan_and_is_resumable(handoff, monkeypatch):
    manifest_path, destination, _, slot, loads, calls = staged_part(handoff)
    path = handoff[0]
    original_digest = c.sha(path)
    inference = len(loads), list(calls)
    transfer.merge(manifest_path)
    assert c.read(destination / "summary.json")["workflow_finished"]
    assert c.sha(path) == original_digest
    assert (len(loads), calls) == inference
    monkeypatch.setattr(
        run, "audit_part", lambda *a, **kw: pytest.fail("Do not import twice")
    )
    transfer.merge(manifest_path)
    assert (
        slot in c.read(manifest_path.parent / "import-summary.json")["imported_slots"]
    )


def test_import_rejects_changed_export(handoff):
    manifest_path, destination, staged, _, _, _ = staged_part(handoff)
    candidate = staged / "candidates.parquet"
    candidate.write_bytes(candidate.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Artifact changed"):
        transfer.merge(manifest_path)
    assert not destination.exists()


def test_rehashed_malformed_raw_response_still_fails_import(handoff):
    manifest_path, destination, staged, _, _, _ = staged_part(handoff)
    raw = next((staged / "work/score/decisions").glob("*.json"))
    raw.write_text("{}")
    inventory_path = manifest_path.parent / "returned/worker-0.json"
    inventory = c.read(inventory_path)
    relative = str(raw.relative_to(manifest_path.parent / "returned"))
    inventory["files"][relative] = {"sha256": c.sha(raw), "bytes": raw.stat().st_size}
    c.write(inventory_path, inventory)
    with pytest.raises((ValueError, KeyError)):
        transfer.merge(manifest_path)
    assert not destination.exists()


class Ledger(dict):
    def put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = value
        return True


class Call:
    def __init__(self, index):
        self.index, self.cancelled = index, False

    def get(self, timeout):
        return {"worker": self.index}

    def cancel(self, terminate_containers):
        assert terminate_containers
        self.cancelled = True


def test_one_cloud_wave_and_no_duplicate_launch():
    ledger, calls = Ledger(), []

    def spawn(index):
        if index:
            assert ledger["startup_ok"]
        else:
            ledger.put("startup_ok", True)
        call = Call(index)
        calls.append(call)
        return call

    results = c.launch_wave(8, ledger, spawn, "plan")
    assert len(results) == len(calls) == 8
    with pytest.raises(RuntimeError, match="already launched"):
        c.launch_wave(8, ledger, spawn, "plan")
    assert len(calls) == 8


def test_startup_failure_never_launches_other_seven_workers():
    ledger, calls = Ledger(), []

    def spawn(index):
        assert index == 0
        call = Call(index)
        calls.append(call)
        ledger.put("worker-done-0", {"exit_code": 1})
        return call

    with pytest.raises(RuntimeError, match="First worker stopped"):
        c.launch_wave(8, ledger, spawn, "plan")
    assert len(calls) == 1 and calls[0].cancelled
    assert "coordinator_done" in ledger


def test_slow_start_stops_before_fanout():
    ledger, call = Ledger(), Call(0)
    clock = iter([0, 1801])
    with pytest.raises(RuntimeError, match="30 minutes"):
        c.launch_wave(8, ledger, lambda i: call, "plan", clock=lambda: next(clock))
    assert call.cancelled
