"""Resume coordination tests: no scheduler/cloud commands run outside stubs."""

from pathlib import Path
import os
import subprocess
import sys

import pytest

from scripts.curation import runtime
from scripts.curation_pool import run as pool
from scripts.youtube.download import _directory_lock, _write_json
from scripts.youtube.profile import _sha256
from scripts.youtube_resume import full_audit, run as resume
from test_curation_single import single_env as _single_env
from test_curation_production import production_env as _production_env
from test_curation_pool import complete_one

single_env, production_env = _single_env, _production_env


def test_controller_lock_probe_preserves_pool_code(tmp_path):
    before = pool.code_hashes()
    assert not resume.controller_busy(tmp_path)
    with _directory_lock(tmp_path):
        assert resume.controller_busy(tmp_path)
    assert not resume.controller_busy(tmp_path)
    assert pool.code_hashes() == before


def test_launcher_imports_without_optional_marlowe_helpers():
    project = Path(__file__).resolve().parents[1]
    code = """
import importlib.abc
import runpy
import sys

class MissingMarloweHelpers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'scripts.marlowe' or fullname.startswith('scripts.marlowe.'):
            raise ModuleNotFoundError(f'Not deployed: {fullname}')

sys.meta_path.insert(0, MissingMarloweHelpers())
sys.argv = ['scripts.youtube_resume.run', '--help']
runpy.run_module('scripts.youtube_resume.run', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project,
        env={**os.environ, "PYTHONPATH": f"{project / 'src'}:{project}"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--status" in result.stdout


@pytest.fixture
def continuation(single_env, monkeypatch):
    path, plan, *_ = single_env
    manifest_path = path.parent.parent / "handoff/manifest.json"
    manifest_path.parent.mkdir()
    manifest = {"held_array": "123", "completed_slots": []}
    _write_json(manifest_path, manifest)
    previous, output = (path.parent.parent / name for name in ("subset", "full"))
    previous.mkdir()
    directory = path.parent / "pool"
    directory.mkdir()
    _write_json(
        directory / "controller.json",
        {
            "plan_sha256": _sha256(path),
            "code": pool.code_hashes(),
            "active_gpu": "123",
            "finished_before_wave": 0,
        },
    )
    monkeypatch.setattr(
        resume.common, "validate_manifest", lambda _: (manifest, path, plan)
    )
    monkeypatch.setattr(resume, "verify_previous", lambda *args: True)
    monkeypatch.setattr(resume, "registered_active", lambda *args: set())
    monkeypatch.setattr(resume.pool, "queued", lambda _: set())
    monkeypatch.setattr(
        resume.transfer,
        "queue",
        lambda: [["123_0", "youtube-pool", "PENDING", "JobHeldUser"]],
    )
    return path, plan, manifest_path, previous, output


def test_waits_for_recovery_then_continues_until_all_attempted(
    continuation, monkeypatch
):
    path, plan, manifest, previous, output = continuation
    ready, commands, windows, sleeps = [False], [], [], []
    before = path.read_bytes()
    monkeypatch.setattr(resume, "verify_previous", lambda *args: ready[0])

    def sleep(seconds):
        assert not commands, "No GPU release before the recovery gate"
        sleeps.append(seconds)
        ready[0] = True

    def command(args):
        commands.append(args)
        return "456" if args[0] == "sbatch" else ""

    def controller(given_path, user, hours, workers, poll):
        assert given_path == path and workers == 8 and hours == 36
        pending, _ = pool.inventory(path, plan)
        windows.append(list(pending))
        for slot in pending[:2] if len(windows) == 1 else pending:
            complete_one(path, plan, slot)
        state = resume.read(path.parent / "pool/controller.json")
        state.pop("active_gpu", None)
        _write_json(path.parent / "pool/controller.json", state)
        return 0

    monkeypatch.setattr(resume.time, "sleep", sleep)
    monkeypatch.setattr(pool, "command", command)
    monkeypatch.setattr(pool, "controller", controller)
    assert resume.execute(path, manifest, previous, output, "fixture") == 0
    assert sleeps == [60] and len(windows) == 2
    assert set(windows[1]) == set(windows[0][2:])
    assert commands[0] == ["scontrol", "release", "123"]
    assert len(commands) == 2 and commands[1][:3] == [
        "sbatch",
        "--parsable",
        "--time=04:00:00",
    ]
    assert commands[1][-1].endswith("youtube_resume/final.sbatch")
    assert before == path.read_bytes()


def test_failed_recovery_never_releases_gpu(continuation, monkeypatch):
    path, _, manifest, previous, output = continuation
    (previous / "submitted-jobs.txt").write_text("999\n")
    monkeypatch.setattr(resume, "verify_previous", lambda *args: False)
    monkeypatch.setattr(
        pool, "command", lambda _: pytest.fail("Do not release or submit GPUs")
    )
    with pytest.raises(RuntimeError, match="finalization did not complete"):
        resume.execute(path, manifest, previous, output, "fixture")


def test_unexpected_gpu_array_refuses_release(continuation, monkeypatch):
    path, _, manifest, previous, output = continuation
    monkeypatch.setattr(
        resume.transfer,
        "queue",
        lambda: [["987_0", "youtube-pool", "PENDING", "JobHeldUser"]],
    )
    monkeypatch.setattr(
        pool, "command", lambda _: pytest.fail("Never release an unrelated array")
    )
    with pytest.raises(ValueError, match="Unexpected YouTube"):
        resume.execute(path, manifest, previous, output, "fixture")


def test_release_intent_recovers_without_duplicate_release(continuation, monkeypatch):
    path, _, manifest_path, _, _ = continuation
    directory = path.parent / "pool/after-modal-resume"
    directory.mkdir()
    state = {"release_intent": "123"}
    monkeypatch.setattr(
        resume.transfer, "queue", lambda: [["123_0", "youtube-pool", "RUNNING", "None"]]
    )
    monkeypatch.setattr(
        pool, "command", lambda _: pytest.fail("Release already happened")
    )
    resume.release_original(resume.read(manifest_path), path, directory, state)
    assert state["released"]
    resume.release_original(resume.read(manifest_path), path, directory, state)


def test_no_progress_wave_stops_without_restarting(continuation, monkeypatch):
    path, _, manifest, previous, output = continuation
    commands = []
    monkeypatch.setattr(pool, "command", lambda args: commands.append(args))
    monkeypatch.setattr(pool, "controller", lambda *args: 2)
    with pytest.raises(RuntimeError, match="status 2"):
        resume.execute(path, manifest, previous, output, "fixture")
    assert commands == [["scontrol", "release", "123"]]


def test_unknown_final_submission_is_not_repeated(tmp_path, monkeypatch):
    state = {"final_submission_in_progress": True}
    monkeypatch.setattr(pool, "queued", lambda _: set())
    monkeypatch.setattr(
        pool, "command", lambda _: pytest.fail("Never duplicate uncertain submission")
    )
    with pytest.raises(ValueError, match="outcome uncertain"):
        resume.launch_final(tmp_path, tmp_path / "full", "fixture", state)


def test_full_audit_refuses_partial_then_covers_all_youtube(single_env, monkeypatch):
    path, plan, loads, calls = single_env
    root = path.parent.parent
    manifest = root / "handoff/manifest.json"
    manifest.parent.mkdir()
    _write_json(manifest, {"fixture": True})
    previous = root / "previous"
    previous.mkdir()
    _write_json(previous / "import-audit.json", {"fixture": True})
    output = root / "full"
    monkeypatch.setattr(full_audit, "verify_previous", lambda *args: True)
    with pytest.raises(ValueError, match="Unfinished YouTube"):
        full_audit.execute(path, manifest, previous, output)
    for slot in pool.inventory(path, plan)[0]:
        pool.single.run_slot(path, plan, slot, runtime.ResidentModel())
    before = len(loads), list(calls)
    report = full_audit.execute(path, manifest, previous, output)
    assert report["audit_complete"] and report["all_planned_work_complete"]
    assert report["completed_parts"] == report["planned_parts"] == 7
    assert not report["unprocessed_slots"]
    checkpoint = resume.read(output / "import-audit.json")
    assert len(checkpoint["completed_slots"]) == 7
    assert checkpoint["audit_summary_sha256"] == _sha256(
        output / "audit/review/summary.json"
    )
    monkeypatch.setattr(
        full_audit.audit, "execute", lambda *args: pytest.fail("Reuse completed audit")
    )
    assert full_audit.execute(path, manifest, previous, output) == report
    assert before == (len(loads), calls)


def test_launcher_uses_only_marlowe_and_preserves_frozen_pool():
    root = Path(__file__).resolve().parents[1] / "scripts/youtube_resume"
    shell = (root / "run.sh").read_text()
    assert "SBATCH_TIMELIMIT:-01:00:00" in shell
    assert "modal run" not in shell and "scancel" not in shell
    cpu = (root / "final.sbatch").read_text()
    assert "--gres" not in cpu and "--gpus" not in cpu
    assert "youtube_resume.full_audit" in cpu
    assert "youtube_finalize.finalize" in cpu


def test_gate_validates_frozen_completion_evidence(tmp_path, monkeypatch):
    # Small cryptographically bound evidence packet; data validation is covered
    # by the finalizer's real-corpus tests, not imitated with fake inference here.
    previous, scoring = tmp_path / "previous", tmp_path / "scoring"
    for name in ("audit/review", "dedup"):
        (previous / name).mkdir(parents=True)
    scoring.mkdir()
    path, manifest = scoring / "plan.json", tmp_path / "manifest.json"
    _write_json(path, {"output_dir": str(scoring), "parts": [{}, {}]})
    _write_json(manifest, {"completed_slots": [0]})
    parts = []
    for slot in (0, 1):
        target = scoring / "parts" / f"{slot:05d}" / "summary.json"
        target.parent.mkdir(parents=True)
        _write_json(target, {"slot": slot})
        parts.append({"slot": slot, "summary_sha256": _sha256(target)})
    for relative in ("audit/frozen.json", "audit/review/frozen.json"):
        _write_json(previous / relative, {"parts": parts})
    _write_json(
        previous / "audit/review/summary.json",
        {
            "audit_complete": True,
            "plan_sha256": _sha256(path),
            "files": {"frozen.json": _sha256(previous / "audit/review/frozen.json")},
        },
    )
    _write_json(
        previous / "import-audit.json",
        {
            "complete": True,
            "manifest_sha256": _sha256(manifest),
            "plan_sha256": _sha256(path),
            "audit_summary_sha256": _sha256(previous / "audit/review/summary.json"),
            "completed_slots": [0, 1],
        },
    )
    _write_json(
        previous / "dedup/plan.json",
        {"import_audit_sha256": _sha256(previous / "import-audit.json")},
    )
    _write_json(previous / "dedup/summary.json", {})
    _write_json(previous / "corpus-manifest.json", {})
    _write_json(
        previous / "summary.json",
        {
            "complete": True,
            "audit_summary_sha256": _sha256(previous / "audit/review/summary.json"),
            "dedup_plan_sha256": _sha256(previous / "dedup/plan.json"),
            "dedup_summary_sha256": _sha256(previous / "dedup/summary.json"),
            "corpus_manifest_sha256": _sha256(previous / "corpus-manifest.json"),
        },
    )
    monkeypatch.setattr(resume, "recovered_slots", lambda _: {1})
    assert resume.verify_previous(previous, manifest, path)
    _write_json(scoring / "parts/00001/summary.json", {"slot": "changed"})
    with pytest.raises(ValueError, match="completed partition changed"):
        resume.verify_previous(previous, manifest, path)
