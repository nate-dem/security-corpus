"""The completed-only path freezes scope and can never schedule inference."""

import subprocess

import pytest

from scripts.curation import runtime
from scripts.curation_single import run as single
from scripts.youtube.download import _write_json
from scripts.youtube_snapshot import run
from test_curation_single import single_env as _single_env
from test_curation_production import production_env as _production_env
from test_near_dedup import corpus as _corpus

single_env, production_env, corpus = _single_env, _production_env, _corpus


def test_snapshot_bridge_runs_real_partial_dedup(single_env, corpus, tmp_path):
    from scripts.near_dedup import run as dedup
    from scripts.youtube_finalize import finalize

    path, plan, loads, calls = single_env
    for slot in (0, 1):
        single.run_slot(path, plan, slot, runtime.ResidentModel())
    project, selected, web, refs, *_ = corpus
    dedup.execute(project, selected, web, workers=1, references=refs)
    output = tmp_path / "snapshot"
    scope = run.prepare(path, output, web, 2)
    before = len(loads), list(calls)
    run.audit_snapshot(path, output, scope, threads=2)
    result = finalize.execute(output, web, workers=1)
    assert result["complete"] and not result["all_planned_work_complete"]
    assert result["completed_parts"] == 2
    assert result["youtube_after_dedup"]["records"] == 1
    assert result["unprocessed_slots"] == scope["unprocessed_slots"]
    assert before == (len(loads), calls)


def test_partial_snapshot_audit_and_resume_never_expands(single_env, tmp_path):
    path, plan, loads, calls = single_env
    single.run_slot(path, plan, 0, runtime.ResidentModel())
    output, web = tmp_path / "snapshot", tmp_path / "web"
    scope = run.prepare(path, output, web, 1)
    assert scope["completed_slots"] == [0] and 1 in scope["unprocessed_slots"]
    # Even if a formerly pending slot later finishes, it cannot enter this run.
    single.run_slot(path, plan, 1, runtime.ResidentModel())
    assert run.prepare(path, output, web, 1) == scope
    before = len(loads), list(calls)
    result = run.audit_snapshot(path, output, scope, threads=2)
    assert result["audit_complete"] and result["completed_parts"] == 1
    assert not result["all_planned_work_complete"]
    assert run.audit_snapshot(path, output, scope, threads=2) == result
    assert before == (len(loads), calls)
    assert run.read(output / "import-audit.json")["completed_slots"] == [0]


def test_scope_count_and_tampering_fail_closed(single_env, tmp_path):
    path, plan, *_ = single_env
    single.run_slot(path, plan, 0, runtime.ResidentModel())
    output, web = tmp_path / "snapshot", tmp_path / "web"
    with pytest.raises(ValueError, match="count differs"):
        run.prepare(path, output, web, 2)
    assert not (output / "scope.json").exists()
    run.prepare(path, output, web, 1)
    frozen = run.read(output / "audit/frozen.json")
    frozen["parts"][0]["summary_sha256"] = "0" * 64
    _write_json(output / "audit/frozen.json", frozen)
    with pytest.raises(ValueError, match="Frozen subset changed"):
        run.prepare(path, output, web, 1)


def test_submission_cpu_only_scrubs_gpu_env_and_reuses_job(tmp_path, monkeypatch):
    _write_json(tmp_path / "scope.json", {"parts": [0]})
    active, calls = set(), []
    monkeypatch.setattr(run.pool, "queued", lambda _: active)

    def submit(args, **kwargs):
        calls.append(args)
        assert run.read(tmp_path / "submission.json")["submission_in_progress"]
        assert not any(k.startswith("SBATCH_") for k in kwargs["env"])
        assert not any("gpu" in arg or "gres" in arg or "array" in arg for arg in args)
        return subprocess.CompletedProcess(args, 0, "321;cluster\n")

    monkeypatch.setattr(run.subprocess, "run", submit)
    assert run.submit(tmp_path, "fixture", {"SBATCH_GPUS": "8"}) == "321"
    active.add("321")
    assert run.submit(tmp_path, "fixture", {}) == "321"
    assert len(calls) == 1


def test_uncertain_submission_cannot_double_submit(tmp_path, monkeypatch):
    _write_json(tmp_path / "scope.json", {})
    monkeypatch.setattr(run.pool, "queued", lambda _: set())

    def fail(*args, **kwargs):
        raise OSError("connection interrupted")

    monkeypatch.setattr(run.subprocess, "run", fail)
    with pytest.raises(OSError):
        run.submit(tmp_path, "fixture", {})
    with pytest.raises(ValueError, match="outcome uncertain"):
        run.submit(tmp_path, "fixture", {})


def test_known_active_gpu_blocks_cpu_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "pool-root"
    (root / "pool").mkdir(parents=True)
    (root / ".download.lock").touch()
    (root / "pool/.download.lock").touch()
    _write_json(root / "pool/controller.json", {"active_gpu": "123"})
    monkeypatch.setattr(run.pool, "queued", lambda _: {"123"})
    with pytest.raises(ValueError, match="still queued"):
        with run.idle_pool({"output_dir": str(root)}, "fixture"):
            pytest.fail("Must not enter")


def test_existing_source_lock_never_creates_files(tmp_path):
    with pytest.raises(RuntimeError, match="no file was created"):
        with run.existing_source_lock(tmp_path):
            pytest.fail("Missing lock must not be created")
    assert list(tmp_path.iterdir()) == []
    path = tmp_path / ".download.lock"
    path.write_bytes(b"preserve")
    before = path.stat()
    with run.existing_source_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another process"):
            with run.existing_source_lock(tmp_path):
                pytest.fail("Exclusive lock must conflict")
    assert path.read_bytes() == b"preserve"
    assert (path.stat().st_ino, path.stat().st_mtime_ns) == (
        before.st_ino,
        before.st_mtime_ns,
    )
