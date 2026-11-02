import json
from pathlib import Path

import pytest

from scripts.curation import runtime
from scripts.curation_audit import run as resumed
from scripts.curation_single import run
from scripts.youtube.download import _write_json
from scripts.youtube.profile import _sha256
from test_curation_single import (
    single_env as _single_env,
    production_env as _production_env,
)

single_env, production_env = _single_env, _production_env


@pytest.fixture
def finished(single_env):
    path, plan, loads, calls = single_env
    for slot in (0, 1, 6):
        run.run_slot(path, plan, slot, runtime.ResidentModel())
    output = path.parent.parent / "cpu-audit"
    return path, plan, output, loads, calls


def test_complete_snapshot_resume_and_no_additional_inference(finished, monkeypatch):
    path, plan, output, loads, calls = finished
    inference = (len(loads), list(calls))
    before = {p: _sha256(p) for p in path.parent.rglob("*") if p.is_file()}
    report = resumed.execute(path, output)
    assert report["audit_complete"] and not report["all_planned_work_complete"]
    assert report["completed_parts"] == 2 and report["planned_parts"] == 7
    assert report["by_source"]["youtube-commons"]["candidate_records"] == 2
    assert set(report["by_source"]) == {"youtube-commons"}
    packet = json.loads((output / "review/review-packet.json").read_text())
    assert len(packet["cases"]) == 6
    draws = json.loads((output / "review/draws.json").read_text())
    assert len(draws["draws"]) == 96
    assert (len(loads), calls) == inference
    assert before == {p: _sha256(p) for p in before}
    monkeypatch.setattr(
        run, "audit_part", lambda *args: pytest.fail("Verified part must use its cache")
    )
    assert resumed.execute(path, output) == report


def test_interruption_resumes_from_verified_part(finished, monkeypatch):
    path, plan, output, _, _ = finished
    original = run.audit_part
    visited = []

    def stop_after_one(path, plan, slot):
        visited.append(slot)
        if slot == 1:
            raise InterruptedError("fixture interruption")
        return original(path, plan, slot)

    monkeypatch.setattr(run, "audit_part", stop_after_one)
    with pytest.raises(InterruptedError):
        resumed.execute(path, output)
    assert (output / "parts/00000/audit.json").exists()
    assert not json.loads((output / "review/summary.json").read_text())[
        "audit_complete"
    ]

    def track(path, plan, slot):
        visited.append(slot)
        return original(path, plan, slot)

    monkeypatch.setattr(run, "audit_part", track)
    assert resumed.execute(path, output)["audit_complete"]
    assert visited == [0, 1, 1]


def test_changed_raw_response_invalidates_verified_cache(finished):
    path, plan, output, _, _ = finished
    resumed.execute(path, output)
    raw = next((run.folder(plan, 0) / "work/score/decisions").glob("*.json"))
    value = json.loads(raw.read_text())
    value["attempts"][-1]["finish_reason"] = "length"
    _write_json(raw, value)
    with pytest.raises(ValueError, match="Cached audit"):
        resumed.execute(path, output)


def test_corrupt_cached_export_is_rejected(finished):
    path, _, output, _, _ = finished
    resumed.execute(path, output)
    cached = output / "parts/00000/decisions.parquet"
    cached.write_bytes(cached.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Cached audit"):
        resumed.execute(path, output)


def test_later_completed_parts_do_not_silently_change_snapshot(finished):
    path, plan, output, _, _ = finished
    first = resumed.execute(path, output)
    run.run_slot(path, plan, 2, runtime.ResidentModel())
    assert resumed.execute(path, output) == first
    assert (
        resumed.execute(path, output.with_name("later-audit"))["completed_parts"] == 3
    )


def test_changed_cached_counts_are_rejected(finished):
    path, _, output, _, _ = finished
    resumed.execute(path, output)
    marker = output / "parts/00000/audit.json"
    report = json.loads(marker.read_text())
    report["candidate_tokens"] += 1
    _write_json(marker, report)
    with pytest.raises(ValueError, match="Cached audit"):
        resumed.execute(path, output)


def test_changed_snapshot_is_rejected(finished):
    path, _, output, _, _ = finished
    resumed.execute(path, output)
    marker = output / "frozen.json"
    frozen = json.loads(marker.read_text())
    frozen["parts"] = frozen["parts"][:1]
    _write_json(marker, frozen)
    with pytest.raises(ValueError, match="Cached audit"):
        resumed.execute(path, output)


def test_bound_plan_hash_optimization_restores_original_function(tmp_path, monkeypatch):
    path = tmp_path / "plan.json"
    path.write_text("large-plan-fixture")
    other = tmp_path / "other.json"
    other.write_text("unrelated bytes")
    seen = []
    original = run._sha256

    def tracking(target):
        seen.append(Path(target))
        return original(target)

    monkeypatch.setattr(run, "_sha256", tracking)
    with resumed.bound_plan_digest(path, _sha256(path)):
        for _ in range(100):
            assert run._sha256(path) == _sha256(path)
        assert run._sha256(other) == _sha256(other)
    assert seen == [other] and run._sha256 is tracking
    with pytest.raises(ValueError, match="during partition"):
        with resumed.bound_plan_digest(path, _sha256(path)):
            path.write_text("changed")
    assert run._sha256 is tracking
