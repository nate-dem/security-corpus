import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

pytest.importorskip("sklearn")

from scripts.web_filter import run as web
from scripts.web_semantic import model, run
from scripts.youtube.download import _write_json
from scripts.youtube.profile import _sha256
from test_web_filter import web_env as _web_env

web_env = _web_env


@pytest.fixture
def setup(web_env, tmp_path):
    path, parent = web_env
    web.baseline_index(path)
    for source in web.SOURCES:
        web.process_source(path, source, workers=1)
    weights = tmp_path / "model"
    weights.mkdir()
    np.savez_compressed(
        weights / "weights.npz",
        coef=np.zeros((3, model.FEATURES["n_features"])),
        intercept=np.array([2.0, 2.0, -2.0]),
    )
    _write_json(
        weights / "model.json",
        {
            "runtime_sha256": _sha256(Path(model.__file__)),
            "weights_sha256": _sha256(weights / "weights.npz"),
        },
    )
    output = tmp_path / "scored"
    plan = run.make_plan(Path(parent["output"]), weights, output)
    return output / "plan.json", plan


def test_scope_resume_audit_and_explicit_selection_preserve_sources(setup, tmp_path):
    path, plan = setup
    before = {p: _sha256(p) for p in Path(plan["web"]).rglob("*.parquet")}
    report = run.score_all(path, workers=2)
    assert report["complete"] and not report["release_ready"]
    assert report["by_source"]["primus-fineweb"]["scored_records"] == 1
    assert report["by_source"]["redsage-cfw"]["deferred_upstream_scope_records"] == 1
    rows = (
        pq.ParquetFile(run.part_folder(plan, 0) / "scores.parquet").read().to_pylist()
    )
    assert rows[0]["selection_decision"] is None
    assert rows[0]["joint_score"] == pytest.approx(1 / (1 + np.exp(-2)))
    assert rows[0]["usability_score"] < 0.2  # Diagnostic only; never a hidden gate.
    assert run.score_all(path, workers=1) == report
    sample = json.loads((Path(plan["output"]) / "review/sample.json").read_text())
    assert len(sample["draws"]) == 32 and len(sample["cases"]) == 1
    assert all(
        0 <= r["token_offset"] < sample["cases"][0]["content_length"]
        for r in sample["draws"]
    )
    selected = run.select(path, tmp_path / "selected", 0.7)
    assert selected["by_source"]["primus-fineweb"]["records"] == 1
    assert "redsage-cfw" not in selected["by_source"]
    assert not selected["release_ready"]
    selected_audit = json.loads((tmp_path / "selected/review/sample.json").read_text())
    assert len(selected_audit["draws"]) == 64
    assert all(c["scores"]["joint_score"] >= 0.7 for c in selected_audit["cases"])
    assert before == {p: _sha256(p) for p in before}
    with pytest.raises(ValueError, match="Changed selection"):
        run.select(path, tmp_path / "selected", 0.8)


def test_input_corruption_fails_before_scoring(setup):
    path, plan = setup
    source = Path(plan["web"]) / plan["parts"][0]["output_part"] / "candidates.parquet"
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        run.score_all(path, workers=1)


def test_changed_weights_and_scores_rejected(setup):
    path, plan = setup
    run.score_all(path, workers=1)
    result = run.part_folder(plan, 0) / "scores.parquet"
    result.write_bytes(result.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        run.score_all(path, workers=1)
    weights = Path(plan["model"]) / "weights.npz"
    weights.write_bytes(weights.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="weights checksum"):
        model.load_model(plan["model"])


def test_source_paths_cannot_escape_bound_directory(tmp_path):
    with pytest.raises(ValueError, match="leaves"):
        run.checked_path(tmp_path, "../outside")
