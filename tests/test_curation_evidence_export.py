import json

import pytest

from scripts.curation import production as prod, production_audit as audit
from scripts.curation_review.export import export
from scripts.youtube.profile import _sha256
from test_curation_production import production_env as _production_env

production_env = _production_env


def test_evidence_export_preserves_raw_attempts_and_source_and_has_no_inference(production_env):
    path, plan, loads, calls = production_env
    for slot in range(len(plan["parts"])):
        assert prod.run_slot(path, slot) == 0
    audit.finalize(path)
    before = {p: _sha256(p) for p in path.parent.rglob("*") if p.is_file()}
    original_calls, original_loads = len(calls), len(loads)
    out = path.parent.parent / "evidence"
    report = export(path, out)
    assert report["complete"] and not report["inference_performed"]
    assert not report["release_ready"] and not report["quality_review_complete"]
    assert report["exported_documents"] == 6
    assert report["all_youtube_candidate_documents"] == 1
    assert len(calls) == original_calls and len(loads) == original_loads
    assert before == {p: _sha256(p) for p in path.parent.rglob("*") if p.is_file()}
    evidence = [json.loads(line) for line in (out / "evidence.jsonl").read_text().splitlines()]
    assert all({s["stage"] for s in e["spans"]} == {"screen", "critic"} for e in evidence)
    expected = audit.read_rows(prod.part_dir(plan, 0) / "spans.parquet")[0]
    observed = next(s for e in evidence if e["slot"] == 0 for s in e["spans"] if s["key"] == expected["key"] and s["stage"] == expected["stage"])
    assert observed["attempts"] == json.loads(expected["attempts_json"])
    diagnostic = json.loads((out / "diagnostics.json").read_text())
    assert sum(r["documents"] for r in diagnostic["route_pairs"]) == 6
    assert {r["combined"] for r in diagnostic["route_pairs"]} == {"eligible", "exclude", "review_required"}
    assert export(path, out) == report


def test_evidence_export_refuses_stale_audit_and_protected_output(production_env):
    path, plan, _, _ = production_env
    for slot in range(len(plan["parts"])):
        prod.run_slot(path, slot)
    audit.finalize(path)
    with pytest.raises(ValueError, match="separate"):
        export(path, path.parent / "evidence")
    with pytest.raises(ValueError, match="separate"):
        export(path, path.parent.parent)
    packet = path.parent / "review/review-packet.json"
    packet.write_text(packet.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        export(path, path.parent.parent / "evidence")


def test_evidence_export_rechecks_candidate_membership(production_env):
    path, plan, _, _ = production_env
    for slot in range(len(plan["parts"])):
        prod.run_slot(path, slot)
    audit.finalize(path)
    folder = prod.part_dir(plan, 0)
    (folder / "candidates.parquet").write_bytes((folder / "documents.parquet").read_bytes())
    with pytest.raises(ValueError):
        export(path, path.parent.parent / "evidence")
