"""Real small-corpus audit/dedup integration; model calls are fixture stubs."""

from pathlib import Path

import pyarrow.parquet as pq
import pytest

pytest.importorskip("datasketch")

from scripts.curation import runtime
from scripts.curation_audit import run as audit
from scripts.curation_single import run as single
from scripts.near_dedup import run as dedup
from scripts.youtube.download import _write_json
from scripts.youtube.profile import _sha256
from scripts.youtube_finalize import finalize, pipeline, submit
from test_curation_single import single_env as _single_env
from test_curation_production import production_env as _production_env
from test_near_dedup import corpus as _corpus

single_env, production_env, corpus = _single_env, _production_env, _corpus


@pytest.fixture
def ready(single_env, corpus, tmp_path):
    path, plan, loads, calls = single_env
    for slot in (0, 1):
        single.run_slot(path, plan, slot, runtime.ResidentModel())
    output = tmp_path / "finalized"
    report = audit.execute(path, output / "audit")
    _write_json(
        output / "import-audit.json",
        {
            "complete": True,
            "plan_path": str(path),
            "plan_sha256": _sha256(path),
            "completed_slots": [0, 1],
            "audit_summary_sha256": _sha256(output / "audit/review/summary.json"),
        },
    )
    project, selected, web, refs, _, _ = corpus
    dedup.execute(project, selected, web, workers=1, references=refs)
    return output, web, path, plan, loads, calls, report


def test_real_audit_export_dedup_and_recount(ready):
    output, web, _, _, loads, calls, audit_report = ready
    before = {p: _sha256(p) for p in web.rglob("*") if p.is_file()}
    inference = len(loads), list(calls)
    summary = finalize.execute(output, web, workers=1)
    assert summary["complete"] and not summary["all_planned_work_complete"]
    assert not summary["release_ready"]
    assert summary["completed_parts"] == 2
    assert summary["planned_parts"] == audit_report["planned_parts"]
    assert summary["youtube_before_dedup"]["records"] == 2
    assert summary["youtube_after_dedup"]["records"] == 1
    assert summary["youtube_routes"]["exact"]["records"] == 1
    kept = next((output / "dedup/records").rglob("*.parquet"))
    row = pq.ParquetFile(kept).read().to_pylist()[0]
    assert row["text"] == "Memory maps connect virtual addresses to physical pages."
    assert summary["youtube_after_dedup"]["tokens"] == row["content_length"]
    web_total = finalize.read(web / "summary.json")["baseline_plus_additions"]
    assert (
        summary["corpus_total"]["tokens"] == web_total["tokens"] + row["content_length"]
    )
    manifest = finalize.read(output / "corpus-manifest.json")
    assert (
        sum(f["tokens"] for f in manifest["files"]) == summary["corpus_total"]["tokens"]
    )
    assert inference == (len(loads), calls)
    assert before == {p: _sha256(p) for p in before}
    assert finalize.execute(output, web, workers=1) == summary


def test_interrupted_finalization_reuses_signatures(ready, monkeypatch):
    output, web, *_ = ready
    original = dedup.resolve
    monkeypatch.setattr(
        dedup,
        "resolve",
        lambda *args: (_ for _ in ()).throw(InterruptedError("fixture stop")),
    )
    with pytest.raises(InterruptedError):
        finalize.execute(output, web, workers=1)
    assert not finalize.read(output / "summary.json")["complete"]
    monkeypatch.setattr(dedup, "resolve", original)
    monkeypatch.setattr(
        dedup,
        "input_rows",
        lambda *args: pytest.fail("Complete signatures must be reused"),
    )
    assert finalize.execute(output, web, workers=1)["complete"]


def test_changed_raw_response_refuses_finalization(ready):
    output, web, _, plan, *_ = ready
    raw = next((single.folder(plan, 0) / "work/score/decisions").glob("*.json"))
    raw.write_bytes(raw.read_bytes() + b" ")
    with pytest.raises(ValueError, match="worker/source files changed"):
        finalize.execute(output, web, workers=1)


def test_corrupt_prior_signature_refuses_reuse(ready):
    output, web, *_ = ready
    path = next((web / "signatures").rglob("*.parquet"))
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="Signature checkpoint changed"):
        finalize.execute(output, web, workers=1)


def test_corrupt_final_export_refuses_resume(ready):
    output, web, *_ = ready
    finalize.execute(output, web, workers=1)
    path = next((output / "dedup/records").rglob("*.parquet"))
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="Export checkpoint changed"):
        finalize.execute(output, web, workers=1)
    assert not finalize.read(output / "summary.json")["complete"]


def test_web_and_baseline_copies_are_preserved(corpus, tmp_path):
    project, selected, web, refs, records, _ = corpus
    dedup.execute(project, selected, web, workers=1, references=refs)
    web_plan, baseline = finalize.completed_web(web)
    root = tmp_path / "reuse"
    root.mkdir()
    plan = {"output": str(root), "web": str(web), "files": baseline}
    finalize.copy_signatures(plan, web_plan, "b" * 64)
    con = dedup.database(root, "b" * 64)
    dedup.load_signatures(con, plan, "b" * 64)
    # Compare exact and whitespace-only copies of both baseline and retained web.
    from test_near_dedup import put_document

    for i, text in enumerate(
        (records[0]["text"], records[2]["text"].replace(" ", "\n"))
    ):
        put_document(con, (len(baseline) + i) * dedup.STRIDE, text, False)
    con.commit()
    plan["files"] += [{"records": 2}]
    dedup.resolve(con, plan)
    assert con.execute(
        "SELECT count(*) FROM docs d JOIN decisions s USING(doc_id) WHERE d.baseline=1 AND s.keep=0"
    ).fetchone() == (0,)
    assert con.execute(
        "SELECT reason FROM docs d JOIN decisions s USING(doc_id) WHERE d.baseline=0 ORDER BY d.doc_id"
    ).fetchall() == [("exact",), ("near",)]
    con.close()


def test_recovery_inventory_includes_all_completed_slots(tmp_path):
    (tmp_path / "returned").mkdir()
    path = tmp_path / "manifest.json"
    _write_json(path, {"jobs": [[1, 2], [3, 4]]})
    digest = _sha256(path)
    workers = {}
    for worker, completed in ((0, [1]), (1, [3, 4])):
        inventory = tmp_path / "returned" / f"worker-{worker}.json"
        _write_json(
            inventory,
            {"worker": worker, "completed_slots": completed, "manifest_sha256": digest},
        )
        workers[str(worker)] = {"inventory_sha256": _sha256(inventory)}
    _write_json(
        tmp_path / "recovery-summary.json",
        {"complete": True, "manifest_sha256": digest, "workers": workers},
    )
    _write_json(
        tmp_path / "returned/downloaded.json",
        {"manifest_sha256": digest, "workers": [0, 1]},
    )
    assert pipeline.recovered_slots(path) == {1, 3, 4}
    _write_json(
        tmp_path / "returned/downloaded.json",
        {"manifest_sha256": digest, "workers": [0]},
    )
    with pytest.raises(ValueError, match="Incomplete recovery"):
        pipeline.recovered_slots(path)


def test_submission_template_requests_cpu_only():
    root = Path(__file__).resolve().parents[1] / "scripts/youtube_finalize"
    text = (root / "job.sbatch").read_text()
    assert "--gres" not in text and "--gpus" not in text
    assert "--cpus-per-task=8" in text
    assert "pipeline" in text and "finalize" in text


@pytest.mark.parametrize("missing_cloud_partition", [False, True])
def test_combined_import_scope_and_audit_checkpoint(
    ready, monkeypatch, missing_cloud_partition
):
    from scripts.modal_youtube import common, transfer

    output, _, path, plan, *_ = ready
    (output / "import-audit.json").unlink()
    manifest_path = output.parent / "handoff/manifest.json"
    pipeline.write(manifest_path, {"completed_slots": [0]})
    manifest = pipeline.read(manifest_path)
    monkeypatch.setattr(common, "validate_manifest", lambda _: (manifest, path, plan))
    monkeypatch.setattr(transfer, "check_held", lambda _: "123")
    monkeypatch.setattr(pipeline, "recovered_slots", lambda _: {1})

    def merge(_):
        pipeline.write(
            manifest_path.parent / "import-summary.json",
            {
                "complete": True,
                "imported_slots": [] if missing_cloud_partition else [1],
            },
        )

    monkeypatch.setattr(transfer, "merge", merge)
    if missing_cloud_partition:
        with pytest.raises(ValueError, match="every recovered"):
            pipeline.execute(manifest_path, output)
        assert not (output / "import-audit.json").exists()
    else:
        result = pipeline.execute(manifest_path, output)
        assert result["audit_complete"] and result["completed_parts"] == 2
        assert pipeline.read(output / "import-audit.json")["completed_slots"] == [0, 1]
        monkeypatch.setattr(
            transfer, "merge", lambda _: pytest.fail("Import checkpoint must be reused")
        )
        monkeypatch.setattr(
            audit,
            "execute",
            lambda *args: pytest.fail("Completed audit must be reused"),
        )
        assert pipeline.execute(manifest_path, output) == result


def test_launcher_blocks_duplicate_and_uncertain_submissions(tmp_path, monkeypatch):
    from types import SimpleNamespace

    output = tmp_path / "finalized"
    output.mkdir()
    monkeypatch.setenv("YT_FINISH_OUTPUT", str(output))
    monkeypatch.setattr(submit.sys, "argv", ["submit.py"])
    calls = []

    def queued(args, **kwargs):
        calls.append(args)
        assert args[0] == "squeue", (
            "Must stop before fetching, importing or allocating compute"
        )
        return SimpleNamespace(stdout="123\n")

    monkeypatch.setattr(submit, "call", queued)
    (output / "submitted-jobs.txt").write_text("123\n")
    submit.main()
    assert len(calls) == 1
    (output / "submitted-jobs.txt").unlink()
    (output / "submission-in-progress").write_text("uncertain")
    with pytest.raises(RuntimeError, match="uncertain"):
        submit.main()
