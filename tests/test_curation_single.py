import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.curation import production as old, runtime
from scripts.curation_single import rubric, run, audit
from scripts.youtube.download import _write_json
from scripts.youtube.profile import _sha256
from test_curation_production import production_env as _production_env

production_env = _production_env


def answer(**changes):
    return {
        "domain": "computing",
        "quality": "useful",
        "language": "english",
        "reason": "Explains a concrete computing mechanism.",
        **changes,
    }


@pytest.fixture
def single_env(production_env, monkeypatch):
    _, parent, loads, calls = production_env
    root = Path(parent["input_dir"])
    manifest = json.loads((root / "summary.json").read_text())
    original = next(p for p in manifest["parts"] if "/youtube-commons/" in p["path"])
    rows = pq.ParquetFile(root / original["path"]).read().to_pylist()
    for i in range(3, 7):
        copied = [{**r, "source_shard": f"raw-{i}.parquet"} for r in rows]
        part = root / f"shards/youtube-commons/{i:05d}/00000.parquet"
        part.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(copied, schema=old.prepare.SCHEMA), part)
        manifest["parts"].append(
            {
                "path": str(part.relative_to(root)),
                "sha256": _sha256(part),
                "records": len(copied),
                "tokens": sum(r["content_length"] for r in copied),
            }
        )
    manifest["records"] = sum(p["records"] for p in manifest["parts"])
    manifest["tokens"] = sum(p["tokens"] for p in manifest["parts"])
    _write_json(root / "summary.json", manifest)
    llm = sys.modules["vllm"].LLM
    original_generate = llm.generate

    def generate(self, prompts, params, **kwargs):
        result = original_generate(self, prompts, params, **kwargs)
        for i, r in enumerate(result):
            value = answer()
            if i == 1:
                value["domain"] = "other"
            if i == 2:
                value["quality"] = "marginal"
            r.outputs[0].text = json.dumps(value)
        return result

    monkeypatch.setattr(llm, "generate", generate)
    output = root.parent / "single"
    plan = run.make_plan(root, output)
    return output / "plan.json", plan, loads, calls


def test_compact_labels_allow_supporting_computing_and_minor_imperfections():
    parsed = rubric.parse_response(
        json.dumps(
            answer(reason="Useful debugging steps despite minor transcription errors.")
        ),
        "source",
        0,
        "stop",
    )
    assert rubric.route(parsed["labels"], parsed["parse_status"]) == "eligible"
    assert rubric.route(answer(quality="marginal")) == "review_required"
    assert rubric.route(answer(domain="other")) == "exclude"
    assert rubric.route(answer(language="uncertain")) == "review_required"
    assert (
        rubric.parse_response(json.dumps(answer()), "x", 0, "length")["labels"] is None
    )
    assert (
        rubric.parse_response(json.dumps({**answer(), "keep": True}), "x", 0, "stop")[
            "labels"
        ]
        is None
    )
    assert rubric.parse_response('{"domain":null}', "x", 0, "stop")["labels"] is None


def test_single_pass_export_resume_audit_and_immutable_source(single_env):
    path, plan, loads, calls = single_env
    root = Path(plan["input_dir"])
    before = {p: _sha256(p) for p in root.rglob("*") if p.is_file()}
    assert [p["path"].split("/")[1] for p in plan["parts"][:8]] == [
        "youtube-commons"
    ] * 6 + ["primus-fineweb", "redsage-cfw"]
    factory = runtime.ResidentModel()
    for slot in (0, 6, 7):
        assert run.run_slot(path, plan, slot, factory) == 0
        _, exports = run.audit_part(path, plan, slot)
        assert len(exports["candidates.parquet"]) == 1
        assert [r["route"] for r in exports["decisions.parquet"]] == [
            "eligible",
            "exclude",
            "review_required",
        ]
        assert all(
            r["selection_decision"] is None for r in exports["decisions.parquet"]
        )
        assert run.run_slot(path, plan, slot, factory) == 0
    assert len(loads) == 1 and calls == [
        3,
        3,
        3,
    ]  # One pass, no critic, no resume inference.
    report = audit.finalize(path, draws=4)
    assert report["audit_complete"] and not report["all_planned_work_complete"]
    assert report["completed_parts"] == 3
    assert all(report["by_source"][s]["candidate_records"] == 1 for s in run.SOURCES)
    assert not report["release_ready"] and not report["quality_review_complete"]
    assert before == {p: _sha256(p) for p in root.rglob("*") if p.is_file()}
    assert len(loads) == 1 and calls == [3, 3, 3]
    assert run.make_plan(root, path.parent) == plan


def test_tamper_and_plan_changes_refuse_resume(single_env):
    path, plan, loads, calls = single_env
    assert run.run_slot(path, plan, 0, runtime.ResidentModel()) == 0
    directory = run.folder(plan, 0)
    raw = next((directory / "work/score/decisions").glob("*.json"))
    original = raw.read_bytes()
    value = json.loads(original)
    value["attempts"][-1]["finish_reason"] = "length"
    _write_json(raw, value)
    with pytest.raises(ValueError, match="Export disagrees"):
        run.audit_part(path, plan, 0)
    raw.write_bytes(original)
    (directory / "candidates.parquet").write_bytes(
        (directory / "documents.parquet").read_bytes()
    )
    with pytest.raises(ValueError, match="checksum"):
        run.run_slot(path, plan, 0, runtime.ResidentModel())
    assert len(loads) == 1 and calls == [3]
    value = json.loads(path.read_text())
    value["code"]["missing.py"] = "bad"
    _write_json(path, value)
    with pytest.raises(ValueError, match="binding"):
        run.load_plan(path)


def test_token_weighted_draws_are_reproducible_and_cover_long_document():
    rows = [
        {"case_id": "a", "content_length": 1},
        {"case_id": "b", "content_length": 999},
    ]
    draws = audit.token_draws(rows, 1000, "fixed")
    assert draws == audit.token_draws(list(reversed(rows)), 1000, "fixed")
    assert sum(d["case_id"] == "b" for d in draws) > 980
    assert len({d["token_offset"] for d in draws if d["case_id"] == "b"}) > 500
    assert all(
        0 <= d["token_offset"] < (999 if d["case_id"] == "b" else 1) for d in draws
    )


def test_pending_worklists_select_only_requested_source(single_env):
    path, plan, _, _ = single_env
    work = path.parent / "work.json"
    assert (
        run.main(
            [
                "pending",
                "--plan",
                str(path),
                "--source",
                "youtube-commons",
                "--work-list",
                str(work),
            ]
        )
        == 0
    )
    value = json.loads(work.read_text())
    indices = [i for row in value["jobs"] for i in row]
    assert len(indices) == 7 and all(
        "/youtube-commons/" in plan["parts"][i]["path"] for i in indices
    )
    with pytest.raises(ValueError, match="overwrite"):
        run.main(
            [
                "pending",
                "--plan",
                str(path),
                "--source",
                "all",
                "--work-list",
                str(work),
            ]
        )
