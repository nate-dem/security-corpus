import json
import os
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ingest.utils import compute_content_hash, compute_token_count
from scripts.curation import production as prod, production_audit as audit, prepare
from scripts.youtube.download import _write_json
from scripts.youtube.profile import _sha256
from scripts.youtube_filter import score as engine
from test_curation_review import fake_model
from test_curation_v3 import response
from test_youtube_evidence_repair import Tokenizer


@pytest.fixture
def production_env(tmp_path, monkeypatch):
    loads = fake_model(monkeypatch)

    class CompressedTokenizer(Tokenizer):
        def encode(self, text, **kwargs):
            return super().encode(text, **kwargs)[::4]

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    monkeypatch.setattr(
        engine, "load_tokenizer", lambda *args: (CompressedTokenizer(), snapshot)
    )
    monkeypatch.setattr(engine, "_versions", lambda *args: {"fixture": "1"})
    monkeypatch.setattr(
        prod.cuda_preflight, "configure_and_check", lambda: {"fixture": "compiler"}
    )
    monkeypatch.setattr(
        prod.cuda_preflight, "check_sampler", lambda: {"fixture": "sampler"}
    )
    llm = sys.modules["vllm"].LLM
    original = llm.generate
    calls = []

    def generate(self, prompts, sampling, **kwargs):
        results = original(self, prompts, sampling, **kwargs)
        calls.append(len(prompts))
        critic = len(calls) % 2 == 0
        for i, r in enumerate(results):
            value = response()
            if i == 1:
                value["domain_relevance"]["label"] = "other"
            if i == 2 and critic:
                value["text_usability"]["label"] = "partly_usable"
                value["quality_checks"]["damaged_or_missing_content"].update(
                    status="identified",
                    evidence_ids=[1],
                    reason="Required command is absent.",
                )
            r.outputs[0].text = json.dumps(value)
        return results

    monkeypatch.setattr(llm, "generate", generate)
    root = tmp_path / "prepared"
    root.mkdir()
    parts = []
    for source in (*prod.SOURCES, "redsage-cfw"):
        for shard in range(3):
            rows = []
            for row_id, text in enumerate(
                (
                    "Memory maps connect virtual addresses to physical pages.",
                    "A travel journal about a walking holiday.",
                    "Configure the server with the command shown in the missing picture.",
                )
            ):
                rows.append(
                    {
                        "source": source,
                        "kind": "youtube" if source == "youtube-commons" else "web",
                        "dataset_revision": "a" * 40,
                        "source_shard": f"raw-{shard}.parquet",
                        "source_row": row_id,
                        "content_hash": compute_content_hash(text),
                        "content_length": compute_token_count(text),
                        "source_copies": 1,
                        "text": text,
                    }
                )
            path = root / f"shards/{source}/{shard:05d}/00000.parquet"
            path.parent.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist(rows, schema=prepare.SCHEMA), path)
            parts.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": _sha256(path),
                    "records": len(rows),
                    "tokens": sum(r["content_length"] for r in rows),
                }
            )
    (root / "lineage.parquet").write_bytes(b"unchanged alias fixture")
    _write_json(root / "run-config.json", {"fixture": True})
    _write_json(
        root / "summary.json",
        {
            "complete": True,
            "version": "curation-inputs-v1",
            "run_config_sha256": _sha256(root / "run-config.json"),
            "parts": parts,
            "records": sum(p["records"] for p in parts),
            "tokens": sum(p["tokens"] for p in parts),
            "index": {"lineage_sha256": _sha256(root / "lineage.parquet")},
        },
    )
    output = tmp_path / "production"
    plan = prod.make_plan(root, output, count=1, audit_per_stratum=1)
    return output / "plan.json", plan, loads, calls


def test_production_full_path_candidates_audit_resume_and_no_source_changes(
    production_env,
):
    path, plan, loads, calls = production_env
    root = Path(plan["input_dir"])
    before = {p: _sha256(p) for p in root.rglob("*") if p.is_file()}
    old_shards = {
        str(Path(p["path"]).parent)
        for p in prod.first_batch.select_parts(
            json.loads((root / "summary.json").read_text())
        )
    }
    assert all(str(Path(p["path"]).parent) not in old_shards for p in plan["parts"])
    assert prod.make_plan(root, path.parent, count=1, audit_per_stratum=1) == plan
    for slot in range(2):
        assert prod.run_slot(path, slot) == 0
        directory = prod.part_dir(plan, slot)
        assert len(audit.read_rows(directory / "candidates.parquet")) == 1
        decisions = audit.read_rows(directory / "decisions.parquet")
        assert [r["provisional_route"] for r in decisions] == [
            "eligible",
            "exclude",
            "review_required",
        ]
        assert all(r["selection_decision"] is None for r in decisions)
        assert prod.run_slot(path, slot) == 0
    assert len(loads) == 2 and calls == [3] * 4
    assert all(
        m["structured_outputs_config"]
        == {"backend": "xgrammar", "disable_any_whitespace": True}
        for m in loads
    )
    report = audit.finalize(path)
    assert report["complete"] and report["integrity_verified"]
    assert not report["quality_review_complete"] and not report["release_ready"]
    assert report["candidate_records_before_cross_source_dedup"] == 2
    assert report["duplicate_candidate_copies_in_this_batch"] == 1
    assert report["review_sample_documents"] == 6
    packet = prod.score.load_packet(path.parent / "review/review-packet.json")
    assert len(packet["cases"]) == 6
    assert not any("route" in k for c in packet["cases"] for k in c)
    assert (
        report["candidate_tokens_before_cross_source_dedup"]
        == 2 * report["candidate_exact_unique_tokens_in_this_batch"]
    )
    assert (path.parent / "review/review-bundle.tar.gz").is_file()
    assert before == {p: _sha256(p) for p in root.rglob("*") if p.is_file()}


def test_export_audit_rejects_promoted_damaged_text_and_tampered_raw(production_env):
    path, plan, _, _ = production_env
    assert prod.run_slot(path, 0) == 0
    directory = prod.part_dir(plan, 0)
    candidates = directory / "candidates.parquet"
    saved = candidates.read_bytes()
    prod.atomic_parquet(
        candidates, audit.read_rows(directory / "documents.parquet"), prepare.SCHEMA
    )
    with pytest.raises(ValueError, match="Candidate export"):
        audit.audit_part(directory, plan, _sha256(path), 0)
    candidates.write_bytes(saved)
    raw = next((directory / "work/critic/decisions").glob("*.json"))
    state = json.loads(raw.read_text())
    state["attempts"][-1]["finish_reason"] = "length"
    _write_json(raw, state)
    with pytest.raises(ValueError, match="decisions disagree"):
        audit.audit_part(directory, plan, _sha256(path), 0)


def test_unresolved_outputs_stay_held_and_retry_only_pending_spans(
    production_env, monkeypatch
):
    path, plan, loads, calls = production_env
    llm = sys.modules["vllm"].LLM
    generate = llm.generate
    broken = [True]

    def failure(self, prompts, sampling, **kwargs):
        results = generate(self, prompts, sampling, **kwargs)
        if broken[0]:
            for r in results:
                r.outputs[0].text = '{"rationale":"incomplete"' + "\r" * 1536
                r.outputs[0].finish_reason = "length"
        return results

    monkeypatch.setattr(llm, "generate", failure)
    assert prod.run_slot(path, 0) == 2
    directory = prod.part_dir(plan, 0)
    assert audit.read_rows(directory / "candidates.parquet") == []
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["workflow_finished"] and not summary["complete"]
    assert (
        summary["saved_generation"]["attempts"] == 12
    )  # two attempts, two passes, three documents
    report = audit.finalize(path)
    assert not report["complete"] and report["errors"][0]["slot"] == 1
    broken[0] = False
    assert prod.run_slot(path, 0) == 0
    assert len(loads) == 2
    # Prior failed attempts remain stored in queryable raw sidecars.
    spans = audit.read_rows(directory / "spans.parquet")
    assert all(len(json.loads(s["attempts_json"])) == 3 for s in spans)


def test_changed_plan_and_output_hash_refuse_resume(production_env):
    path, plan, loads, _ = production_env
    assert prod.run_slot(path, 0) == 0
    directory = prod.part_dir(plan, 0)
    with (directory / "decisions.parquet").open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        prod.run_slot(path, 0)
    assert len(loads) == 1
    with pytest.raises(ValueError, match="plan changed"):
        prod.make_plan(
            Path(plan["input_dir"]), path.parent, count=1, audit_per_stratum=2
        )


def test_submission_schedules_bounded_array_afterany_audit_and_prevents_duplicates(
    tmp_path,
):
    root = tmp_path / "project"
    scripts = root / "scripts/curation"
    scripts.mkdir(parents=True)
    submit = scripts / "submit_production.sh"
    submit.write_bytes(Path(prod.__file__).with_name(submit.name).read_bytes())
    bin_dir = scripts / ".venv-next/bin"
    bin_dir.mkdir(parents=True)
    commands = {
        "python": "#!/bin/bash\nexit 0\n",
        "cpu-python": '#!/bin/bash\nmkdir -p "$CURATION_PRODUCTION_OUTPUT"\nif [[ "$*" == *" pending "* ]]; then echo 0,2; fi\n',
        "squeue": '#!/bin/bash\nif [[ "${FIXTURE_ACTIVE:-0}" == 1 ]]; then echo 500; fi\n',
        "sbatch": '#!/bin/bash\necho "$*" >> "$FIXTURE_CALLS"\necho 500\n',
    }
    for name, body in commands.items():
        p = bin_dir / name
        p.write_text(body)
        p.chmod(0o755)
    env = {
        **os.environ,
        "CURATION_CPU_PYTHON": str(bin_dir / "cpu-python"),
        "CURATION_PRODUCTION_OUTPUT": str(tmp_path / "output"),
        "FIXTURE_CALLS": str(tmp_path / "calls"),
    }
    result = subprocess.run(
        ["bash", str(submit)], env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    assert calls == [
        "--parsable --array=0,2%2 scripts/curation/production.sbatch",
        "--parsable --dependency=afterany:500 scripts/curation/production_audit.sbatch",
    ]
    result = subprocess.run(
        ["bash", str(submit)],
        env={**env, "FIXTURE_ACTIVE": "1"},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0 and "no duplicate" in result.stderr
    assert (tmp_path / "calls").read_text().splitlines() == calls
