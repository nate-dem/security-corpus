import hashlib
import json
import sys

import pytest

from scripts.curation import quality_pass, retry_quality, score
from scripts.youtube.download import _write_json
from scripts.youtube.profile import _sha256
from scripts.youtube_filter import score as engine
from test_curation import packet
from test_curation_review import fake_model
from test_youtube_evidence_repair import Tokenizer


@pytest.fixture
def failed_quality_run(tmp_path, monkeypatch):
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
        quality_pass.cuda_preflight, "configure_and_check", lambda: {"fixture": "nvcc"}
    )
    monkeypatch.setattr(
        quality_pass.cuda_preflight, "check_sampler", lambda: {"fixture": "native"}
    )
    llm = sys.modules["vllm"].LLM
    generate = llm.generate
    calls = []

    def one_whitespace_failure(self, prompts, sampling, **kwargs):
        results = generate(self, prompts, sampling, **kwargs)
        calls.append(len(prompts))
        if len(calls) == 3:
            results[0].outputs[0].text = '{"rationale":"incomplete"' + "\r" * 1536
            results[0].outputs[0].finish_reason = "length"
        return results

    monkeypatch.setattr(llm, "generate", one_whitespace_failure)
    source = packet()
    other = packet("Page tables associate virtual page numbers with physical frames.")
    other["cases"][0]["case_id"] = "other"
    source["cases"] += other["cases"]
    source.pop("packet_sha256")
    source["packet_sha256"] = hashlib.sha256(
        json.dumps(source, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    first = {c["case_id"]: {"model_route": "eligible"} for c in source["cases"]}
    model = json.loads((quality_pass.HERE / "next_models.json").read_text())["models"][
        1
    ]
    monkeypatch.setattr(
        quality_pass,
        "load_plan",
        lambda *args: (
            {"model": model, "scope": "test"},
            [(name, source, None, first) for name in quality_pass.PACKETS],
        ),
    )
    prior = tmp_path / "prior"
    assert (
        quality_pass.main(
            ["--prior-dir", str(tmp_path / "first"), "--output-dir", str(prior)]
        )
        == 2
    )
    assert len(loads) == 1 and calls == [2] * 5
    return prior, loads, calls


def test_retry_only_failed_document_compact_json_resume_and_originals_unchanged(
    failed_quality_run, tmp_path
):
    prior, loads, calls = failed_quality_run
    before = {p: _sha256(p) for p in prior.rglob("*") if p.is_file()}
    binding, jobs = retry_quality.load_plan(prior)
    name = quality_pass.PACKETS[2]
    assert binding["prior_total_spans"] == 10
    assert len(jobs) == 1 and jobs[0][0] == name
    assert [c["case_id"] for c in jobs[0][1]["cases"]] == ["case"]
    assert binding["packets"][name]["old_unresolved_spans"] == 1
    assert len(binding["packets"][name]["prior_decision_sha256"]) == 1
    original = score.load_packet(prior / "packets" / f"{name}.json")
    assert jobs[0][1]["cases"][0] == original["cases"][0]
    output = tmp_path / "retry"
    args = ["--prior-dir", str(prior), "--output-dir", str(output)]
    assert retry_quality.main(args + ["--check-env"]) == 0
    assert len(loads) == 1 and not output.exists()
    assert retry_quality.main(args) == 0
    assert calls == [2] * 5 + [1] and len(loads) == 2
    assert loads[-1]["structured_outputs_config"] == {
        "backend": "xgrammar",
        "disable_any_whitespace": True,
    }
    config = json.loads((output / "runs" / name / "run-config.json").read_text())
    assert config["compact_json"] and config["max_output_tokens"] == 1536
    summary = json.loads((output / "summary.json").read_text())
    assert summary["complete"] and summary["scoring_coverage_complete"]
    assert summary["selection_decision"] is None
    assert (output / "review-bundle.tar.gz").is_file()
    assert retry_quality.main(args) == 0 and len(loads) == 2
    assert before == {p: _sha256(p) for p in prior.rglob("*") if p.is_file()}


def test_retry_rejects_changed_prompt_binding(
    failed_quality_run, tmp_path, monkeypatch
):
    prior, _, _ = failed_quality_run
    binding, jobs = retry_quality.load_plan(prior)
    config = next(iter(binding["packets"].values()))
    key = config["expected_span_keys"][0]
    config["expected_task_sha256"][key] = "0" * 64
    monkeypatch.setattr(retry_quality, "load_plan", lambda *args: (binding, jobs))
    with pytest.raises(ValueError, match="spans or prompts changed"):
        retry_quality.main(
            ["--prior-dir", str(prior), "--output-dir", str(tmp_path / "retry")]
        )
    summary = json.loads((tmp_path / "retry/summary.json").read_text())
    assert not summary["complete"]


def test_retry_rejects_incompatible_prior_before_gpu(failed_quality_run):
    prior, loads, _ = failed_quality_run
    summary = json.loads((prior / "summary.json").read_text())
    summary["model"]["revision"] = "f" * 40
    _write_json(prior / "summary.json", summary)
    with pytest.raises(ValueError, match="27B quality-pass"):
        retry_quality.load_plan(prior)
    assert len(loads) == 1
