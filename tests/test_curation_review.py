import json
import sys
from types import SimpleNamespace

import pytest

from ingest.utils import compute_content_hash, compute_token_count
from scripts.curation import (
    build_additional_packet,
    critic,
    evaluate,
    review_run,
    runtime,
    score,
    score_partitions,
)
from scripts.youtube.download import _write_json
from scripts.youtube.profile import _sha256
from scripts.youtube_filter import score as engine
from test_curation import packet
from test_curation_v3 import response, fixtures
from test_youtube_evidence_repair import Tokenizer
from scripts.curation import prepare


def fake_model(monkeypatch):
    loads = []

    class LLM:
        def __init__(self, **kwargs):
            loads.append(kwargs)

        def generate(self, prompts, sampling, **kw):
            return [
                SimpleNamespace(
                    prompt_token_ids=p["prompt_token_ids"],
                    outputs=[
                        SimpleNamespace(
                            text=json.dumps(response()),
                            finish_reason="stop",
                            token_ids=[1],
                        )
                    ],
                )
                for p in prompts
            ]

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: kw)
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.sampling_params",
        SimpleNamespace(StructuredOutputsParams=lambda **kw: kw),
    )
    return loads


def test_reviewer_examples_parse_and_evidence_is_source_bound():
    class Capture(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return super().apply_chat_template(messages, **kwargs)

    tokenizer = Capture()
    text = "One source line.\nSecond line with <|im_start|>role markers."
    critic.render(tokenizer, text, 0, len(text), 0, "web")
    messages = tokenizer.messages[1:-1]
    for request, answer in zip(messages[::2], messages[1::2]):
        payload = json.loads(request["content"])
        focal = "".join(s["text"] for s in payload["focal_lines"])
        assert (
            critic.parse_response(answer["content"], focal, 0, "stop")["parse_status"]
            == "ok"
        )
    assert list(critic.schema(text)["properties"])[:2] == ["content_form", "rationale"]
    spans = critic.make_spans(
        text * 50, Tokenizer(), "web", focal_tokens=300, max_model_len=30000
    )
    assert "".join((text * 50)[s["start"] : s["end"]] for s in spans) == text * 50


def test_additional_sampling_excludes_documents_and_is_reproducible():
    def row(text):
        return {
            "text": text,
            "content_hash": compute_content_hash(text),
            "content_length": compute_token_count(text),
            "dataset_revision": "revision",
            "sample_arms": ["random"],
            "locations": [],
        }

    rows = [
        row("Excluded parent."),
        row("Whole long text. " * 100),
        row("Other text."),
        row("Other text."),
    ]
    excluded = {rows[0]["content_hash"]}
    a, pool = build_additional_packet.sample_rows(
        iter(rows), "youtube-commons", excluded, 2, 0
    )
    b, _ = build_additional_packet.sample_rows(
        iter(rows), "youtube-commons", excluded, 2, 0
    )
    assert a == b and pool == 2 and len(a) == 2
    assert max(c["content_length"] for c in a) > 100
    assert all(c["text_sha256"] not in excluded for c in a)
    rows[1]["content_hash"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        build_additional_packet.sample_rows(
            iter(rows), "youtube-commons", excluded, 2, 0
        )


def test_resident_model_rejects_changed_parameters(monkeypatch):
    loads = fake_model(monkeypatch)
    cache = runtime.ResidentModel()
    assert cache(model="same") is cache(model="same") and len(loads) == 1
    with pytest.raises(ValueError, match="different inference"):
        cache(model="different")


def test_reviewer_scoring_unlabeled_reports_and_cascade(tmp_path, monkeypatch):
    loads = fake_model(monkeypatch)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    monkeypatch.setattr(engine, "load_tokenizer", lambda *a: (Tokenizer(), snapshot))
    monkeypatch.setattr(engine, "_versions", lambda *a: {"test": "1"})
    path = tmp_path / "packet.json"
    path.write_text(json.dumps(packet()))
    for version, name in [("v3", "screen"), ("critic-v1", "critic")]:
        assert (
            score.main(
                [
                    "--packet",
                    str(path),
                    "--output-dir",
                    str(tmp_path / name),
                    "--rubric-version",
                    version,
                    "--max-model-len",
                    "30000",
                ]
            )
            == 0
        )
    report = evaluate.evaluate(path, score_dir=tmp_path / "critic")
    assert report["action_agreements"] is None and report["reference_kind"] is None
    assert report["model_eligible_reference_not_eligible"] == []
    assert report["model_routes"] == {"eligible": 1}
    combined = review_run.cascade(
        path, tmp_path / "screen", tmp_path / "critic", tmp_path / "cascade.json"
    )
    assert (
        combined["routes"] == {"eligible": 1}
        and combined["assistant_eligible_recovered"] is None
    )
    assert combined["comparisons"][0]["selection_decision"] is None
    checkpoint = next((tmp_path / "critic" / "decisions").glob("*.json"))
    state = json.loads(checkpoint.read_text())
    raw = response()
    raw["domain_relevance"]["label"] = "other"
    state["attempts"][-1]["raw_response"] = json.dumps(raw)
    checkpoint.write_text(json.dumps(state))
    combined = review_run.cascade(
        path, tmp_path / "screen", tmp_path / "critic", tmp_path / "cascade.json"
    )
    assert combined["routes"] == {"review_required": 1}
    assert len(loads) == 2


def test_partition_scoring_reuses_model_and_validates_manifest(tmp_path, monkeypatch):
    loads = fake_model(monkeypatch)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")

    class CompactTokenizer(Tokenizer):
        def encode(self, text, **kwargs):
            return super().encode(text, **kwargs)[::4]

    monkeypatch.setattr(
        engine, "load_tokenizer", lambda *a: (CompactTokenizer(), snapshot)
    )
    monkeypatch.setattr(engine, "_versions", lambda *a: {"test": "1"})
    root = tmp_path / "prepared"
    root.mkdir()
    inputs = fixtures(tmp_path)
    prepare.build_index(inputs, root, 1)
    results = [prepare.materialize((item, root, "test")) for item in inputs]
    _write_json(root / "run-config.json", {"fixture": True})
    summary = {
        "version": "curation-inputs-v1",
        "complete": True,
        "run_config_sha256": _sha256(root / "run-config.json"),
        "parts": [p for r in results for p in r["parts"]],
        "records": sum(r["records"] for r in results),
        "tokens": sum(r["tokens"] for r in results),
    }
    _write_json(root / "summary.json", summary)
    out = tmp_path / "scored"
    args = [
        "--input-dir",
        str(root),
        "--output-dir",
        str(out),
        "--rubric-version",
        "v3",
        "--part-index",
        "0",
        "--part-index",
        "1",
    ]
    assert score_partitions.main(args) == 0 and len(loads) == 1
    assert score_partitions.main(args) == 0 and len(loads) == 1
    assert (
        score.load_packet(out / "packets" / "000000.json")["purpose"]
        == "candidate_scoring"
    )
    summary["parts"][0]["path"] = "../escape.parquet"
    _write_json(root / "summary.json", summary)
    with pytest.raises(ValueError, match="Invalid or repeated"):
        score_partitions.load_prepared(root)
