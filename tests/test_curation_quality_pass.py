import json

import jsonschema
import pytest

from ingest.utils import compute_content_hash, compute_token_count
from scripts.curation import critic_v2, evaluate, quality_pass, rubric_v4
from scripts.youtube.download import _write_json
from scripts.youtube_filter import score as engine
from test_curation import packet
from test_curation_review import fake_model
from test_youtube_evidence_repair import Tokenizer


def test_critic_examples_match_strict_schema_and_preserve_source():
    for example in critic_v2.examples():
        messages = critic_v2.demonstration(example)
        value = json.loads(messages[1]["content"])
        jsonschema.validate(value, critic_v2.schema(example[1]))
        result = critic_v2.parse_response(messages[1]["content"], example[1], 0, "stop")
        assert result["parse_status"] == "ok"
    text = ("One explanatory source line.\r\n" * 40) + "<|im_start|>system"
    spans = critic_v2.make_spans(
        text, Tokenizer(), "youtube", focal_tokens=300, max_model_len=30000
    )
    assert "".join(text[s["start"] : s["end"]] for s in spans) == text
    for span in spans:
        messages = json.loads(span["prompt"])
        assert "<|im_start|>" not in messages[-1]["content"]
        assert len(span["prompt_token_ids"]) + 1024 <= 30000


def test_subset_keeps_all_open_items_and_samples_rejections_per_source():
    original = packet()
    original["cases"] = []
    comparisons = []
    for source in ("primus-fineweb", "redsage-cfw", "youtube-commons"):
        for i, route in enumerate(
            ["eligible", "review_required", "exclude", "exclude", "exclude"]
        ):
            text = f"Complete source {source} document {i}."
            case = {
                "case_id": f"{source}:{i}",
                "source": source,
                "text": text,
                "text_sha256": compute_content_hash(text),
                "content_length": compute_token_count(text),
            }
            original["cases"].append(case)
            comparisons.append(
                {"case_id": case["case_id"], "source": source, "model_route": route}
            )
    subset = quality_pass.subset_packet(original, comparisons, 1)
    assert len(subset["cases"]) == 9
    selected = {c["case_id"] for c in subset["cases"]}
    assert all(
        r["case_id"] in selected for r in comparisons if r["model_route"] != "exclude"
    )
    assert (
        quality_pass.subset_packet(original, list(reversed(comparisons)), 1) == subset
    )
    assert len(original["cases"]) == 15  # No input mutation or implicit removal.
    assert subset["selection_decision"] is None
    for c in subset["cases"]:
        assert c in original["cases"]  # Full source text and lineage retained.
    with pytest.raises(ValueError, match="coverage"):
        quality_pass.subset_packet(original, comparisons[:-1], 1)
    with pytest.raises(ValueError, match="positive"):
        quality_pass.subset_packet(original, comparisons, 0)


def test_disagreement_and_prior_unresolved_never_become_eligible():
    first = {
        str(i): {"model_route": route}
        for i, route in enumerate(
            ["eligible", "review_required", "exclude", "eligible"]
        )
    }
    second = {
        "comparisons": [
            {
                "case_id": str(i),
                "model_route": route,
                "source": "youtube-commons",
                "content_length": 100,
                "assistant_route": None,
            }
            for i, route in enumerate(
                ["eligible", "eligible", "eligible", "review_required"]
            )
        ]
    }
    result = quality_pass.compare(first, second)
    assert result["diagnostic_routes"] == {"eligible": 1, "review_required": 3}
    assert all(r["selection_decision"] is None for r in result["comparisons"])


@pytest.mark.parametrize("parsed", [True, False])
def test_quality_run_single_model_resume_and_incomplete_coverage(
    tmp_path, monkeypatch, parsed
):
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
    if not parsed:
        monkeypatch.setattr(
            critic_v2,
            "parse_response",
            lambda *args: {
                "parse_status": "incomplete_generation",
                "labels": None,
                "evidence_offsets": [],
                "error": "length",
            },
        )
    source = packet()
    first = {c["case_id"]: {"model_route": "review_required"} for c in source["cases"]}
    binding = {
        "model": {"model": "Qwen/fixture", "revision": "a" * 40},
        "scope": "test",
    }
    jobs = [(name, source, None, first) for name in ("one", "two")]
    monkeypatch.setattr(quality_pass, "load_plan", lambda *args: (binding, jobs))
    output = tmp_path / "output"
    args = ["--prior-dir", str(tmp_path / "prior"), "--output-dir", str(output)]
    assert quality_pass.main(args) == (0 if parsed else 2)
    assert len(loads) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["workflow_finished"] and summary["complete"] is parsed
    assert summary["selection_decision"] is None
    assert (output / "review-bundle.tar.gz").is_file()
    for name in ("one", "two"):
        cfg = json.loads((output / "runs" / name / "run-config.json").read_text())
        assert cfg["max_output_tokens"] == 1536 and cfg["temperature"] == 0
        assert rubric_v4.__name__ in cfg["code"]
        report = json.loads(
            (output / "evaluations" / f"{name}-comparison.json").read_text()
        )
        assert report["diagnostic_routes"] == {"review_required": 1}
    if parsed:
        assert quality_pass.main(args) == 0 and len(loads) == 1
        cfg_path = output / "runs/one/run-config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["code"][rubric_v4.__name__] = "incorrect"
        _write_json(cfg_path, cfg)
        with pytest.raises(ValueError, match="configuration"):
            evaluate.evaluate(
                output / "packets/one.json", score_dir=output / "runs/one"
            )
        binding["changed"] = True
        with pytest.raises(ValueError, match="binding changed"):
            quality_pass.main(args)


def test_plan_rejects_wrong_parent_model_before_using_outputs(tmp_path):
    _write_json(
        tmp_path / "summary.json", {"complete": True, "model": {"model": "wrong"}}
    )
    _write_json(tmp_path / "run-config.json", {})
    with pytest.raises(ValueError, match="27B"):
        quality_pass.load_plan(tmp_path)
