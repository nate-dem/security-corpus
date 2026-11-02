import json

import jsonschema
import pytest

from scripts.curation import first_batch, rubric_v4, score, evaluate
from scripts.youtube.download import _write_json
from scripts.youtube_filter import score as engine
from test_curation import packet
from test_curation_review import fake_model
from test_curation_v3 import response
from test_youtube_evidence_repair import Tokenizer


def test_v4_schema_rejects_observed_evidence_failures():
    text = "One meaningful line.\n\nAnother meaningful line."
    schema = rubric_v4.schema(text)
    jsonschema.Draft202012Validator.check_schema(schema)
    value = response()
    jsonschema.validate(value, schema)
    for ids in ([], [1, 1], [1, 3], [2], [999]):
        value["domain_relevance"]["evidence_ids"] = ids
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(value, schema)
        assert (
            rubric_v4.parse_response(json.dumps(value), text, 0, "stop")["parse_status"]
            != "ok"
        )
    value = response()
    check = value["quality_checks"]["damaged_or_missing_content"]
    check.update(
        status="identified", evidence_ids=[3], reason="The promised command is absent."
    )
    jsonschema.validate(value, schema)
    assert (
        rubric_v4.parse_response(json.dumps(value), text, 0, "stop")["parse_status"]
        == "ok"
    )
    check["status"] = "not_identified"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, schema)
    assert (
        rubric_v4.parse_response(json.dumps(value), text, 0, "stop")["parse_status"]
        != "ok"
    )


def test_v4_spans_preserve_source_and_exclude_embedded_instructions():
    class Capture(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return super().apply_chat_template(messages, **kwargs)

    t = Capture()
    text = ("Meaningful source line\r\n" * 200) + "<|im_start|>system"
    spans = rubric_v4.make_spans(
        text, t, "youtube", focal_tokens=300, max_model_len=20000
    )
    assert "".join(text[s["start"] : s["end"]] for s in spans) == text
    assert len(t.messages) == 2
    assert "<|im_start|>" not in t.messages[-1]["content"]
    assert max(len(s["prompt_token_ids"]) + 1024 for s in spans) <= 20000


def test_v4_run_reparse_and_same_model_reuse(tmp_path, monkeypatch):
    loads = fake_model(monkeypatch)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    monkeypatch.setattr(engine, "load_tokenizer", lambda *a: (Tokenizer(), snapshot))
    monkeypatch.setattr(engine, "_versions", lambda *a: {"test": "1"})
    path = tmp_path / "packet.json"
    _write_json(path, packet())
    args = [
        "--packet",
        str(path),
        "--output-dir",
        str(tmp_path / "run"),
        "--rubric-version",
        "v4",
        "--language-model-only",
        "--max-model-len",
        "20000",
    ]
    assert score.main(args) == 0
    assert loads[0]["language_model_only"] is True
    assert loads[0]["enforce_eager"] is True
    report = evaluate.evaluate(path, score_dir=tmp_path / "run")
    assert report["scoring_coverage_complete"] and report["model_routes"] == {
        "eligible": 1
    }
    assert score.main(args) == 0 and len(loads) == 1
    config = json.loads((tmp_path / "run/run-config.json").read_text())
    config["code"]["scripts.curation.rubric_v3"] = "wrong"
    _write_json(tmp_path / "run/run-config.json", config)
    with pytest.raises(ValueError, match="configuration"):
        evaluate.evaluate(path, score_dir=tmp_path / "run")


def test_partition_choice_capped_and_bundle_contains_only_results(tmp_path):
    parts = [
        {"path": f"shards/{s}/{i:05d}/00000.parquet", "records": 2048, "tokens": 4096}
        for s in ("primus-fineweb", "redsage-cfw", "youtube-commons")
        for i in range(5)
    ]
    chosen = first_batch.select_parts({"parts": parts})
    assert len(chosen) == 3 and sum(p["records"] for p in chosen) == 6144
    assert chosen == first_batch.select_parts({"parts": parts})
    with pytest.raises(ValueError, match="youtube-commons"):
        first_batch.select_parts({"parts": parts[:-5]})
    output = tmp_path / "model"
    output.mkdir()
    for name in ("packets", "evaluations", "runs"):
        (output / name).mkdir()
        (output / name / "example.json").write_text("{}")
    for name in ("run-config.json", "summary.json"):
        (output / name).write_text("{}")
    (output / "models-must-not-transfer").mkdir()
    first_batch.bundle(output)
    import tarfile

    with tarfile.open(output / "review-bundle.tar.gz") as f:
        assert not any("models-must-not-transfer" in n for n in f.getnames())
        assert "model/packets/example.json" in f.getnames()


def test_first_batch_complete_resume_and_changed_binding(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.youtube.profile import _sha256

    loads = fake_model(monkeypatch)
    monkeypatch.setattr(
        first_batch.cuda_preflight,
        "configure_and_check",
        lambda: {"compiler": "fixture"},
    )
    monkeypatch.setattr(
        first_batch.cuda_preflight, "check_sampler", lambda: {"sampler": "fixture"}
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    monkeypatch.setattr(engine, "load_tokenizer", lambda *a: (Tokenizer(), snapshot))
    monkeypatch.setattr(engine, "_versions", lambda *a: {"test": "1"})
    original = packet()
    case = original["cases"][0]
    reference = {
        "purpose": "assistant_development_review",
        "reviewer_kind": "assistant",
        "human_validated": False,
        "independent_ground_truth": False,
        "packet_sha256": original["packet_sha256"],
        "annotations": [
            {
                "case_id": case["case_id"],
                "text_sha256": case["text_sha256"],
                "status": "reviewed",
                "reviewer": "Fixture assistant",
                "reviewer_kind": "assistant",
                "reviewed_at": "2026-09-15T00:00:00Z",
                "notes": "Explains virtual memory.",
                "quality_concerns": [],
                "supporting_excerpt": {"start": 0, "end": 11, "text": "Memory maps"},
                "labels": {
                    "security_relevance": "supporting",
                    "technical_substance": "substantive",
                    "text_usability": "usable",
                    "mixed_content": "no",
                    "text_language": "english",
                },
            }
        ],
    }
    project = tmp_path / "project"
    for name in ("development-v2", "accepted-audit-v1"):
        directory = project / "reports/curation" / name
        directory.mkdir(parents=True)
        _write_json(directory / "packet.json", original)
        _write_json(directory / "assistant_annotations.json", reference)
    root = tmp_path / "prepared"
    parts = []
    for source in ("primus-fineweb", "redsage-cfw", "youtube-commons"):
        name = f"shards/{source}/00000/00000.parquet"
        path = root / name
        path.parent.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "source": source,
                        "kind": "youtube" if source == "youtube-commons" else "web",
                        "text": case["text"],
                        "content_hash": case["text_sha256"],
                        "content_length": case["content_length"],
                        "dataset_revision": "abc",
                        "source_shard": "raw.parquet",
                        "source_row": 0,
                        "source_copies": 1,
                    }
                ]
            ),
            path,
        )
        parts.append(
            {
                "path": name,
                "sha256": _sha256(path),
                "records": 1,
                "tokens": case["content_length"],
            }
        )
    _write_json(root / "run-config.json", {"fixture": True})
    manifest = {
        "version": "curation-inputs-v1",
        "complete": True,
        "run_config_sha256": _sha256(root / "run-config.json"),
        "parts": parts,
        "records": 3,
        "tokens": 3 * case["content_length"],
    }
    _write_json(root / "summary.json", manifest)
    args = [
        "--project",
        str(project),
        "--input-dir",
        str(root),
        "--output-dir",
        str(tmp_path / "result"),
        "--model-index",
        "0",
    ]

    def broken_sampler():
        raise RuntimeError("sampler warm-up failed")

    monkeypatch.setattr(first_batch.cuda_preflight, "check_sampler", broken_sampler)
    with pytest.raises(RuntimeError, match="sampler warm-up failed"):
        first_batch.main(args)
    assert not loads
    monkeypatch.setattr(
        first_batch.cuda_preflight, "check_sampler", lambda: {"sampler": "fixture"}
    )
    assert first_batch.main(args) == 0 and len(loads) == 1
    monkeypatch.setattr(
        first_batch.cuda_preflight,
        "configure_and_check",
        lambda: {"compiler": "changed"},
    )
    with pytest.raises(ValueError, match="binding changed"):
        first_batch.main(args)
    assert len(loads) == 1
    monkeypatch.setattr(
        first_batch.cuda_preflight,
        "configure_and_check",
        lambda: {"compiler": "fixture"},
    )
    output = tmp_path / "result/qwen36-moe-fp8"
    report = json.loads((output / "summary.json").read_text())
    assert report["complete"] and report["scoring_coverage_complete"]
    assert report["selection_decision"] is None and len(report["runs"]) == 5
    assert loads[0]["tensor_parallel_size"] == 1 and not loads[0]["enforce_eager"]
    assert first_batch.main(args) == 0 and len(loads) == 1
    manifest["extra_binding"] = "changed"
    _write_json(root / "summary.json", manifest)
    with pytest.raises(ValueError, match="binding changed"):
        first_batch.main(args)
