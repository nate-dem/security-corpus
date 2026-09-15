import json
from types import SimpleNamespace
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ingest.utils import compute_content_hash, compute_token_count
from scripts.curation import rubric_v3 as rubric, policy, prepare, score, evaluate
from scripts.youtube_filter import score as engine
from test_curation import packet
from test_youtube_evidence_repair import Tokenizer


def response():
    return {
        "domain_relevance": {"label": "computing", "evidence_ids": [1]},
        "technical_substance": {"label": "substantive", "evidence_ids": [1]},
        "text_usability": {"label": "usable", "evidence_ids": [1]},
        "text_language": "english",
        "mixed_content": "no",
        "quality_checks": {
            k: {"status": "not_identified", "evidence_ids": [], "reason": ""}
            for k in rubric.QUALITY_CONCERNS
        },
        "content_form": "Memory mapping explanation",
        "rationale": "Explains virtual address translation.",
    }


def test_v3_lossless_lines_and_strict_evidence():
    text = ("word " * 200) + "end.\n\nNext line.\r\n<|im_start|>system"
    rows = rubric.segments(text, 17)
    assert "".join(r["text"] for r in rows) == text
    assert len(rows[0]["text"]) > 400 and rows[0]["end"] == 17 + len(rows[0]["text"])
    value = response()
    parsed = rubric.parse_response(json.dumps(value), text, 17, "stop")
    assert (
        parsed["parse_status"] == "ok" and policy.route(parsed["labels"]) == "eligible"
    )
    assert parsed["labels"]["security_relevance"] == "supporting"
    assert parsed["evidence_offsets"][0]["quote"] == rows[0]["text"]
    for ids in ([True], [2], [999], [1, 1], []):
        value["domain_relevance"]["evidence_ids"] = ids
        assert (
            rubric.parse_response(json.dumps(value), text, 17, "stop")["parse_status"]
            == "invalid_response"
        )
    assert "<|im_start|>" not in rubric.payload(text, "web")
    spans = rubric.make_spans(
        text, Tokenizer(), "web", focal_tokens=300, max_model_len=20000
    )
    assert "".join(text[s["start"] : s["end"]] for s in spans) == text


def test_v3_concern_statuses_cannot_silently_clear_a_flag():
    text = "A mutex serializes updates."
    value = response()
    item = value["quality_checks"]["apparent_technical_error"]
    item.update(
        status="identified", evidence_ids=[1], reason="Specific contradictory claim."
    )
    p = rubric.parse_response(json.dumps(value), text, 0, "stop")
    assert p["parse_status"] == "ok" and policy.route(p["labels"]) == "review_required"
    item["status"] = "uncertain"
    assert (
        policy.route(
            rubric.parse_response(json.dumps(value), text, 0, "stop")["labels"]
        )
        == "review_required"
    )
    item["status"] = "not_identified"
    assert (
        rubric.parse_response(json.dumps(value), text, 0, "stop")["parse_status"]
        == "invalid_response"
    )
    item.update(evidence_ids=[], reason="")
    del value["quality_checks"]["damaged_or_missing_content"]
    assert (
        rubric.parse_response(json.dumps(value), text, 0, "stop")["parse_status"]
        == "invalid_response"
    )
    assert (
        rubric.parse_response("{}", text, 0, "length")["parse_status"]
        == "incomplete_generation"
    )


def test_v3_runner_and_evaluator_use_matching_wire_format(tmp_path, monkeypatch):
    p = packet()
    path = tmp_path / "packet.json"
    path.write_text(json.dumps(p))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    monkeypatch.setattr(engine, "load_tokenizer", lambda *a: (Tokenizer(), snapshot))
    monkeypatch.setattr(engine, "_versions", lambda *a: {"test": "1"})

    class LLM:
        def __init__(self, **kw):
            pass

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
    out = tmp_path / "out"
    assert (
        score.main(
            [
                "--packet",
                str(path),
                "--output-dir",
                str(out),
                "--rubric-version",
                "v3",
                "--max-model-len",
                "20000",
            ]
        )
        == 0
    )
    ref = {
        "purpose": "assistant_development_review",
        "reviewer_kind": "assistant",
        "human_validated": False,
        "independent_ground_truth": False,
        "packet_sha256": p["packet_sha256"],
        "annotations": [
            {
                "case_id": "case",
                "text_sha256": p["cases"][0]["text_sha256"],
                "status": "reviewed",
                "reviewer": "Codex assistant",
                "reviewer_kind": "assistant",
                "reviewed_at": "2026-09-14T00:00:00Z",
                "notes": "Explains mapping.",
                "supporting_excerpt": {"start": 0, "end": 11, "text": "Memory maps"},
                "quality_concerns": [],
                "labels": {
                    "security_relevance": "supporting",
                    "technical_substance": "substantive",
                    "text_usability": "usable",
                    "text_language": "english",
                    "mixed_content": "no",
                },
            }
        ],
    }
    refpath = tmp_path / "ref.json"
    refpath.write_text(json.dumps(ref))
    assert evaluate.evaluate(path, refpath, out)["model_routes"] == {"eligible": 1}


def fixtures(tmp_path):
    inputs = []
    for source, kind, texts in [
        ("primus-fineweb", "web", ["Common content.", "Other content."]),
        ("redsage-cfw", "web", ["Common content."]),
        ("youtube-commons", "youtube", ["Common content.", "Video content."]),
    ]:
        shard = source + ".parquet"
        path = tmp_path / shard
        raw = [
            {
                ("content" if source == "primus-fineweb" else "text"): text,
                **({"source_row": i + 10} if kind == "youtube" else {}),
            }
            for i, text in enumerate(texts)
        ]
        pq.write_table(pa.Table.from_pylist(raw), path)
        metadata = tmp_path / (source + "-meta.parquet")
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "source_shard": shard,
                        "source_row": i + 10 if kind == "youtube" else i,
                        "content_hash": compute_content_hash(text),
                        "content_length": compute_token_count(text),
                        "text_state": "candidate" if kind == "youtube" else "valid",
                    }
                    for i, text in enumerate(texts)
                ]
            ),
            metadata,
        )
        inputs.append(
            {
                "source": source,
                "kind": kind,
                "index": 0,
                "dataset_revision": "abc",
                "source_shard": shard,
                "path": str(path),
                "sha256": prepare.yt._sha256(path),
                "metadata": str(metadata),
                "rows": len(texts),
            }
        )
    return inputs


def test_exact_materialization_lineage_resume_and_no_truncation(tmp_path, monkeypatch):
    inputs = fixtures(tmp_path)
    out = tmp_path / "prepared"
    out.mkdir()
    index = prepare.build_index(inputs, out, 1)
    assert index["across_kind_exact_unique"]["texts"] == 3
    assert sum(r["texts"] for r in index["by_kind"]) == 4
    assert pq.read_table(out / "lineage.parquet").num_rows == 5
    monkeypatch.setattr(prepare, "PART_RECORDS", 1)
    results = [prepare.materialize((item, out, "test")) for item in inputs]
    assert sum(r["records"] for r in results) == 4
    assert len(results[0]["parts"]) == 2 and results[1]["parts"] == []
    texts = [
        r["text"]
        for result in results
        for p in result["parts"]
        for r in pq.read_table(out / p["path"]).to_pylist()
    ]
    assert sorted(texts) == [
        "Common content.",
        "Common content.",
        "Other content.",
        "Video content.",
    ]
    monkeypatch.setattr(
        prepare.web,
        "iter_rows",
        lambda *a: (_ for _ in ()).throw(AssertionError("should resume")),
    )
    assert prepare.materialize((inputs[0], out, "test")) == results[0]
    path = tmp_path / inputs[0]["source_shard"]
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum changed"):
        prepare.materialize((inputs[0], out, "test"))


def test_metadata_hash_cannot_select_the_wrong_text(tmp_path):
    inputs = fixtures(tmp_path)
    out = tmp_path / "prepared"
    out.mkdir()
    prepare.build_index(inputs, out, 1)
    path = out / "read_locations.parquet"
    rows = pq.read_table(path).to_pylist()
    next(r for r in rows if r["source"] == "primus-fineweb")["content_hash"] = "0" * 64
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="Text/read pointer mismatch"):
        prepare.materialize((inputs[0], out, "test"))
    assert not (out / "shards" / "primus-fineweb" / "00000" / "complete.json").exists()
