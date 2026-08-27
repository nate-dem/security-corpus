import pytest

from classify.qwen import (
    QwenTask,
    build_messages,
    make_qwen_sidecar_row,
    parse_qwen_response,
)
from scripts.classify.score_qwen_vllm import (
    _next_part_index,
    _validate_model_revision,
)


def test_parse_qwen_response_strict_json():
    parsed = parse_qwen_response(
        '{"security_relevance": 3, "quality": 2, "should_keep": true, "reason": "useful"}'
    )

    assert parsed.parse_status == "ok"
    assert parsed.security_relevance == 3
    assert parsed.quality == 2
    assert parsed.should_keep is True
    assert parsed.reason == "useful"


def test_parse_qwen_response_extracts_first_json_object():
    parsed = parse_qwen_response(
        '```json\n{"security_relevance": 2, "quality": 1, "should_keep": false, "reason": "thin"}\n```'
    )

    assert parsed.parse_status == "extracted_json"
    assert parsed.security_relevance == 2
    assert parsed.quality == 1
    assert parsed.should_keep is False


def test_parse_qwen_response_marks_failures_for_audit():
    parsed = parse_qwen_response("not json")

    assert parsed.parse_status == "parse_failure"
    assert parsed.security_relevance is None
    assert parsed.quality is None
    assert parsed.should_keep is None
    assert parsed.reason == "parse_failure_requires_rescore"


def test_qa_prompt_has_compact_json_contract_without_cot_request():
    row = {
        "source_id": "stackexchange-infosec",
        "record_id": "stackexchange-infosec:1",
        "title": "How do salts work?",
        "tags": ["passwords", "hashing"],
        "content": "Question and answer content",
    }

    messages = build_messages(row, QwenTask.QA)
    prompt = "\n".join(message["content"] for message in messages)

    assert "Return exactly one compact JSON object" in prompt
    assert "chain-of-thought" in prompt
    assert "security_relevance" in prompt
    assert "passwords, hashing" in prompt


def test_arxiv_abstract_prompt_uses_metadata_not_full_content():
    row = {
        "source_id": "arxiv",
        "record_id": "arxiv:2401.00001",
        "content_hash": "abc",
        "title": "A Privacy-Preserving Protocol",
        "abstract": "We study private set intersection.",
        "categories": ["cs.CR"],
        "content": "FULL PAPER TEXT SHOULD NOT BE IN ABSTRACT PROMPT",
    }

    messages = build_messages(row, QwenTask.ARXIV_ABSTRACT)
    prompt = "\n".join(message["content"] for message in messages)

    assert "private set intersection" in prompt
    assert "FULL PAPER TEXT" not in prompt


def test_make_qwen_sidecar_row_preserves_key_and_model_metadata():
    parsed = parse_qwen_response(
        '{"security_relevance": 3, "quality": 3, "should_keep": true, "reason": "direct"}'
    )
    row = make_qwen_sidecar_row(
        {"source_id": "arxiv", "record_id": "arxiv:1", "content_hash": "h", "arxiv_id": "1"},
        parsed,
        task="arxiv-abstract",
        model="Qwen/Qwen3-4B",
        model_revision="0123456789abcdef",
        prompt_version="test-prompt",
        shard_id="7",
        raw_response="{}",
    )

    assert row["source_id"] == "arxiv"
    assert row["qwen_model"] == "Qwen/Qwen3-4B"
    assert row["qwen_model_revision"] == "0123456789abcdef"
    assert row["qwen_prompt_version"] == "test-prompt"
    assert row["qwen_shard_id"] == "7"
    assert row["arxiv_id"] == "1"


def test_model_revision_requires_full_commit_sha():
    _validate_model_revision("b968826d9c46dd6066d109eabc6255188de91218")
    with pytest.raises(ValueError, match="40-character"):
        _validate_model_revision("main")


def test_next_part_index_uses_maximum_existing_suffix(tmp_path):
    (tmp_path / "part-shard-3-000000.parquet").touch()
    (tmp_path / "part-shard-3-000004.parquet").touch()
    assert _next_part_index(tmp_path, "3") == 5
