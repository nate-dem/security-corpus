"""Prompt construction and response parsing for local Qwen filtering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from typing import Any, Mapping


class QwenTask(str, Enum):
    """Supported V3 Qwen filtering tasks."""

    QA = "qa"
    ARXIV_ABSTRACT = "arxiv_abstract"
    ARXIV_FULL = "arxiv_full"


QWEN_PROMPT_VERSIONS: dict[QwenTask, str] = {
    QwenTask.QA: "qwen-qa-v3-all-records-head-tail",
    QwenTask.ARXIV_ABSTRACT: "qwen-arxiv-abstract-v1",
    QwenTask.ARXIV_FULL: "qwen-arxiv-full-v1",
}

QWEN_INPUT_KINDS: dict[QwenTask, str] = {
    QwenTask.QA: "qa_thread",
    QwenTask.ARXIV_ABSTRACT: "arxiv_metadata_abstract",
    QwenTask.ARXIV_FULL: "arxiv_full_text_or_chunk",
}

QWEN_REQUIRED_COLUMNS: dict[QwenTask, tuple[str, ...]] = {
    QwenTask.QA: (
        "source_id",
        "record_id",
        "content_hash",
        "content",
    ),
    QwenTask.ARXIV_ABSTRACT: (
        "source_id",
        "record_id",
        "content_hash",
        "title",
        "abstract",
    ),
    QwenTask.ARXIV_FULL: (
        "source_id",
        "record_id",
        "content_hash",
        "content",
    ),
}

QWEN_OPTIONAL_COLUMNS: dict[QwenTask, tuple[str, ...]] = {
    QwenTask.QA: (
        "title",
        "score",
        "answer_count",
        "has_accepted_answer",
        "closed",
        "tags",
    ),
    QwenTask.ARXIV_ABSTRACT: (
        "authors",
        "categories",
        "primary_category",
        "arxiv_id",
        "doi",
        "journal_ref",
    ),
    QwenTask.ARXIV_FULL: (
        "title",
        "abstract",
        "categories",
        "primary_category",
        "arxiv_id",
        "source_format",
    ),
}

_SYSTEM_PROMPT = (
    "You label records for a cybersecurity continued-pretraining corpus. "
    "Return exactly one compact JSON object and no markdown. Be selective: "
    "most merely plausible, basic, thin, generic, or helpdesk-like records "
    "should be dropped. Do not include chain-of-thought or explanation outside JSON."
)

_OUTPUT_CONTRACT = (
    'Use this schema: {"security_relevance": 0, "quality": 0, '
    '"should_keep": false, "reason": "short explanation"}. '
    "security_relevance: 0 not security-relevant, 1 weak/general technical, "
    "2 security-adjacent or useful security context, 3 directly security-relevant. "
    "quality: 0 garbage/broken/spam, 1 low value or thin, 2 usable, "
    "3 high quality/substantive. Set should_keep=true only for records clearly "
    "worth spending training tokens on. When uncertain, prefer lower scores "
    "and should_keep=false."
)

_QA_KEEP_RULES = (
    "Keep only substantive cybersecurity, privacy, cryptography, reverse "
    "engineering, malware, vulnerability, exploit, detection, incident response, "
    "network security, authentication, authorization, cloud security, secure "
    "coding, or systems-security content. Drop ordinary programming or debugging "
    "unless security is central. Drop product support, shopping, account recovery, "
    "career advice, shallow recommendations, homework, opinion polls, jokes, social "
    "chatter, duplicate low-information answers, and generic IT administration. "
    "A thread from a security site can still be too thin or speculative to keep."
)


@dataclass(frozen=True)
class QwenParsedResponse:
    """Parsed compact JSON response plus audit status."""

    security_relevance: int | None
    quality: int | None
    should_keep: bool | None
    reason: str
    parse_status: str


def coerce_task(task: str | QwenTask) -> QwenTask:
    """Normalize CLI/API task names."""
    if isinstance(task, QwenTask):
        return task
    normalized = task.replace("-", "_")
    return QwenTask(normalized)


def build_messages(
    row: Mapping[str, Any],
    task: str | QwenTask,
    *,
    max_content_chars: int = 24_000,
) -> list[dict[str, str]]:
    """Build chat messages for one record."""
    qwen_task = coerce_task(task)
    if qwen_task is QwenTask.QA:
        user_prompt = _qa_user_prompt(row, max_content_chars=max_content_chars)
    elif qwen_task is QwenTask.ARXIV_ABSTRACT:
        user_prompt = _arxiv_abstract_user_prompt(row)
    elif qwen_task is QwenTask.ARXIV_FULL:
        user_prompt = _arxiv_full_user_prompt(row, max_content_chars=max_content_chars)
    else:  # pragma: no cover - exhaustive enum guard
        raise ValueError(f"Unsupported Qwen task: {task}")
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def render_prompt(
    row: Mapping[str, Any],
    task: str | QwenTask,
    *,
    tokenizer: Any | None = None,
    max_content_chars: int = 24_000,
) -> str:
    """Render a prompt, using the tokenizer chat template when available."""
    messages = build_messages(row, task, max_content_chars=max_content_chars)
    if tokenizer is not None:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}"
        for message in messages
    ) + "\n\nASSISTANT:\n"


def parse_qwen_response(
    response_text: str,
    *,
    parse_failure_should_keep: bool | None = None,
) -> QwenParsedResponse:
    """Parse Qwen's compact JSON response.

    Strict JSON is tried first. If that fails, the first balanced JSON object is
    extracted and parsed. Failures are explicit and have no keep/drop decision.
    """
    stripped = response_text.strip()
    payload: Any
    try:
        payload = json.loads(stripped)
        status = "ok"
    except json.JSONDecodeError:
        extracted = extract_first_json_object(stripped)
        if extracted is None:
            return _parse_failure(parse_failure_should_keep)
        try:
            payload = json.loads(extracted)
            status = "extracted_json"
        except json.JSONDecodeError:
            return _parse_failure(parse_failure_should_keep)

    parsed = _validate_payload(payload, status)
    if parsed is None:
        return _parse_failure(parse_failure_should_keep)
    return parsed


def extract_first_json_object(text: str) -> str | None:
    """Return the first balanced JSON object substring, respecting strings."""
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def make_qwen_sidecar_row(
    input_row: Mapping[str, Any],
    parsed: QwenParsedResponse,
    *,
    task: str | QwenTask,
    model: str,
    model_revision: str | None = None,
    prompt_version: str | None = None,
    scored_at: str | None = None,
    shard_id: str | None = None,
    raw_response: str | None = None,
) -> dict[str, Any]:
    """Build one standard sidecar row from a parsed Qwen response."""
    qwen_task = coerce_task(task)
    timestamp = scored_at or datetime.now(timezone.utc).isoformat()
    return {
        "source_id": input_row.get("source_id"),
        "record_id": input_row.get("record_id"),
        "content_hash": input_row.get("content_hash"),
        "qwen_security_relevance": parsed.security_relevance,
        "qwen_quality": parsed.quality,
        "qwen_should_keep": parsed.should_keep,
        "qwen_reason": parsed.reason,
        "qwen_parse_status": parsed.parse_status,
        "qwen_model": model,
        "qwen_model_revision": model_revision,
        "qwen_prompt_version": prompt_version or QWEN_PROMPT_VERSIONS[qwen_task],
        "qwen_scored_at": timestamp,
        "qwen_shard_id": shard_id,
        "qwen_task": qwen_task.value,
        "qwen_input_kind": QWEN_INPUT_KINDS[qwen_task],
        "qwen_raw_response": raw_response,
        "arxiv_id": input_row.get("arxiv_id"),
    }


def qwen_schema_extra_fields(task: str | QwenTask) -> dict[str, Any]:
    """Return task-specific extra sidecar fields."""
    qwen_task = coerce_task(task)
    if qwen_task in {QwenTask.ARXIV_ABSTRACT, QwenTask.ARXIV_FULL}:
        import pyarrow as pa

        return {"arxiv_id": pa.string()}
    return {}


def _qa_user_prompt(row: Mapping[str, Any], *, max_content_chars: int) -> str:
    tags = _join_list(row.get("tags"))
    parts = [
        "Evaluate this Q&A/social thread for a security-domain mid-training corpus.",
        "Judge whether it teaches useful security concepts, techniques, failure modes, artifacts, or reasoning.",
        _QA_KEEP_RULES,
        _OUTPUT_CONTRACT,
        "",
        "Record metadata:",
        f"source_id: {_text(row.get('source_id'))}",
        f"record_id: {_text(row.get('record_id'))}",
        f"title: {_text(row.get('title'))}",
        f"tags: {tags}",
        f"score: {_text(row.get('score'))}",
        f"answer_count: {_text(row.get('answer_count'))}",
        f"has_accepted_answer: {_text(row.get('has_accepted_answer'))}",
        f"closed: {_text(row.get('closed'))}",
        "",
        "Thread content:",
        _truncate(_text(row.get("content")), max_content_chars),
    ]
    return "\n".join(parts)


def _arxiv_abstract_user_prompt(row: Mapping[str, Any]) -> str:
    parts = [
        "Evaluate this arXiv paper metadata for a security-domain mid-training corpus.",
        "The main question is security/privacy/cryptography/safety relevance.",
        "Reject high-quality general CS/math/ML/systems papers with no security angle.",
        "Keep likely security-relevant or borderline adjacent work worth full-text review.",
        _OUTPUT_CONTRACT,
        "",
        "Paper metadata:",
        f"arxiv_id: {_text(row.get('arxiv_id'))}",
        f"title: {_text(row.get('title'))}",
        f"authors: {_join_list(row.get('authors'))}",
        f"primary_category: {_text(row.get('primary_category'))}",
        f"categories: {_join_list(row.get('categories'))}",
        f"doi: {_text(row.get('doi'))}",
        f"journal_ref: {_text(row.get('journal_ref'))}",
        "",
        "Abstract:",
        _text(row.get("abstract")),
    ]
    return "\n".join(parts)


def _arxiv_full_user_prompt(row: Mapping[str, Any], *, max_content_chars: int) -> str:
    parts = [
        "Evaluate this arXiv full paper or chunk for a security-domain mid-training corpus.",
        "Judge whether the text contains useful cybersecurity/security/privacy/cryptography/safety content.",
        "Reject merely general CS/math/ML/systems content with no security angle.",
        _OUTPUT_CONTRACT,
        "",
        "Paper metadata:",
        f"arxiv_id: {_text(row.get('arxiv_id'))}",
        f"title: {_text(row.get('title'))}",
        f"primary_category: {_text(row.get('primary_category'))}",
        f"categories: {_join_list(row.get('categories'))}",
        f"source_format: {_text(row.get('source_format'))}",
        "",
        "Abstract:",
        _truncate(_text(row.get("abstract")), 3_000),
        "",
        "Paper or chunk text:",
        _truncate(_text(row.get("content")), max_content_chars),
    ]
    return "\n".join(parts)


def _validate_payload(payload: Any, status: str) -> QwenParsedResponse | None:
    if not isinstance(payload, dict):
        return None
    security_relevance = _coerce_score(payload.get("security_relevance"))
    quality = _coerce_score(payload.get("quality"))
    should_keep = _coerce_bool(payload.get("should_keep"))
    reason = _text(payload.get("reason")).strip()
    if security_relevance is None or quality is None or should_keep is None:
        return None
    if not reason:
        reason = "No reason provided."
    return QwenParsedResponse(
        security_relevance=security_relevance,
        quality=quality,
        should_keep=should_keep,
        reason=reason[:280],
        parse_status=status,
    )


def _parse_failure(parse_failure_should_keep: bool | None) -> QwenParsedResponse:
    if parse_failure_should_keep is None:
        reason = "parse_failure_requires_rescore"
    elif parse_failure_should_keep:
        reason = "parse_failure_keep_for_review"
    else:
        reason = "parse_failure_no_keep_fallback"
    return QwenParsedResponse(
        security_relevance=None,
        quality=None,
        should_keep=parse_failure_should_keep,
        reason=reason,
        parse_status="parse_failure",
    )


def _coerce_score(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    if 0 <= score <= 3:
        return score
    return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return _join_list(value)
    return str(value)


def _join_list(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return ", ".join(str(item) for item in value if item is not None)
    except TypeError:
        return str(value)


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head_chars = max_chars * 2 // 3
    tail_chars = max_chars - head_chars
    head = text[:head_chars].rsplit("\n", 1)[0].rstrip() or text[:head_chars].rstrip()
    tail = text[-tail_chars:].split("\n", 1)[-1].lstrip() or text[-tail_chars:].lstrip()
    return f"{head}\n[TRUNCATED: MIDDLE OMITTED]\n{tail}"
