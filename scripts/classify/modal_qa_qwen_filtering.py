#!/usr/bin/env python3
"""Modal Qwen filter for one QA source partition.

This script is intended for source-by-source QA filtering when local/Slurm GPU
compute is unavailable. It runs Qwen via vLLM on Modal and reads only records
that are marked high quality by the QA binary classifier candidate sidecar.

One-time setup from your local machine:

    modal setup
    modal volume create security-corpus-data

Upload the candidate sidecar and the QA source partition(s) you want to run:

    modal volume put security-corpus-data \
      data/filtering/v3/qa_qwen_candidates.parquet \
      /filtering/v3/qa_qwen_candidates.parquet

    modal volume put security-corpus-data \
      data/training-clean-v2/normalized/source_id=stackexchange-infosec \
      /training-clean-v2/normalized/source_id=stackexchange-infosec

Dry-run one source without loading Qwen:

    modal run scripts/classify/modal_qa_qwen_filtering.py \
      --source-id stackexchange-infosec \
      --dry-run \
      --max-records 3

Real run, one source and one shard:

    modal run scripts/classify/modal_qa_qwen_filtering.py \
      --source-id stackexchange-infosec \
      --model Qwen/Qwen3-8B \
      --shard-id 0 \
      --num-shards 1

By default the Modal function requests GPU fallbacks `["H100", "A100-80GB"]`.
Edit `GPU_REQUEST` below if you want a different GPU policy.

The default vLLM context is intentionally 8192 tokens. That is enough for the
current QA filtering prompt after content truncation and avoids oversized KV
cache allocation during engine startup. If a source needs longer records, raise
`--max-model-len` and lower `--batch-size` / `--max-num-seqs` together.

For large sources, launch multiple independent shard jobs:

    modal run scripts/classify/modal_qa_qwen_filtering.py \
      --source-id stackoverflow \
      --model Qwen/Qwen3-8B \
      --shard-id 0 \
      --num-shards 100

Then repeat with shard IDs 0..99. Outputs are append/resume JSONL files under:

    /filtering/v3/qwen_qa_modal/{source_id}/shard-{shard_id}-of-{num_shards}.jsonl

inside the Modal Volume.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import modal


APP_NAME = "security-corpus-qa-qwen-filter"
VOLUME_NAME = "security-corpus-data"
MOUNT_PATH = Path("/data")
DEFAULT_INPUT_ROOT = MOUNT_PATH / "training-clean-v2" / "normalized"
DEFAULT_CANDIDATE_SIDECAR = MOUNT_PATH / "filtering" / "v3" / "qa_qwen_candidates.parquet"
DEFAULT_OUTPUT_ROOT = MOUNT_PATH / "filtering" / "v3" / "qwen_qa_modal"
DEFAULT_MODEL = "Qwen/Qwen3-8B"
PROMPT_VERSION = "qwen-qa-v2-strict-modal"
DEFAULT_KEEP_POLICY = "strict"
TASK = "qa"
INPUT_KIND = "qa_thread"
GPU_REQUEST = ["H100", "A100-80GB"]


image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.13")
    .entrypoint([])
    .uv_pip_install(
        "pyarrow",
        "transformers",
        "huggingface-hub==0.36.0",
        "vllm==0.13.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name("vllm-cache", create_if_missing=True)
app = modal.App(APP_NAME, image=image)


SYSTEM_PROMPT = (
    "You label Q&A and social-thread records for a cybersecurity "
    "continued-pretraining corpus. Return exactly one compact JSON object and "
    "no markdown. Be selective: most merely plausible, basic, thin, generic, "
    "or helpdesk-like threads should be dropped. Do not include chain-of-thought "
    "or explanation outside JSON."
)

OUTPUT_CONTRACT = (
    'Use this schema: {"security_relevance": 0, "quality": 0, '
    '"should_keep": false, "reason": "short explanation"}. '
    "security_relevance: 0 not security-relevant, 1 weak/general technical, "
    "2 security-adjacent or useful security context, 3 directly security-relevant. "
    "quality: 0 garbage/broken/spam, 1 low value or thin, 2 usable, "
    "3 high quality/substantive. "
    "Set should_keep=true only for records that are clearly worth spending "
    "training tokens on after a QA-quality prefilter. When uncertain, prefer "
    "lower scores and should_keep=false."
)

STRICT_KEEP_RULES = (
    "Strict keep rules for QA filtering: Keep only if the thread contains "
    "substantive cybersecurity, privacy, cryptography, reverse engineering, "
    "malware, vulnerability, exploit, detection, incident response, network "
    "security, authentication, authorization, cloud security, secure coding, "
    "or systems-security content. Drop ordinary programming/debugging threads "
    "unless the security angle is central. Drop product support, antivirus "
    "shopping, account recovery, career advice, tool recommendations without "
    "technical depth, homework, opinion polls, jokes, social chatter, duplicate "
    "low-information answers, and generic IT/admin questions. A thread can be "
    "from a security site and still be should_keep=false if it is thin, basic, "
    "unresolved, speculative, or not useful for continued pretraining."
)


@app.function(
    gpu=GPU_REQUEST,
    timeout=60 * 60 * 24,
    volumes={
        MOUNT_PATH.as_posix(): volume,
        "/root/.cache/huggingface": hf_cache_volume,
        "/root/.cache/vllm": vllm_cache_volume,
    },
)
def score_qa_source(
    *,
    source_id: str,
    input_root: str = DEFAULT_INPUT_ROOT.as_posix(),
    candidate_sidecar: str = DEFAULT_CANDIDATE_SIDECAR.as_posix(),
    output_root: str = DEFAULT_OUTPUT_ROOT.as_posix(),
    model: str = DEFAULT_MODEL,
    batch_size: int = 16,
    read_batch_size: int = 2048,
    max_records: int | None = None,
    max_content_chars: int = 24_000,
    max_tokens: int = 96,
    max_model_len: int = 8_192,
    max_num_seqs: int = 16,
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.82,
    shard_id: int = 0,
    num_shards: int = 1,
    keep_policy: str = DEFAULT_KEEP_POLICY,
    parse_failure_should_keep: bool = False,
    dry_run: bool = False,
    overwrite: bool = False,
    trust_remote_code: bool = True,
) -> dict[str, Any]:
    """Score one QA source/shard on Modal."""
    _validate_args(
        source_id=source_id,
        shard_id=shard_id,
        num_shards=num_shards,
        batch_size=batch_size,
        keep_policy=keep_policy,
    )
    volume.reload()

    output_path = _output_path(Path(output_root), source_id, shard_id, num_shards)
    if overwrite and output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys = _load_done_keys(output_path)
    candidate_keys = _load_candidate_keys(Path(candidate_sidecar), source_id)
    rows = _iter_source_rows(
        input_root=Path(input_root),
        source_id=source_id,
        candidate_keys=candidate_keys,
        done_keys=done_keys,
        shard_id=shard_id,
        num_shards=num_shards,
        read_batch_size=read_batch_size,
        max_records=max_records,
    )

    if dry_run:
        preview = []
        for row in rows:
            preview.append(
                {
                    "source_id": row["source_id"],
                    "record_id": row["record_id"],
                    "content_hash": row["content_hash"],
                    "prompt": _render_prompt_without_tokenizer(
                        row,
                        max_content_chars=max_content_chars,
                    ),
                }
            )
            if len(preview) >= (max_records or 3):
                break
        return {
            "dry_run": True,
            "source_id": source_id,
            "shard_id": shard_id,
            "num_shards": num_shards,
            "candidate_keys": len(candidate_keys),
            "already_done": len(done_keys),
            "preview": preview,
        }

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
    llm = LLM(
        model=model,
        dtype=dtype,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        enable_prefix_caching=True,
        trust_remote_code=trust_remote_code,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    start = time.monotonic()
    scored = 0
    kept = 0
    parse_failures = 0
    batch_rows: list[dict[str, Any]] = []
    batch_prompts: list[str] = []

    with output_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            batch_rows.append(row)
            batch_prompts.append(
                _render_prompt(
                    row,
                    tokenizer=tokenizer,
                    max_content_chars=max_content_chars,
                    max_model_len=max_model_len,
                    max_tokens=max_tokens,
                )
            )
            if len(batch_rows) >= batch_size:
                n, k, p = _score_batch(
                    llm,
                    sampling_params,
                    batch_rows,
                    batch_prompts,
                    handle,
                    model=model,
                    shard_id=shard_id,
                    keep_policy=keep_policy,
                    parse_failure_should_keep=parse_failure_should_keep,
                )
                scored += n
                kept += k
                parse_failures += p
                _print_progress(source_id, shard_id, scored, kept, parse_failures, start)
                batch_rows = []
                batch_prompts = []

        if batch_rows:
            n, k, p = _score_batch(
                llm,
                sampling_params,
                batch_rows,
                batch_prompts,
                handle,
                model=model,
                shard_id=shard_id,
                keep_policy=keep_policy,
                parse_failure_should_keep=parse_failure_should_keep,
            )
            scored += n
            kept += k
            parse_failures += p
            _print_progress(source_id, shard_id, scored, kept, parse_failures, start)

    volume.commit()
    return {
        "dry_run": False,
        "source_id": source_id,
        "shard_id": shard_id,
        "num_shards": num_shards,
        "candidate_keys": len(candidate_keys),
        "already_done_at_start": len(done_keys),
        "scored": scored,
        "kept": kept,
        "rejected": scored - kept,
        "parse_failures": parse_failures,
        "output_path": output_path.as_posix(),
        "elapsed_seconds": round(time.monotonic() - start, 2),
    }


@app.local_entrypoint()
def main(
    source_id: str,
    input_root: str = DEFAULT_INPUT_ROOT.as_posix(),
    candidate_sidecar: str = DEFAULT_CANDIDATE_SIDECAR.as_posix(),
    output_root: str = DEFAULT_OUTPUT_ROOT.as_posix(),
    model: str = DEFAULT_MODEL,
    batch_size: int = 16,
    read_batch_size: int = 2048,
    max_records: int | None = None,
    max_content_chars: int = 24_000,
    max_tokens: int = 96,
    max_model_len: int = 8_192,
    max_num_seqs: int = 16,
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.82,
    shard_id: int = 0,
    num_shards: int = 1,
    keep_policy: str = DEFAULT_KEEP_POLICY,
    parse_failure_should_keep: bool = False,
    dry_run: bool = False,
    overwrite: bool = False,
) -> None:
    """Launch one source/shard on Modal."""
    result = score_qa_source.remote(
        source_id=source_id,
        input_root=input_root,
        candidate_sidecar=candidate_sidecar,
        output_root=output_root,
        model=model,
        batch_size=batch_size,
        read_batch_size=read_batch_size,
        max_records=max_records,
        max_content_chars=max_content_chars,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        shard_id=shard_id,
        num_shards=num_shards,
        keep_policy=keep_policy,
        parse_failure_should_keep=parse_failure_should_keep,
        dry_run=dry_run,
        overwrite=overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def _validate_args(
    *,
    source_id: str,
    shard_id: int,
    num_shards: int,
    batch_size: int,
    keep_policy: str,
) -> None:
    if not source_id:
        raise ValueError("source_id is required")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must be in [0, num_shards)")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if keep_policy not in {"strict", "model"}:
        raise ValueError("keep_policy must be one of: strict, model")


def _output_path(output_root: Path, source_id: str, shard_id: int, num_shards: int) -> Path:
    safe_source = source_id.replace("/", "_")
    return output_root / safe_source / f"shard-{shard_id:05d}-of-{num_shards:05d}.jsonl"


def _load_candidate_keys(candidate_sidecar: Path, source_id: str) -> set[tuple[str, str, str]]:
    import pyarrow.parquet as pq

    if not candidate_sidecar.exists():
        raise FileNotFoundError(f"Candidate sidecar not found: {candidate_sidecar}")
    table = pq.read_table(
        candidate_sidecar,
        columns=["source_id", "record_id", "content_hash", "qa_candidate_for_qwen"],
    )
    keys = set()
    for row in table.to_pylist():
        if row.get("source_id") != source_id:
            continue
        if row.get("qa_candidate_for_qwen") is not True:
            continue
        keys.add(_key(row))
    if not keys:
        raise ValueError(f"No qa_candidate_for_qwen=true keys found for {source_id}")
    return keys


def _load_done_keys(output_path: Path) -> set[tuple[str, str, str]]:
    if not output_path.exists():
        return set()
    keys = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add(_key(row))
    return keys


def _iter_source_rows(
    *,
    input_root: Path,
    source_id: str,
    candidate_keys: set[tuple[str, str, str]],
    done_keys: set[tuple[str, str, str]],
    shard_id: int,
    num_shards: int,
    read_batch_size: int,
    max_records: int | None,
) -> Iterable[dict[str, Any]]:
    import pyarrow.dataset as ds

    source_path = input_root / f"source_id={source_id}"
    if not source_path.exists():
        raise FileNotFoundError(f"Source partition not found: {source_path}")

    dataset = ds.dataset(str(source_path), format="parquet")
    required = ["source_id", "record_id", "content_hash", "content"]
    optional = ["title", "score", "answer_count", "has_accepted_answer", "closed", "tags"]
    available = set(dataset.schema.names)
    missing = [column for column in required if column not in available]
    if missing:
        raise ValueError(f"{source_path} is missing required QA columns: {missing}")
    columns = required + [column for column in optional if column in available]

    emitted = 0
    scanner = dataset.scanner(columns=columns, batch_size=max(1, read_batch_size))
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            key = _key(row)
            if key not in candidate_keys:
                continue
            if key in done_keys:
                continue
            if _shard_for_key(key, num_shards) != shard_id:
                continue
            _validate_row(row)
            for column in optional:
                row.setdefault(column, None)
            yield row
            emitted += 1
            if max_records is not None and emitted >= max_records:
                return


def _validate_row(row: Mapping[str, Any]) -> None:
    missing = [
        column
        for column in ("source_id", "record_id", "content_hash", "content")
        if _is_blank(row.get(column))
    ]
    if missing:
        raise ValueError(f"Record {row.get('record_id')!r} has missing QA fields: {missing}")


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _shard_for_key(key: tuple[str, str, str], num_shards: int) -> int:
    digest = sha1("\x1f".join(key).encode("utf-8")).hexdigest()
    return int(digest, 16) % num_shards


def _score_batch(
    llm: Any,
    sampling_params: Any,
    rows: list[dict[str, Any]],
    prompts: list[str],
    handle: Any,
    *,
    model: str,
    shard_id: int,
    keep_policy: str,
    parse_failure_should_keep: bool,
) -> tuple[int, int, int]:
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    scored_at = datetime.now(timezone.utc).isoformat()
    kept = 0
    parse_failures = 0
    for row, output in zip(rows, outputs):
        response_text = output.outputs[0].text if output.outputs else ""
        parsed = _parse_qwen_response(
            response_text,
            parse_failure_should_keep=parse_failure_should_keep,
        )
        policy = _apply_keep_policy(parsed, keep_policy=keep_policy)
        if policy["should_keep"]:
            kept += 1
        if parsed["parse_status"] == "parse_failure":
            parse_failures += 1
        sidecar_row = {
            "source_id": row.get("source_id"),
            "record_id": row.get("record_id"),
            "content_hash": row.get("content_hash"),
            "qwen_security_relevance": parsed["security_relevance"],
            "qwen_quality": parsed["quality"],
            "qwen_model_should_keep": parsed["should_keep"],
            "qwen_should_keep": policy["should_keep"],
            "qwen_reason": parsed["reason"],
            "qwen_parse_status": parsed["parse_status"],
            "qwen_keep_policy": keep_policy,
            "qwen_keep_policy_passed": policy["passed"],
            "qwen_keep_policy_reason": policy["reason"],
            "qwen_model": model,
            "qwen_prompt_version": PROMPT_VERSION,
            "qwen_scored_at": scored_at,
            "qwen_shard_id": str(shard_id),
            "qwen_task": TASK,
            "qwen_input_kind": INPUT_KIND,
            "qwen_raw_response": response_text,
        }
        handle.write(json.dumps(sidecar_row, ensure_ascii=True))
        handle.write("\n")
    handle.flush()
    return len(rows), kept, parse_failures


def _render_prompt(
    row: Mapping[str, Any],
    *,
    tokenizer: Any,
    max_content_chars: int,
    max_model_len: int,
    max_tokens: int,
) -> str:
    target_prompt_tokens = max_model_len - max_tokens - 64
    if target_prompt_tokens <= 0:
        raise ValueError(
            "max_model_len must leave room for max_tokens plus prompt safety margin"
        )

    prompt = _apply_chat_template(
        tokenizer,
        _messages(row, max_content_chars=max_content_chars),
    )
    if _prompt_token_length(tokenizer, prompt) <= target_prompt_tokens:
        return prompt

    content = _text(row.get("content"))
    low = 0
    high = min(max_content_chars, len(content))
    best_prompt: str | None = None

    while low <= high:
        mid = (low + high) // 2
        candidate = _apply_chat_template(
            tokenizer,
            _messages(row, max_content_chars=mid),
        )
        if _prompt_token_length(tokenizer, candidate) <= target_prompt_tokens:
            best_prompt = candidate
            low = mid + 1
        else:
            high = mid - 1

    if best_prompt is not None:
        return best_prompt

    metadata_only = _apply_chat_template(
        tokenizer,
        _messages(row, max_content_chars=0),
    )
    if _prompt_token_length(tokenizer, metadata_only) <= target_prompt_tokens:
        return metadata_only
    raise ValueError(
        f"Prompt metadata exceeds token budget for record {row.get('record_id')!r}"
    )


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
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


def _prompt_token_length(tokenizer: Any, prompt: str) -> int:
    return len(tokenizer(prompt, add_special_tokens=False).input_ids)


def _render_prompt_without_tokenizer(row: Mapping[str, Any], *, max_content_chars: int) -> str:
    messages = _messages(row, max_content_chars=max_content_chars)
    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}"
        for message in messages
    ) + "\n\nASSISTANT:\n"


def _messages(row: Mapping[str, Any], *, max_content_chars: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _qa_user_prompt(row, max_content_chars=max_content_chars)},
    ]


def _qa_user_prompt(row: Mapping[str, Any], *, max_content_chars: int) -> str:
    return "\n".join(
        [
            "Evaluate this Q&A/social thread for a security-domain mid-training corpus.",
            "Only keep high-quality, security-relevant records. This is a final semantic filter after a QA quality prefilter, so be much stricter than a broad relevance search.",
            "Judge whether the thread teaches useful security-domain concepts, techniques, failure modes, artifacts, or reasoning.",
            "Do not keep a record just because it appears on a security forum or mentions a security keyword.",
            STRICT_KEEP_RULES,
            "Use should_keep=true only when the record has security_relevance 3 and quality at least 2, or security_relevance 2 and quality 3.",
            "Use should_keep=false for weak/general relevance, thin answers, unresolved questions, basic troubleshooting, or content you would not want repeated in mid-training data.",
            OUTPUT_CONTRACT,
            "",
            "Record metadata:",
            f"source_id: {_text(row.get('source_id'))}",
            f"record_id: {_text(row.get('record_id'))}",
            f"title: {_text(row.get('title'))}",
            f"tags: {_join_list(row.get('tags'))}",
            f"score: {_text(row.get('score'))}",
            f"answer_count: {_text(row.get('answer_count'))}",
            f"has_accepted_answer: {_text(row.get('has_accepted_answer'))}",
            f"closed: {_text(row.get('closed'))}",
            "",
            "Thread content:",
            _truncate(_text(row.get("content")), max_content_chars),
        ]
    )


def _parse_qwen_response(response_text: str, *, parse_failure_should_keep: bool) -> dict[str, Any]:
    stripped = response_text.strip()
    try:
        payload = json.loads(stripped)
        status = "ok"
    except json.JSONDecodeError:
        extracted = _extract_first_json_object(stripped)
        if extracted is None:
            return _parse_failure(parse_failure_should_keep)
        try:
            payload = json.loads(extracted)
            status = "extracted_json"
        except json.JSONDecodeError:
            return _parse_failure(parse_failure_should_keep)
    if not isinstance(payload, dict):
        return _parse_failure(parse_failure_should_keep)
    security_relevance = _coerce_score(payload.get("security_relevance"))
    quality = _coerce_score(payload.get("quality"))
    should_keep = _coerce_bool(payload.get("should_keep"))
    reason = _text(payload.get("reason")).strip()[:280]
    if security_relevance is None or quality is None or should_keep is None:
        return _parse_failure(parse_failure_should_keep)
    return {
        "security_relevance": security_relevance,
        "quality": quality,
        "should_keep": should_keep,
        "reason": reason or "No reason provided.",
        "parse_status": status,
    }


def _apply_keep_policy(parsed: Mapping[str, Any], *, keep_policy: str) -> dict[str, Any]:
    model_should_keep = bool(parsed.get("should_keep"))
    if keep_policy == "model":
        return {
            "should_keep": model_should_keep,
            "passed": model_should_keep,
            "reason": "model_decision",
        }

    security_relevance = parsed.get("security_relevance")
    quality = parsed.get("quality")
    if parsed.get("parse_status") == "parse_failure":
        return {
            "should_keep": False,
            "passed": False,
            "reason": "parse_failure_not_accepted_by_strict_policy",
        }
    if not model_should_keep:
        return {
            "should_keep": False,
            "passed": False,
            "reason": "model_should_keep_false",
        }
    if security_relevance is None or quality is None:
        return {
            "should_keep": False,
            "passed": False,
            "reason": "missing_security_or_quality_score",
        }
    if security_relevance >= 3 and quality >= 2:
        return {
            "should_keep": True,
            "passed": True,
            "reason": "direct_security_and_usable_quality",
        }
    if security_relevance >= 2 and quality >= 3:
        return {
            "should_keep": True,
            "passed": True,
            "reason": "security_adjacent_and_high_quality",
        }
    return {
        "should_keep": False,
        "passed": False,
        "reason": "strict_policy_requires_security_3_quality_2_or_security_2_quality_3",
    }


def _extract_first_json_object(text: str) -> str | None:
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


def _parse_failure(parse_failure_should_keep: bool) -> dict[str, Any]:
    return {
        "security_relevance": None,
        "quality": None,
        "should_keep": parse_failure_should_keep,
        "reason": "parse_failure_keep_for_review"
        if parse_failure_should_keep
        else "parse_failure_no_keep_fallback",
        "parse_status": "parse_failure",
    }


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


def _key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("source_id") or ""),
        str(row.get("record_id") or ""),
        str(row.get("content_hash") or ""),
    )


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
    if max_chars <= 0:
        return "[TRUNCATED]" if text else ""
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars].rsplit("\n", 1)[0].rstrip()
    if not trimmed:
        trimmed = text[:max_chars].rstrip()
    return f"{trimmed}\n[TRUNCATED]"


def _print_progress(
    source_id: str,
    shard_id: int,
    scored: int,
    kept: int,
    parse_failures: int,
    start: float,
) -> None:
    elapsed = max(time.monotonic() - start, 1e-6)
    print(
        f"{source_id} shard={shard_id} scored={scored:,} "
        f"kept={kept:,} rejected={scored - kept:,} "
        f"parse_failures={parse_failures:,} "
        f"records_per_sec={scored / elapsed:.2f}",
        flush=True,
    )
