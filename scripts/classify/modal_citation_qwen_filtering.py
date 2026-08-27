#!/usr/bin/env python3
"""Modal Qwen title+abstract filter for citation-expanded arXiv papers.

This runner is for the paper-level citation metadata Parquet uploaded to a
Modal Volume. It intentionally uses only title, abstract, categories, and
metadata; it does not load full paper content.

One-time setup from your local machine:

    modal setup
    modal volume create security-corpus-data

Upload the slim citation metadata Parquet:

    modal volume put security-corpus-data \
      data/arxiv/normalized/source_id=arxiv/citation_metadata_for_qwen.parquet \
      /arxiv/citation_metadata_for_qwen.parquet

Dry-run without loading Qwen:

    modal run scripts/classify/modal_citation_qwen_filtering.py \
      --dry-run \
      --max-records 3

Real run, one shard:

    modal run -d scripts/classify/modal_citation_qwen_filtering.py \
      --model Qwen/Qwen3-8B \
      --shard-id 0 \
      --num-shards 20

Outputs are append/resume JSONL files under:

    /filtering/v3/qwen_citation_abstract_modal/shard-{shard_id}-of-{num_shards}.jsonl

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


APP_NAME = "security-corpus-citation-qwen-filter"
VOLUME_NAME = "security-corpus-data"
MOUNT_PATH = Path("/data")
DEFAULT_INPUT = MOUNT_PATH / "arxiv" / "citation_metadata_for_qwen.parquet"
DEFAULT_OUTPUT_ROOT = MOUNT_PATH / "filtering" / "v3" / "qwen_citation_abstract_modal"
DEFAULT_MODEL = "Qwen/Qwen3-8B"
PROMPT_VERSION = "qwen-arxiv-citation-abstract-v2-modal"
DEFAULT_KEEP_POLICY = "strict"
TASK = "arxiv_citation_abstract"
INPUT_KIND = "arxiv_metadata_abstract"
GPU_REQUEST = ["H100", "A100-80GB"]

REQUIRED_COLUMNS = ("record_id", "content_hash", "title", "abstract")
OPTIONAL_COLUMNS = (
    "source_id",
    "source_record_id",
    "arxiv_id",
    "authors",
    "categories",
    "primary_category",
    "doi",
    "journal_ref",
    "source_url",
    "license",
)


image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.13")
    .entrypoint([])
    .uv_pip_install(
        "pyarrow",
        "tzdata",
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
    "You filter academic paper metadata for a cybersecurity continued-pretraining "
    "corpus. Judge only from the provided title, abstract, categories, and "
    "metadata. Be strict about security relevance: reject high-quality general "
    "CS, ML, math, systems, software engineering, hardware, or networking papers "
    "unless the abstract clearly connects to security, privacy, cryptography, "
    "adversaries, threats, vulnerabilities, exploitation, malware, abuse, "
    "defense, robustness against attacks, safety/security misuse, or secure "
    "design. Count theoretical cryptography and security-theory work as relevant "
    "when the abstract is about protocol security, key distribution, secrecy, "
    "wiretap channels, privacy amplification, cryptanalysis, randomness "
    "extractors against adversaries, secure computation, formal verification "
    "of security properties, or adversarial/threat models. Return exactly one "
    "compact JSON object and no markdown. Do not "
    "include chain-of-thought or explanation outside JSON."
)

OUTPUT_CONTRACT = (
    'Use this schema: {"security_relevance": 0, "quality": 0, '
    '"should_keep": false, "reason": "short explanation"}. '
    "security_relevance: 0 not security-relevant, "
    "1 weak/speculative security relevance, "
    "2 security-adjacent or useful security context, "
    "3 directly security/privacy/cryptography relevant. "
    "quality: 0 garbage/broken, 1 low value or too thin, "
    "2 usable, 3 high quality/substantive. "
    "Set should_keep=true only for papers worth keeping or sending to full-text "
    "review for a security-domain mid-training corpus. Reject low-quality papers "
    "and high-quality papers that are merely general technical work."
)

STRICT_KEEP_RULES = (
    "Strict keep rules: Keep only if the paper is directly security-relevant "
    "or meaningfully security-adjacent. Directly relevant topics include "
    "cybersecurity, privacy, cryptography, secure systems, vulnerability "
    "discovery, exploitation, malware, phishing, authentication, authorization, "
    "network security, cloud security, software supply-chain security, detection, "
    "incident response, adversarial robustness, misuse/safety with a security "
    "angle, or formal methods applied to security. Security-adjacent topics that "
    "should usually receive security_relevance=2 include quantum key distribution, "
    "wiretap/secrecy-capacity communication, privacy amplification, extractors "
    "secure against adversaries, cryptographic protocol analysis, cryptanalysis, "
    "secure multiparty computation, and software/model checking when the abstract "
    "connects it to safety, correctness, vulnerabilities, or security properties. "
    "Reject papers where security is absent, only a vague future application, "
    "or only a generic word in a title/category."
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
def score_citation_abstracts(
    *,
    input_path: str = DEFAULT_INPUT.as_posix(),
    output_root: str = DEFAULT_OUTPUT_ROOT.as_posix(),
    model: str = DEFAULT_MODEL,
    batch_size: int = 32,
    read_batch_size: int = 4096,
    max_records: int | None = None,
    max_abstract_chars: int = 8_000,
    max_tokens: int = 96,
    max_model_len: int = 4_096,
    max_num_seqs: int = 32,
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
    """Score one citation-abstract shard on Modal."""
    _validate_args(
        shard_id=shard_id,
        num_shards=num_shards,
        batch_size=batch_size,
        keep_policy=keep_policy,
    )
    volume.reload()

    input_file = Path(input_path)
    output_path = _output_path(Path(output_root), shard_id, num_shards)
    if overwrite and output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys = _load_done_keys(output_path)
    candidate_count = _count_candidate_rows(input_file, read_batch_size=read_batch_size)
    rows = _iter_rows(
        input_file=input_file,
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
                    "source_id": row.get("source_id") or "arxiv",
                    "record_id": row["record_id"],
                    "content_hash": row["content_hash"],
                    "arxiv_id": row.get("arxiv_id"),
                    "prompt": _render_prompt_without_tokenizer(
                        row,
                        max_abstract_chars=max_abstract_chars,
                    ),
                }
            )
            if len(preview) >= (max_records or 3):
                break
        return {
            "dry_run": True,
            "shard_id": shard_id,
            "num_shards": num_shards,
            "candidate_keys": candidate_count,
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
                    max_abstract_chars=max_abstract_chars,
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
                _print_progress(shard_id, scored, kept, parse_failures, start)
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
            _print_progress(shard_id, scored, kept, parse_failures, start)

    volume.commit()
    return {
        "dry_run": False,
        "shard_id": shard_id,
        "num_shards": num_shards,
        "candidate_keys": candidate_count,
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
    input_path: str = DEFAULT_INPUT.as_posix(),
    output_root: str = DEFAULT_OUTPUT_ROOT.as_posix(),
    model: str = DEFAULT_MODEL,
    batch_size: int = 32,
    read_batch_size: int = 4096,
    max_records: int | None = None,
    max_abstract_chars: int = 8_000,
    max_tokens: int = 96,
    max_model_len: int = 4_096,
    max_num_seqs: int = 32,
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
    """Launch one citation abstract shard on Modal."""
    result = score_citation_abstracts.remote(
        input_path=input_path,
        output_root=output_root,
        model=model,
        batch_size=batch_size,
        read_batch_size=read_batch_size,
        max_records=max_records,
        max_abstract_chars=max_abstract_chars,
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
    shard_id: int,
    num_shards: int,
    batch_size: int,
    keep_policy: str,
) -> None:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must be in [0, num_shards)")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if keep_policy not in {"strict", "model"}:
        raise ValueError("keep_policy must be one of: strict, model")


def _output_path(output_root: Path, shard_id: int, num_shards: int) -> Path:
    return output_root / f"shard-{shard_id:05d}-of-{num_shards:05d}.jsonl"


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


def _count_candidate_rows(input_file: Path, *, read_batch_size: int) -> int:
    count = 0
    for _ in _iter_rows(
        input_file=input_file,
        done_keys=set(),
        shard_id=0,
        num_shards=1,
        read_batch_size=read_batch_size,
        max_records=None,
    ):
        count += 1
    return count


def _iter_rows(
    *,
    input_file: Path,
    done_keys: set[tuple[str, str, str]],
    shard_id: int,
    num_shards: int,
    read_batch_size: int,
    max_records: int | None,
) -> Iterable[dict[str, Any]]:
    import pyarrow.dataset as ds

    if not input_file.exists():
        raise FileNotFoundError(f"Input Parquet not found: {input_file}")
    dataset = ds.dataset(str(input_file), format="parquet")
    columns = _available_columns(dataset)

    emitted = 0
    scanner = dataset.scanner(columns=columns, batch_size=max(1, read_batch_size))
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            row.setdefault("source_id", "arxiv")
            if row.get("source_id") and str(row.get("source_id")) != "arxiv":
                continue
            if any(_is_blank(row.get(column)) for column in REQUIRED_COLUMNS):
                continue
            key = _key(row)
            if key in done_keys:
                continue
            if _shard_for_key(key, num_shards) != shard_id:
                continue
            for column in OPTIONAL_COLUMNS:
                row.setdefault(column, None)
            yield row
            emitted += 1
            if max_records is not None and emitted >= max_records:
                return


def _available_columns(dataset: Any) -> list[str]:
    available = set(dataset.schema.names)
    missing = [column for column in REQUIRED_COLUMNS if column not in available]
    if missing:
        raise ValueError(
            "Input is missing required citation abstract columns: "
            + ", ".join(missing)
        )
    return [
        column
        for column in REQUIRED_COLUMNS + OPTIONAL_COLUMNS
        if column in available
    ]


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
            "source_id": row.get("source_id") or "arxiv",
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
            "arxiv_id": row.get("arxiv_id"),
            "title": row.get("title"),
            "primary_category": row.get("primary_category"),
            "categories": row.get("categories"),
            "abstract_preview": _preview(row.get("abstract"), 700),
        }
        handle.write(json.dumps(sidecar_row, ensure_ascii=True, default=str))
        handle.write("\n")
    handle.flush()
    return len(rows), kept, parse_failures


def _render_prompt(
    row: Mapping[str, Any],
    *,
    tokenizer: Any,
    max_abstract_chars: int,
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
        _messages(row, max_abstract_chars=max_abstract_chars),
    )
    if _prompt_token_length(tokenizer, prompt) <= target_prompt_tokens:
        return prompt

    abstract = _text(row.get("abstract"))
    low = 0
    high = min(max_abstract_chars, len(abstract))
    best_prompt: str | None = None
    while low <= high:
        mid = (low + high) // 2
        candidate = _apply_chat_template(
            tokenizer,
            _messages(row, max_abstract_chars=mid),
        )
        if _prompt_token_length(tokenizer, candidate) <= target_prompt_tokens:
            best_prompt = candidate
            low = mid + 1
        else:
            high = mid - 1
    if best_prompt is not None:
        return best_prompt
    raise ValueError(
        f"Prompt metadata exceeds token budget for record {row.get('record_id')!r}"
    )


def _render_prompt_without_tokenizer(
    row: Mapping[str, Any],
    *,
    max_abstract_chars: int,
) -> str:
    messages = _messages(row, max_abstract_chars=max_abstract_chars)
    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}"
        for message in messages
    ) + "\n\nASSISTANT:\n"


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


def _messages(row: Mapping[str, Any], *, max_abstract_chars: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _citation_user_prompt(row, max_abstract_chars=max_abstract_chars)},
    ]


def _citation_user_prompt(row: Mapping[str, Any], *, max_abstract_chars: int) -> str:
    return "\n".join(
        [
            "Evaluate this arXiv paper metadata for a security-domain mid-training corpus.",
            "The main decision is whether the paper is security/privacy/cryptography relevant enough to keep.",
            "Reject high-quality general CS/math/ML/systems papers when the abstract has no real security angle.",
            "Keep directly security-relevant work and strong security-adjacent theory worth full-text review.",
            "Do not reject theoretical crypto or information-theoretic security just because it is abstract or mathematical.",
            STRICT_KEEP_RULES,
            "Use should_keep=true only when security_relevance is at least 2 and quality is at least 2.",
            OUTPUT_CONTRACT,
            "",
            "Paper metadata:",
            f"arxiv_id: {_truncate(_text(row.get('arxiv_id')), 80)}",
            f"title: {_truncate(_text(row.get('title')), 800)}",
            f"authors: {_truncate(_join_list(row.get('authors')), 1200)}",
            f"primary_category: {_truncate(_text(row.get('primary_category')), 120)}",
            f"categories: {_truncate(_join_list(row.get('categories')), 800)}",
            f"doi: {_truncate(_text(row.get('doi')), 200)}",
            f"journal_ref: {_truncate(_text(row.get('journal_ref')), 800)}",
            "",
            "Abstract:",
            _truncate(_text(row.get("abstract")), max_abstract_chars),
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
    if parsed.get("parse_status") == "parse_failure":
        return {
            "should_keep": False,
            "passed": False,
            "reason": "parse_failure_not_accepted_by_strict_policy",
        }
    security_relevance = parsed.get("security_relevance")
    quality = parsed.get("quality")
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
    if security_relevance >= 2 and quality >= 2:
        return {
            "should_keep": True,
            "passed": True,
            "reason": "security_relevance_at_least_2_and_quality_at_least_2",
        }
    return {
        "should_keep": False,
        "passed": False,
        "reason": "strict_policy_requires_security_2_quality_2",
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
        str(row.get("source_id") or "arxiv"),
        str(row.get("record_id") or ""),
        str(row.get("content_hash") or ""),
    )


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


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


def _preview(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _print_progress(
    shard_id: int,
    scored: int,
    kept: int,
    parse_failures: int,
    start: float,
) -> None:
    elapsed = max(time.monotonic() - start, 1e-6)
    print(
        f"citation-abstract shard={shard_id} scored={scored:,} "
        f"kept={kept:,} rejected={scored - kept:,} "
        f"parse_failures={parse_failures:,} "
        f"records_per_sec={scored / elapsed:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit("Use `modal run scripts/classify/modal_citation_qwen_filtering.py ...`")
