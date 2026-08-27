#!/usr/bin/env python3
"""Evaluate vLLM models on the Tier 1 reverse-endpoint graph benchmark via Modal.

This evaluator is for the current JSON-chain benchmark, not the old
multiple-choice benchmark.

One-time setup:

    modal setup
    modal volume create security-corpus-data

Upload the final benchmark into the Modal volume:

    modal volume put security-corpus-data \
      data/benchmarks/tier1_reverse_endpoint_prediction/reverse_endpoint_gpt55_100_256k_tol4k_final_clean.jsonl \
      /benchmarks/tier1_reverse_endpoint_prediction/reverse_endpoint_gpt55_100_256k_tol4k_final_clean.jsonl

Dry-run prompt assembly without loading a model:

    modal run scripts/evaluate_tier1_reverse_endpoint_modal_vllm.py \
      --dry-run \
      --max-records 1

Evaluate a small long-context model on five examples:

    modal run scripts/evaluate_tier1_reverse_endpoint_modal_vllm.py \
      --model Qwen/Qwen2.5-7B-Instruct-1M \
      --max-records 5 \
      --max-model-len 300000 \
      --max-num-seqs 1 \
      --batch-size 1

Evaluate a larger model with two GPUs:

    modal run scripts/evaluate_tier1_reverse_endpoint_modal_vllm.py \
      --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
      --qwen3-1m-config \
      --tensor-parallel-size 2 \
      --max-records 1 \
      --max-model-len 300000 \
      --max-num-seqs 1 \
      --batch-size 1
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Iterable

import modal


APP_NAME = "security-corpus-tier1-reverse-endpoint-vllm-eval"
VOLUME_NAME = "security-corpus-data"
MOUNT_PATH = Path("/data")
DEFAULT_INPUT = (
    MOUNT_PATH
    / "benchmarks"
    / "tier1_reverse_endpoint_prediction"
    / "reverse_endpoint_gpt55_100_256k_tol4k_final_clean.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    MOUNT_PATH
    / "benchmarks"
    / "tier1_reverse_endpoint_prediction"
    / "open_model_evals"
)
DEFAULT_MODEL_ROOT = MOUNT_PATH / "models"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"
GPU_REQUEST = ["H100:2", "A100-80GB:2"]
QWEN3_30B_2507_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
CHAIN_FIELDS = ("sigma_rule", "attack_technique", "capec", "cwe", "cve")
HOP_SPECS = {
    "attack_to_sigma": ("attack_technique", "detected_by_sigma_rule", "sigma_rule"),
    "capec_to_attack": ("capec", "maps_to_attack_technique", "attack_technique"),
    "cwe_to_capec": ("cwe", "related_attack_pattern", "capec"),
    "cve_to_cwe": ("cve", "has_weakness", "cwe"),
}
FALLBACK_HOP_RELATIONSHIPS = {
    "cwe_to_capec": ("related_attack_pattern", "related_weakness_attack_pattern"),
}
SYSTEM_PROMPT = (
    "You are solving a cybersecurity graph reasoning benchmark. Use only the "
    "provided graph input. Return exactly one JSON object and no markdown."
)


image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.13")
    .entrypoint([])
    .uv_pip_install(
        "transformers",
        "huggingface-hub==0.36.0",
    )
    .run_commands(
        "python -m pip install -U --extra-index-url https://wheels.vllm.ai/nightly vllm"
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name("vllm-cache", create_if_missing=True)
app = modal.App(APP_NAME, image=image)


@app.function(
    gpu=GPU_REQUEST,
    timeout=60 * 60 * 24,
    volumes={
        MOUNT_PATH.as_posix(): volume,
        "/root/.cache/huggingface": hf_cache_volume,
        "/root/.cache/vllm": vllm_cache_volume,
    },
)
def evaluate_reverse_endpoint_benchmark(
    *,
    input_path: str = DEFAULT_INPUT.as_posix(),
    output_root: str = DEFAULT_OUTPUT_ROOT.as_posix(),
    model_root: str = DEFAULT_MODEL_ROOT.as_posix(),
    model: str = DEFAULT_MODEL,
    qwen3_1m_config: bool = False,
    max_records: int | None = None,
    batch_size: int = 1,
    max_tokens: int = 384,
    max_model_len: int = 300_000,
    max_num_seqs: int = 1,
    max_num_batched_tokens: int | None = None,
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.86,
    shard_id: int = 0,
    num_shards: int = 1,
    temperature: float = 0.0,
    validate_prompt_tokens: bool = True,
    skip_too_long_prompts: bool = False,
    enable_chunked_prefill: bool = False,
    enforce_eager: bool = False,
    attention_backend: str | None = None,
    vllm_use_v1: str | None = None,
    allow_long_max_model_len: bool = False,
    overwrite: bool = False,
    trust_remote_code: bool = True,
    use_chat_template: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Evaluate one model/shard and write JSONL predictions plus a summary."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if max_num_seqs < 1:
        raise ValueError("max_num_seqs must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must be in [0, num_shards)")

    volume.reload()
    input_file = Path(input_path)
    model_slug = _slug_model_name(model)
    output_dir = Path(output_root) / model_slug
    if num_shards > 1:
        output_path = output_dir / f"predictions-shard-{shard_id}-of-{num_shards}.jsonl"
        summary_path = output_dir / f"summary-shard-{shard_id}-of-{num_shards}.json"
    else:
        output_path = output_dir / "predictions.jsonl"
        summary_path = output_dir / "summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite and output_path.exists():
        output_path.unlink()

    done_ids = _load_done_ids(output_path)
    examples = list(
        _iter_examples(
            input_file,
            max_records=max_records,
            shard_id=shard_id,
            num_shards=num_shards,
            done_ids=done_ids,
        )
    )
    if dry_run:
        preview = []
        for example in examples[: max_records or 1]:
            prompt = _assemble_model_prompt(example)
            preview.append(
                {
                    "benchmark_id": example["benchmark_id"],
                    "metadata_prompt_tokens": example.get("metadata", {}).get("model_prompt_token_count"),
                    "prompt_prefix": prompt[:4000],
                }
            )
        return {
            "dry_run": True,
            "input_path": input_path,
            "examples_loaded": len(examples),
            "already_done": len(done_ids),
            "preview": preview,
        }

    model_path = model
    model_display_name = model
    if qwen3_1m_config:
        model_path = _prepare_qwen3_1m_model(
            model=model,
            model_root=Path(model_root),
        )
        model_display_name = f"{model} (config_1m)"
        # The Qwen model card recommends DUAL_CHUNK_FLASH_ATTN for 1M serving.
        # The Modal image installs vLLM from the nightly wheel index so this
        # backend is available for the 1M-config path.
        attention_backend = attention_backend or "DUAL_CHUNK_FLASH_ATTN"
        vllm_use_v1 = vllm_use_v1 if vllm_use_v1 is not None else "0"
        enable_chunked_prefill = True
        enforce_eager = True
        allow_long_max_model_len = True
        max_num_batched_tokens = max_num_batched_tokens or 131_072
        volume.commit()

    if attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend
    if vllm_use_v1 is not None:
        os.environ["VLLM_USE_V1"] = str(vllm_use_v1)
    if allow_long_max_model_len:
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    llm_kwargs = {
        "model": model_path,
        "dtype": dtype,
        "max_model_len": max_model_len,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_seqs": max_num_seqs,
        "enable_prefix_caching": False,
        "trust_remote_code": trust_remote_code,
    }
    if enable_chunked_prefill:
        llm_kwargs["enable_chunked_prefill"] = True
    if enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    llm = LLM(
        **llm_kwargs,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    start = time.monotonic()
    written = 0
    batch_examples: list[dict[str, Any]] = []
    batch_prompts: list[str] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for example in examples:
            batch_examples.append(example)
            batch_prompts.append(
                _render_prompt(
                    example,
                    tokenizer=tokenizer,
                    use_chat_template=use_chat_template,
                )
            )
            if validate_prompt_tokens:
                prompt_token_count = _model_prompt_token_count(tokenizer, batch_prompts[-1])
                if prompt_token_count > max_model_len:
                    message = (
                        f"Benchmark example {example['benchmark_id']} is {prompt_token_count:,} "
                        f"{model} tokens, which exceeds max_model_len={max_model_len:,}. "
                        "Regenerate a shorter benchmark for this tokenizer, use a model/config "
                        "with a larger context window, or pass --skip-too-long-prompts to skip."
                    )
                    if skip_too_long_prompts:
                        print(message, flush=True)
                        batch_examples.pop()
                        batch_prompts.pop()
                        continue
                    raise ValueError(message)
            if len(batch_examples) >= batch_size:
                written += _score_batch(
                    llm,
                    sampling_params,
                    batch_examples,
                    batch_prompts,
                    handle,
                    model=model_display_name,
                )
                batch_examples = []
                batch_prompts = []
                _print_progress(written, len(examples), start)

        if batch_examples:
            written += _score_batch(
                llm,
                sampling_params,
                batch_examples,
                batch_prompts,
                handle,
                model=model_display_name,
            )
            _print_progress(written, len(examples), start)

    rows = _load_prediction_rows(output_path)
    summary = _summarize_prediction_rows(
        rows,
        model=model,
        model_path=model_path,
        qwen3_1m_config=qwen3_1m_config,
        input_path=input_path,
        output_path=output_path.as_posix(),
        max_records=max_records,
        shard_id=shard_id,
        num_shards=num_shards,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=enable_chunked_prefill,
        enforce_eager=enforce_eager,
        attention_backend=attention_backend,
        vllm_use_v1=vllm_use_v1,
        allow_long_max_model_len=allow_long_max_model_len,
        validate_prompt_tokens=validate_prompt_tokens,
        skip_too_long_prompts=skip_too_long_prompts,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    volume.commit()
    return {
        "dry_run": False,
        "model": model,
        "model_path": model_path,
        "qwen3_1m_config": qwen3_1m_config,
        "input_path": input_path,
        "output_path": output_path.as_posix(),
        "summary_path": summary_path.as_posix(),
        "examples_loaded": len(examples),
        "already_done_at_start": len(done_ids),
        "written_this_run": written,
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "summary": summary,
    }


def _score_batch(
    llm: Any,
    sampling_params: Any,
    examples: list[dict[str, Any]],
    prompts: list[str],
    handle: Any,
    *,
    model: str,
) -> int:
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    for example, output in zip(examples, outputs):
        response = output.outputs[0].text.strip() if output.outputs else ""
        row = _make_prediction_row(
            example=example,
            model=model,
            raw_response=response,
        )
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    return len(examples)


def _prepare_qwen3_1m_model(
    *,
    model: str,
    model_root: Path,
) -> str:
    """Download Qwen3-30B-2507 and replace config.json with config_1m.json."""
    if model != QWEN3_30B_2507_MODEL:
        raise ValueError(
            "--qwen3-1m-config currently supports only "
            f"{QWEN3_30B_2507_MODEL}, got {model!r}"
        )

    target_dir = model_root / f"{_slug_model_name(model)}__config_1m"
    marker = target_dir / ".qwen3_1m_config_ready"
    if not marker.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=model,
            local_dir=target_dir.as_posix(),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        config_path = target_dir / "config.json"
        config_1m_path = target_dir / "config_1m.json"
        backup_path = target_dir / "config_native_262k.json"
        if not config_1m_path.exists():
            raise FileNotFoundError(f"Missing expected 1M config: {config_1m_path}")
        if config_path.exists() and not backup_path.exists():
            shutil.copy2(config_path, backup_path)
        shutil.copy2(config_1m_path, config_path)
        marker.write_text(
            json.dumps(
                {
                    "model": model,
                    "prepared_at": datetime.now(timezone.utc).isoformat(),
                    "config": "config_1m.json",
                    "native_config_backup": backup_path.name,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return target_dir.as_posix()


def _render_prompt(
    example: dict[str, Any],
    *,
    tokenizer: Any,
    use_chat_template: bool,
) -> str:
    prompt = _assemble_model_prompt(example)
    if not use_chat_template:
        return prompt + "\n\nAnswer:\n"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return SYSTEM_PROMPT + "\n\n" + prompt + "\n\nAnswer:\n"


def _assemble_model_prompt(example: dict[str, Any]) -> str:
    model_input = {
        "target_cve_description": example["input"]["target_cve_description"],
        "target_sigma_rule": example["input"]["target_sigma_rule"],
        "nodes": example["input"]["nodes"],
        "edges": example["input"]["edges"],
    }
    return (
        f"{example['question'].strip()}\n\n"
        "Provided graph input:\n"
        f"{json.dumps(model_input, ensure_ascii=False, sort_keys=True)}\n\n"
        "Required output JSON schema:\n"
        f"{json.dumps(example['output_json_schema'], ensure_ascii=False, sort_keys=True)}"
    )


def _make_prediction_row(
    *,
    example: dict[str, Any],
    model: str,
    raw_response: str,
) -> dict[str, Any]:
    parsed, parse_status = _parse_model_response(raw_response)
    score = _score_prediction(parsed, example)
    return {
        "benchmark_id": example["benchmark_id"],
        "model": model,
        "parse_status": parse_status,
        "raw_response": raw_response,
        "parsed_response": parsed,
        "metadata_prompt_tokens": example.get("metadata", {}).get("model_prompt_token_count"),
        **score,
    }


def _parse_model_response(response: str) -> tuple[dict[str, Any] | None, str]:
    json_text = _extract_first_json_object(response)
    if not json_text:
        return None, "no_json_object"
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(parsed, dict):
        return None, "json_not_object"
    return parsed, "ok"


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
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


def _score_prediction(parsed: dict[str, Any] | None, example: dict[str, Any]) -> dict[str, Any]:
    gold = example["answer"]
    gold_chain = gold["chain"]
    pred_chain = _prediction_chain(parsed)
    pred_cve_id = _prediction_cve_id(parsed, pred_chain)
    edge_index = _edge_index(example["input"]["edges"])

    cve_id_correct = _same_cve_id(pred_cve_id, gold["cve_id"])
    field_correct = {
        field: _clean_string(pred_chain.get(field)) == _clean_string(gold_chain[field])
        for field in CHAIN_FIELDS
    }
    hop_correct = {
        hop_name: _hop_exists(hop_name, pred_chain, edge_index)
        for hop_name in HOP_SPECS
    }
    points = int(cve_id_correct) + sum(int(value) for value in hop_correct.values())
    exact_chain_match = cve_id_correct and all(field_correct.values())
    return {
        "cve_id_correct": cve_id_correct,
        "field_correct": field_correct,
        "hop_correct": hop_correct,
        "points": points,
        "max_points": 1 + len(HOP_SPECS),
        "score": points / (1 + len(HOP_SPECS)),
        "exact_chain_match": exact_chain_match,
        "predicted_cve_id": pred_cve_id,
        "predicted_chain": pred_chain,
        "gold_cve_id": gold["cve_id"],
        "gold_chain": gold_chain,
    }


def _prediction_chain(parsed: dict[str, Any] | None) -> dict[str, str | None]:
    chain = parsed.get("chain") if isinstance(parsed, dict) else None
    if not isinstance(chain, dict):
        chain = {}
    return {field: _clean_string(chain.get(field)) for field in CHAIN_FIELDS}


def _prediction_cve_id(
    parsed: dict[str, Any] | None,
    pred_chain: dict[str, str | None],
) -> str | None:
    if isinstance(parsed, dict):
        cve_id = _clean_string(parsed.get("cve_id"))
        if cve_id:
            return cve_id.upper()
    chain_cve = _clean_string(pred_chain.get("cve"))
    if chain_cve and chain_cve.startswith("nvd:"):
        return chain_cve.split(":", 1)[1].upper()
    return chain_cve.upper() if chain_cve else None


def _edge_index(edges: Iterable[dict[str, Any]]) -> set[tuple[str, str, str]]:
    index: set[tuple[str, str, str]] = set()
    for edge in edges:
        source = _clean_string(edge.get("source"))
        relationship = _clean_string(edge.get("relationship"))
        target = _clean_string(edge.get("target"))
        if source and relationship and target:
            index.add((source, relationship, target))
    return index


def _hop_exists(
    hop_name: str,
    pred_chain: dict[str, str | None],
    edge_index: set[tuple[str, str, str]],
) -> bool:
    source_field, relationship, target_field = HOP_SPECS[hop_name]
    source = pred_chain.get(source_field)
    target = pred_chain.get(target_field)
    if not source or not target:
        return False
    relationships = FALLBACK_HOP_RELATIONSHIPS.get(hop_name, (relationship,))
    return any((source, rel, target) in edge_index for rel in relationships)


def _iter_examples(
    input_path: Path,
    *,
    max_records: int | None,
    shard_id: int,
    num_shards: int,
    done_ids: set[str],
) -> Iterable[dict[str, Any]]:
    yielded = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index % num_shards != shard_id:
                continue
            example = json.loads(line)
            benchmark_id = str(example["benchmark_id"])
            if benchmark_id in done_ids:
                continue
            yield example
            yielded += 1
            if max_records is not None and yielded >= max_records:
                break


def _load_done_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            benchmark_id = row.get("benchmark_id")
            if benchmark_id:
                done.add(str(benchmark_id))
    return done


def _load_prediction_rows(output_path: Path) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []
    rows = []
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def _summarize_prediction_rows(
    rows: Iterable[dict[str, Any]],
    *,
    model: str,
    model_path: str,
    qwen3_1m_config: bool,
    input_path: str,
    output_path: str,
    max_records: int | None,
    shard_id: int,
    num_shards: int,
    max_model_len: int,
    max_tokens: int,
    max_num_batched_tokens: int | None,
    enable_chunked_prefill: bool,
    enforce_eager: bool,
    attention_backend: str | None,
    vllm_use_v1: str | None,
    allow_long_max_model_len: bool,
    validate_prompt_tokens: bool,
    skip_too_long_prompts: bool,
) -> dict[str, Any]:
    rows = list(rows)
    total = len(rows)
    parse_failures = sum(1 for row in rows if row.get("parse_status") != "ok")
    exact = sum(1 for row in rows if row.get("exact_chain_match"))
    points = sum(int(row.get("points", 0)) for row in rows)
    max_points = sum(int(row.get("max_points", 0)) for row in rows)
    cve_correct = sum(1 for row in rows if row.get("cve_id_correct"))

    hop_counts: dict[str, Counter[str]] = {
        hop_name: Counter() for hop_name in HOP_SPECS
    }
    field_counts: dict[str, Counter[str]] = {
        field: Counter() for field in CHAIN_FIELDS
    }
    for row in rows:
        for hop_name, is_correct in row.get("hop_correct", {}).items():
            hop_counts.setdefault(hop_name, Counter())["total"] += 1
            hop_counts[hop_name]["correct"] += int(is_correct)
        for field, is_correct in row.get("field_correct", {}).items():
            field_counts.setdefault(field, Counter())["total"] += 1
            field_counts[field]["correct"] += int(is_correct)

    return {
        "model": model,
        "model_path": model_path,
        "qwen3_1m_config": qwen3_1m_config,
        "input_path": input_path,
        "output_path": output_path,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "max_records": max_records,
        "shard_id": shard_id,
        "num_shards": num_shards,
        "max_model_len": max_model_len,
        "max_tokens": max_tokens,
        "max_num_batched_tokens": max_num_batched_tokens,
        "enable_chunked_prefill": enable_chunked_prefill,
        "enforce_eager": enforce_eager,
        "attention_backend": attention_backend,
        "vllm_use_v1": vllm_use_v1,
        "allow_long_max_model_len": allow_long_max_model_len,
        "validate_prompt_tokens": validate_prompt_tokens,
        "skip_too_long_prompts": skip_too_long_prompts,
        "examples": total,
        "parse_failures": parse_failures,
        "parse_failure_rate": _safe_rate(parse_failures, total),
        "exact_chain_matches": exact,
        "exact_chain_accuracy": _safe_rate(exact, total),
        "cve_id_correct": cve_correct,
        "cve_id_accuracy": _safe_rate(cve_correct, total),
        "points": points,
        "max_points": max_points,
        "partial_credit_accuracy": _safe_rate(points, max_points),
        "hop_accuracy": _counter_rates(hop_counts),
        "field_accuracy": _counter_rates(field_counts),
    }


def _counter_rates(groups: dict[str, Counter[str]]) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            "total": counter["total"],
            "correct": counter["correct"],
            "accuracy": _safe_rate(counter["correct"], counter["total"]),
        }
        for key, counter in sorted(groups.items())
    }


def _same_cve_id(predicted: str | None, gold: str) -> bool:
    predicted_text = _clean_string(predicted)
    gold_text = _clean_string(gold)
    if not predicted_text or not gold_text:
        return False
    return predicted_text.upper() == gold_text.upper()


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _model_prompt_token_count(tokenizer: Any, prompt: str) -> int:
    encoded = tokenizer(prompt, add_special_tokens=False)
    return len(encoded["input_ids"])


def _slug_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model).strip("_") or "model"


def _print_progress(written: int, total: int, start: float) -> None:
    elapsed = time.monotonic() - start
    rate = written / elapsed if elapsed else 0.0
    print(
        f"scored={written:,}/{total:,} "
        f"rate={rate:.3f}/s elapsed={elapsed:.1f}s",
        flush=True,
    )


@app.local_entrypoint()
def main(
    input_path: str = DEFAULT_INPUT.as_posix(),
    output_root: str = DEFAULT_OUTPUT_ROOT.as_posix(),
    model_root: str = DEFAULT_MODEL_ROOT.as_posix(),
    model: str = DEFAULT_MODEL,
    qwen3_1m_config: bool = False,
    max_records: int | None = None,
    batch_size: int = 1,
    max_tokens: int = 384,
    max_model_len: int = 300_000,
    max_num_seqs: int = 1,
    max_num_batched_tokens: int | None = None,
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.86,
    shard_id: int = 0,
    num_shards: int = 1,
    temperature: float = 0.0,
    validate_prompt_tokens: bool = True,
    skip_too_long_prompts: bool = False,
    enable_chunked_prefill: bool = False,
    enforce_eager: bool = False,
    attention_backend: str | None = None,
    vllm_use_v1: str | None = None,
    allow_long_max_model_len: bool = False,
    overwrite: bool = False,
    trust_remote_code: bool = True,
    use_chat_template: bool = True,
    dry_run: bool = False,
) -> None:
    """Launch one benchmark eval on Modal."""
    runner = evaluate_reverse_endpoint_benchmark
    result = runner.remote(
        input_path=input_path,
        output_root=output_root,
        model_root=model_root,
        model=model,
        qwen3_1m_config=qwen3_1m_config,
        max_records=max_records,
        batch_size=batch_size,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        shard_id=shard_id,
        num_shards=num_shards,
        temperature=temperature,
        validate_prompt_tokens=validate_prompt_tokens,
        skip_too_long_prompts=skip_too_long_prompts,
        enable_chunked_prefill=enable_chunked_prefill,
        enforce_eager=enforce_eager,
        attention_backend=attention_backend,
        vllm_use_v1=vllm_use_v1,
        allow_long_max_model_len=allow_long_max_model_len,
        overwrite=overwrite,
        trust_remote_code=trust_remote_code,
        use_chat_template=use_chat_template,
        dry_run=dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
