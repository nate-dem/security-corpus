#!/usr/bin/env python3
"""Standalone Qwen title+abstract filter for arXiv Parquet.

This script does not import the security-corpus repo. It is meant to be copied
to a separate GPU workstation along with the arXiv paper-level Parquet records.

Install on the GPU machine:

    python -m pip install pyarrow transformers vllm

Dry run:

    python score_arxiv_abstract_qwen_standalone.py \
      --input /path/to/source_id=arxiv \
      --dry-run \
      --dry-run-limit 2

Real run on two GPUs:

    python score_arxiv_abstract_qwen_standalone.py \
      --input /path/to/source_id=arxiv \
      --output qwen_arxiv_abstract.jsonl \
      --model Qwen/Qwen3-14B \
      --tensor-parallel-size 2 \
      --dtype float16 \
      --batch-size 32

The output JSONL is append/resume safe. If the job stops, rerun the same command
and already-scored records will be skipped.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import pyarrow.dataset as ds


TASK = "arxiv_abstract"
PROMPT_VERSION = "qwen-arxiv-abstract-v1"
INPUT_KIND = "arxiv_metadata_abstract"
DEFAULT_MODEL = "Qwen/Qwen3-14B"
DEFAULT_OUTPUT = Path("qwen_arxiv_abstract.jsonl")
REQUIRED_COLUMNS = ("record_id", "content_hash", "title", "abstract")
OPTIONAL_COLUMNS = (
    "source_id",
    "arxiv_id",
    "authors",
    "categories",
    "primary_category",
    "doi",
    "journal_ref",
)

SYSTEM_PROMPT = (
    "You are filtering academic papers for a cybersecurity midtraining corpus. "
    "Judge only from the provided title, abstract, categories, and metadata. "
    "Your job is to decide whether the paper is security-relevant enough to keep "
    "in the corpus. Be strict about security relevance: do not keep general "
    "CS, ML, math, systems, or software papers unless the abstract clearly connects "
    "to security, privacy, cryptography, adversaries, threats, vulnerabilities, "
    "defense, misuse, robustness against attacks, or secure design. Any of these "
    "topics are considered security-relevant. Return exactly one compact "
    "JSON object and no markdown. Do not include chain-of-thought or any  "
    "explanation outside the JSON object."
)

OUTPUT_CONTRACT = (
    'Use this schema: {"security_relevance": 0, "quality": 0, "should_keep": false}. '
    "security_relevance: 0 not security-relevant, "
    "1 weak or speculative security relevance, "
    "2 security-adjacent or useful security context, "
    "3 directly security-relevant. "
    "quality: 0 garbage/broken/spam, 1 low value or thin, "
    "2 usable, 3 high quality/substantive. "
    "Set should_keep using your judgment: keep papers that are directly security-relevant "
    "or meaningfully security-adjacent, and reject papers that are high-quality but only "
    "generally technical. Reject any paper that is low-quality."
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_args(args)

    if args.overwrite and args.output.exists():
        args.output.unlink()

    done_keys = _load_done_keys(args.output)
    _warn_if_model_may_not_fit(args)
    rows = _iter_rows(args, done_keys)

    if args.dry_run:
        _print_dry_run(rows, args)
        return 0

    _run_scoring(rows, args)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="arXiv paper-level normalized Parquet file or directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Append/resume JSONL output. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--read-batch-size", type=int, default=2048)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument(
        "--quantization",
        default=None,
        help="Optional vLLM quantization mode, e.g. awq or gptq.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="Optional model download/cache directory.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--parse-failure-should-keep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep parse failures for review by default.",
    )
    parser.add_argument("--abstract-preview-chars", type=int, default=700)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-limit", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in the range [0, --num-shards)")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")


def _run_scoring(rows: Iterable[dict[str, Any]], args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    args.output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": True,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    if args.download_dir is not None:
        llm_kwargs["download_dir"] = str(args.download_dir)

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    start = time.monotonic()
    scored = 0
    kept = 0
    parse_failures = 0
    batch_rows: list[dict[str, Any]] = []
    batch_prompts: list[str] = []

    with args.output.open("a", encoding="utf-8") as handle:
        for row in rows:
            batch_rows.append(row)
            batch_prompts.append(render_prompt(row, tokenizer=tokenizer))
            if len(batch_rows) >= args.batch_size:
                n, k, p = _score_batch(
                    llm,
                    sampling_params,
                    batch_rows,
                    batch_prompts,
                    handle,
                    args,
                )
                scored += n
                kept += k
                parse_failures += p
                _print_progress(scored, kept, parse_failures, start)
                batch_rows = []
                batch_prompts = []

        if batch_rows:
            n, k, p = _score_batch(
                llm,
                sampling_params,
                batch_rows,
                batch_prompts,
                handle,
                args,
            )
            scored += n
            kept += k
            parse_failures += p
            _print_progress(scored, kept, parse_failures, start)

    print(
        "Complete: "
        f"scored={scored:,}, kept={kept:,}, rejected={scored - kept:,}, "
        f"parse_failures={parse_failures:,}, output={args.output}"
    )


def _score_batch(
    llm: Any,
    sampling_params: Any,
    rows: list[dict[str, Any]],
    prompts: list[str],
    handle: Any,
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    scored_at = datetime.now(timezone.utc).isoformat()
    kept = 0
    parse_failures = 0

    for row, output in zip(rows, outputs):
        response_text = output.outputs[0].text if output.outputs else ""
        parsed = parse_qwen_response(
            response_text,
            parse_failure_should_keep=args.parse_failure_should_keep,
        )
        if parsed["should_keep"]:
            kept += 1
        if parsed["parse_status"] == "parse_failure":
            parse_failures += 1

        out = {
            "source_id": row.get("source_id") or "arxiv",
            "record_id": row.get("record_id"),
            "content_hash": row.get("content_hash"),
            "qwen_security_relevance": parsed["security_relevance"],
            "qwen_quality": parsed["quality"],
            "qwen_should_keep": parsed["should_keep"],
            "qwen_reason": parsed["reason"],
            "qwen_parse_status": parsed["parse_status"],
            "qwen_model": args.model,
            "qwen_prompt_version": PROMPT_VERSION,
            "qwen_scored_at": scored_at,
            "qwen_shard_id": str(args.shard_id),
            "qwen_task": TASK,
            "qwen_input_kind": INPUT_KIND,
            "qwen_raw_response": response_text,
            "arxiv_id": row.get("arxiv_id"),
            "title": row.get("title"),
            "primary_category": row.get("primary_category"),
            "categories": row.get("categories"),
            "abstract_preview": _preview(
                row.get("abstract"),
                args.abstract_preview_chars,
            ),
        }
        handle.write(json.dumps(out, ensure_ascii=True))
        handle.write("\n")

    handle.flush()
    return len(rows), kept, parse_failures


def _iter_rows(
    args: argparse.Namespace,
    done_keys: set[tuple[str, str, str]],
) -> Iterable[dict[str, Any]]:
    dataset = _open_dataset(args.input)
    columns = _available_columns(dataset)
    scanner = dataset.scanner(
        columns=columns,
        batch_size=max(args.read_batch_size, 1),
    )

    emitted = 0
    arxiv_index = 0
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            row.setdefault("source_id", "arxiv")
            if row.get("source_id") and str(row.get("source_id")) != "arxiv":
                continue
            if any(_is_missing(row.get(column)) for column in REQUIRED_COLUMNS):
                continue

            if arxiv_index % args.num_shards != args.shard_id:
                arxiv_index += 1
                continue
            arxiv_index += 1

            key = _key(row)
            if key in done_keys:
                continue

            for column in OPTIONAL_COLUMNS:
                row.setdefault(column, None)
            yield row

            emitted += 1
            if args.max_records is not None and emitted >= args.max_records:
                return


def _open_dataset(input_path: Path) -> ds.Dataset:
    if input_path.is_file():
        return ds.dataset(str(input_path), format="parquet")
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    arxiv_partition = input_path / "source_id=arxiv"
    if arxiv_partition.is_dir():
        return ds.dataset(str(arxiv_partition), format="parquet")
    if input_path.name == "source_id=arxiv":
        return ds.dataset(str(input_path), format="parquet")
    return ds.dataset(str(input_path), format="parquet", partitioning="hive")


def _available_columns(dataset: ds.Dataset) -> list[str]:
    available = set(dataset.schema.names)
    missing = [column for column in REQUIRED_COLUMNS if column not in available]
    if missing:
        raise ValueError(
            "Input is missing required arXiv abstract columns: "
            + ", ".join(missing)
        )
    return [
        column
        for column in REQUIRED_COLUMNS + OPTIONAL_COLUMNS
        if column in available
    ]


def render_prompt(row: Mapping[str, Any], *, tokenizer: Any | None = None) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": arxiv_abstract_user_prompt(row)},
    ]
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


def arxiv_abstract_user_prompt(row: Mapping[str, Any]) -> str:
    parts = [
        "Evaluate this arXiv paper metadata for a security-domain mid-training corpus.",
        "The main question is security/privacy/cryptography/safety relevance.",
        "Reject high-quality general CS/math/ML/systems papers with no security angle.",
        "Keep likely security-relevant or borderline adjacent work worth full-text review.",
        OUTPUT_CONTRACT,
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


def parse_qwen_response(
    response_text: str,
    *,
    parse_failure_should_keep: bool | None = True,
) -> dict[str, Any]:
    stripped = response_text.strip()
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


def _validate_payload(payload: Any, status: str) -> dict[str, Any] | None:
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
    return {
        "security_relevance": security_relevance,
        "quality": quality,
        "should_keep": should_keep,
        "reason": reason[:280],
        "parse_status": status,
    }


def _parse_failure(parse_failure_should_keep: bool | None) -> dict[str, Any]:
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


def _load_done_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()

    keys: set[tuple[str, str, str]] = set()
    bad_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            keys.add(_key(row))

    if bad_lines:
        print(
            f"Warning: ignored {bad_lines:,} malformed existing JSONL lines in {path}",
            file=sys.stderr,
        )
    if keys:
        print(f"Resume: found {len(keys):,} already-scored records in {path}")
    return keys


def _print_dry_run(rows: Iterable[dict[str, Any]], args: argparse.Namespace) -> None:
    for index, row in enumerate(rows):
        if index >= args.dry_run_limit:
            break
        print(
            json.dumps(
                {
                    "source_id": row.get("source_id"),
                    "record_id": row.get("record_id"),
                    "content_hash": row.get("content_hash"),
                    "arxiv_id": row.get("arxiv_id"),
                    "prompt_version": PROMPT_VERSION,
                    "prompt": render_prompt(row),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    print("Dry run complete; vLLM was not imported and no model was loaded.")


def _print_progress(
    scored: int,
    kept: int,
    parse_failures: int,
    start: float,
) -> None:
    elapsed = max(time.monotonic() - start, 1e-6)
    rate = scored / elapsed
    kept_pct = kept / scored * 100 if scored else 0.0
    print(
        f"Scored {scored:,} | kept {kept:,} ({kept_pct:.1f}%) | "
        f"parse failures {parse_failures:,} | {rate:.1f} rec/s"
    )


def _warn_if_model_may_not_fit(args: argparse.Namespace) -> None:
    model_name = args.model.lower()
    if "30b" not in model_name:
        return
    if args.quantization or any(marker in model_name for marker in ("awq", "gptq", "int4", "4bit")):
        return
    print(
        "Warning: a non-quantized 30B checkpoint may not fit on 2x24GB RTX 3090s. "
        "If vLLM OOMs, use a quantized checkpoint and pass --quantization when needed.",
        file=sys.stderr,
    )


def _key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("source_id") or "arxiv"),
        str(row.get("record_id") or ""),
        str(row.get("content_hash") or ""),
    )


def _is_missing(value: Any) -> bool:
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


def _preview(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


if __name__ == "__main__":
    raise SystemExit(main())
