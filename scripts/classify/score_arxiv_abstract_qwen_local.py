#!/usr/bin/env python3
"""Score arXiv title+abstract security relevance on a local GPU workstation.

This is the non-Slurm runner for the arXiv abstract Qwen filter. It is designed
for a small GPU box, for example a workstation with two RTX 3090s. Results are
appended to JSONL after every batch, so rerunning the same command resumes by
skipping records already present in the output file.

Example, two GPUs with tensor parallelism:

    CUDA_VISIBLE_DEVICES=0,1 python scripts/classify/score_arxiv_abstract_qwen_local.py \
      --input data/arxiv/normalized \
      --output data/filtering/v3/qwen_arxiv_abstract.jsonl \
      --model Qwen/Qwen3-30B-A3B \
      --tensor-parallel-size 2 \
      --dtype float16 \
      --batch-size 32

Use a quantized model/checkpoint when the full model does not fit in 2x24GB.
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


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))

from classify.qwen import (  # noqa: E402
    QWEN_PROMPT_VERSIONS,
    QwenTask,
    make_qwen_sidecar_row,
    parse_qwen_response,
    render_prompt,
)


TASK = QwenTask.ARXIV_ABSTRACT
DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B"
DEFAULT_OUTPUT = Path("data/filtering/v3/qwen_arxiv_abstract.jsonl")
REQUIRED_PROMPT_COLUMNS = ("record_id", "content_hash", "title", "abstract")
OPTIONAL_PROMPT_COLUMNS = (
    "source_id",
    "arxiv_id",
    "authors",
    "categories",
    "primary_category",
    "doi",
    "journal_ref",
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in the range [0, --num-shards)")
    if args.overwrite and args.output.exists():
        args.output.unlink()

    prompt_version = args.prompt_version or QWEN_PROMPT_VERSIONS[TASK]
    done_keys = _load_done_keys(args.output)
    _warn_if_model_may_not_fit(args)

    rows = _iter_rows(args, done_keys)
    if args.dry_run:
        _print_dry_run(rows, args, prompt_version)
        return 0

    _run_scoring(rows, args, prompt_version)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="arXiv normalized Parquet file or directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Append/resume JSONL output path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--read-batch-size", type=int, default=2048)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Short title+abstract prompts do not need a large context window.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        help="RTX 3090s should usually use float16. Default: float16.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument(
        "--quantization",
        default=None,
        help="Optional vLLM quantization mode, e.g. awq or gptq, when applicable.",
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
    parser.add_argument(
        "--abstract-preview-chars",
        type=int,
        default=700,
        help="Audit preview length stored in JSONL output.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-limit", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _run_scoring(
    rows: Iterable[dict[str, Any]],
    args: argparse.Namespace,
    prompt_version: str,
) -> None:
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
            batch_prompts.append(render_prompt(row, TASK, tokenizer=tokenizer))
            if len(batch_rows) >= args.batch_size:
                batch_scored, batch_kept, batch_parse_failures = _score_batch(
                    llm,
                    sampling_params,
                    batch_rows,
                    batch_prompts,
                    handle,
                    args,
                    prompt_version,
                )
                scored += batch_scored
                kept += batch_kept
                parse_failures += batch_parse_failures
                _print_progress(scored, kept, parse_failures, start)
                batch_rows = []
                batch_prompts = []

        if batch_rows:
            batch_scored, batch_kept, batch_parse_failures = _score_batch(
                llm,
                sampling_params,
                batch_rows,
                batch_prompts,
                handle,
                args,
                prompt_version,
            )
            scored += batch_scored
            kept += batch_kept
            parse_failures += batch_parse_failures
            _print_progress(scored, kept, parse_failures, start)

    print(
        "Complete: "
        f"scored={scored:,}, kept={kept:,}, "
        f"rejected={scored - kept:,}, parse_failures={parse_failures:,}, "
        f"output={args.output}"
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


def _score_batch(
    llm: Any,
    sampling_params: Any,
    rows: list[dict[str, Any]],
    prompts: list[str],
    handle: Any,
    args: argparse.Namespace,
    prompt_version: str,
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
        if parsed.should_keep:
            kept += 1
        if parsed.parse_status == "parse_failure":
            parse_failures += 1

        out = make_qwen_sidecar_row(
            row,
            parsed,
            task=TASK,
            model=args.model,
            prompt_version=prompt_version,
            scored_at=scored_at,
            shard_id=str(args.shard_id),
            raw_response=response_text,
        )
        out.update(
            {
                "title": row.get("title"),
                "primary_category": row.get("primary_category"),
                "categories": row.get("categories"),
                "abstract_preview": _preview(
                    row.get("abstract"),
                    args.abstract_preview_chars,
                ),
            }
        )
        handle.write(json.dumps(out, ensure_ascii=True))
        handle.write("\n")

    handle.flush()
    return len(rows), kept, parse_failures


def _iter_rows(
    args: argparse.Namespace,
    done_keys: set[tuple[str, str, str]],
) -> Iterable[dict[str, Any]]:
    dataset = _open_arxiv_dataset(args.input)
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
            if _has_missing_required_value(row):
                continue

            if arxiv_index % args.num_shards != args.shard_id:
                arxiv_index += 1
                continue
            arxiv_index += 1

            key = _key(row)
            if key in done_keys:
                continue

            for column in OPTIONAL_PROMPT_COLUMNS:
                row.setdefault(column, None)
            yield row

            emitted += 1
            if args.max_records is not None and emitted >= args.max_records:
                return


def _open_arxiv_dataset(input_path: Path) -> ds.Dataset:
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
    missing = [
        column
        for column in REQUIRED_PROMPT_COLUMNS
        if column not in available
    ]
    if missing:
        raise ValueError(
            "Input is missing required arXiv abstract columns: "
            + ", ".join(missing)
        )

    return [
        column
        for column in REQUIRED_PROMPT_COLUMNS + OPTIONAL_PROMPT_COLUMNS
        if column in available
    ]


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


def _print_dry_run(
    rows: Iterable[dict[str, Any]],
    args: argparse.Namespace,
    prompt_version: str,
) -> None:
    for index, row in enumerate(rows):
        if index >= args.dry_run_limit:
            break
        payload = {
            "source_id": row.get("source_id"),
            "record_id": row.get("record_id"),
            "content_hash": row.get("content_hash"),
            "arxiv_id": row.get("arxiv_id"),
            "prompt_version": prompt_version,
            "prompt": render_prompt(row, TASK),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
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


def _has_missing_required_value(row: Mapping[str, Any]) -> bool:
    return any(_is_missing(row.get(column)) for column in REQUIRED_PROMPT_COLUMNS)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("source_id") or "arxiv"),
        str(row.get("record_id") or ""),
        str(row.get("content_hash") or ""),
    )


def _preview(value: Any, max_chars: int) -> str:
    text = "" if value is None else str(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


if __name__ == "__main__":
    raise SystemExit(main())
