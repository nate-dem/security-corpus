#!/usr/bin/env python3
"""Evaluate OpenAI API models on the Tier 1 reverse-endpoint graph benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from ingest.derived.tier1_reverse_endpoint_eval import (
    build_model_prompt,
    make_prediction_row,
    summarize_prediction_rows,
)


DEFAULT_INPUT = Path(
    "data/benchmarks/tier1_reverse_endpoint_prediction/"
    "reverse_endpoint_gpt55_100_256k_tol4k_final_clean.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path(
    "data/benchmarks/tier1_reverse_endpoint_prediction/openai_evals"
)
SYSTEM_PROMPT = (
    "You are solving a cybersecurity graph reasoning benchmark. Use only the "
    "provided graph input. Return exactly one JSON object and no markdown."
)


def main() -> int:
    args = _build_parser().parse_args()
    examples = _load_examples(args.input, max_records=args.max_records)
    if not examples:
        raise ValueError(f"No examples loaded from {args.input}")

    output_dir = args.output_root / _slug_model_name(args.model)
    output_path = args.output or output_dir / "predictions.jsonl"
    summary_path = args.summary or output_dir / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_path.exists():
        output_path.unlink()

    done_ids = _load_done_ids(output_path)
    from openai import OpenAI

    client = OpenAI(
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=args.base_url
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or None,
    )

    rows = _load_prediction_rows(output_path)
    start = time.monotonic()
    total_input_tokens = 0
    total_output_tokens = 0
    total_estimated_cost = 0.0
    written = 0

    with output_path.open("a", encoding="utf-8") as handle:
        for example in examples:
            benchmark_id = str(example["benchmark_id"])
            if benchmark_id in done_ids:
                continue
            prompt = build_model_prompt(example)
            request = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "max_completion_tokens": args.max_completion_tokens,
            }
            if args.reasoning_effort:
                request["reasoning_effort"] = args.reasoning_effort

            response = client.chat.completions.create(**request)
            raw_response = response.choices[0].message.content or ""
            usage = _usage_dict(response)
            input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            output_tokens = int(
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            )
            estimated_cost = _estimate_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost_per_million=args.input_cost_per_million,
                output_cost_per_million=args.output_cost_per_million,
            )
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_estimated_cost += estimated_cost

            row = make_prediction_row(
                example=example,
                model=args.model,
                raw_response=raw_response,
                prompt_tokens=input_tokens,
            )
            row["usage"] = usage
            row["estimated_cost_usd"] = estimated_cost
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            rows.append(row)
            written += 1
            _print_progress(
                written=written,
                total=len(examples) - len(done_ids),
                total_estimated_cost=total_estimated_cost,
                start=start,
            )

            if (
                args.openai_max_cost_usd is not None
                and total_estimated_cost >= args.openai_max_cost_usd
            ):
                print(
                    f"Reached local cost cap ${args.openai_max_cost_usd:.4f}; stopping.",
                    flush=True,
                )
                break

    summary = summarize_prediction_rows(
        rows,
        model=args.model,
        input_path=str(args.input),
        output_path=str(output_path),
    )
    summary.update(
        {
            "summary_path": str(summary_path),
            "base_url": str(client.base_url),
            "max_records_requested": args.max_records,
            "written_this_run": written,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_estimated_cost_usd": total_estimated_cost,
            "input_cost_per_million": args.input_cost_per_million,
            "output_cost_per_million": args.output_cost_per_million,
            "local_cost_cap_usd": args.openai_max_cost_usd,
            "elapsed_seconds": round(time.monotonic() - start, 2),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=384)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--openai-max-cost-usd", type=float, default=None)
    parser.add_argument("--input-cost-per-million", type=float, default=0.25)
    parser.add_argument("--output-cost-per-million", type=float, default=2.00)
    return parser


def _load_examples(path: Path, *, max_records: int | None) -> list[dict[str, Any]]:
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            examples.append(json.loads(line))
            if max_records is not None and len(examples) >= max_records:
                break
    return examples


def _load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            benchmark_id = row.get("benchmark_id")
            if benchmark_id:
                done.add(str(benchmark_id))
    return done


def _load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def _usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {
        key: getattr(usage, key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if hasattr(usage, key)
    }


def _estimate_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    input_cost_per_million: float,
    output_cost_per_million: float,
) -> float:
    return (
        input_tokens * input_cost_per_million / 1_000_000
        + output_tokens * output_cost_per_million / 1_000_000
    )


def _slug_model_name(model: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "__" for char in model)


def _print_progress(
    *,
    written: int,
    total: int,
    total_estimated_cost: float,
    start: float,
) -> None:
    elapsed = time.monotonic() - start
    print(
        f"scored={written:,}/{total:,} "
        f"estimated_cost=${total_estimated_cost:.4f} elapsed={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
