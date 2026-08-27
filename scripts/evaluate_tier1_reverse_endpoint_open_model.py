#!/usr/bin/env python3
"""Evaluate an open causal LM on the Tier 1 reverse-endpoint benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable


_LABEL_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)
_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE)


def main() -> int:
    args = _build_parser().parse_args()
    examples = _load_examples(args.input, max_records=args.max_records)
    if not examples:
        raise ValueError(f"No examples loaded from {args.input}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if args.dtype == "float32" else "auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    correct = 0
    parse_failures = 0
    predictions: list[dict[str, Any]] = []

    with args.output.open("w") as handle:
        for index, example in enumerate(examples, start=1):
            prompt = _build_prompt(example)
            response = _generate_response(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
            )
            parsed = _parse_prediction(response, example)
            is_correct = parsed["predicted_label"] == example["answer_label"]
            correct += int(is_correct)
            parse_failures += int(parsed["parse_status"] != "ok")
            row = {
                "benchmark_id": example["benchmark_id"],
                "model": args.model,
                "answer_label": example["answer_label"],
                "answer_cve_id": example["answer_cve_id"],
                "raw_response": response,
                "is_correct": is_correct,
                **parsed,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            predictions.append(row)
            if index % args.progress_every == 0 or index == len(examples):
                elapsed = time.monotonic() - start
                accuracy = correct / index
                print(
                    f"scored={index:,}/{len(examples):,} "
                    f"accuracy={accuracy:.3f} parse_failures={parse_failures:,} "
                    f"elapsed={elapsed:.1f}s"
                )

    summary = _make_summary(args, examples, predictions, correct, parse_failures, start)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmarks/tier1_reverse_endpoint_prediction/reverse_endpoint_prediction_1k.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmarks/tier1_reverse_endpoint_prediction/open_model_predictions.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/benchmarks/tier1_reverse_endpoint_prediction/open_model_summary.json"),
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dtype", choices=["auto", "float32"], default="float32")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def _load_examples(path: Path, *, max_records: int | None) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            examples.append(json.loads(line))
            if max_records is not None and len(examples) >= max_records:
                break
    return examples


def _build_prompt(example: dict[str, Any]) -> str:
    return (
        "You are answering a multiple-choice cybersecurity graph reasoning benchmark.\n"
        "Use only the given question. Respond with exactly the candidate label and CVE ID, "
        "for example: A. CVE-2024-12345\n\n"
        f"{example['question']}\n\nAnswer:"
    )


def _generate_response(model: Any, tokenizer: Any, prompt: str, *, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _parse_prediction(response: str, example: dict[str, Any]) -> dict[str, Any]:
    label_match = _LABEL_RE.search(response)
    cve_match = _CVE_RE.search(response)
    predicted_label = label_match.group(1).upper() if label_match else None
    predicted_cve = cve_match.group(0).upper() if cve_match else None

    if predicted_cve and not predicted_label:
        for candidate in example["candidates"]:
            if candidate["cve_id"].upper() == predicted_cve:
                predicted_label = candidate["label"]
                break
    if predicted_label and not predicted_cve:
        for candidate in example["candidates"]:
            if candidate["label"] == predicted_label:
                predicted_cve = candidate["cve_id"]
                break

    return {
        "predicted_label": predicted_label,
        "predicted_cve_id": predicted_cve,
        "parse_status": "ok" if predicted_label and predicted_cve else "parse_failure",
    }


def _make_summary(
    args: argparse.Namespace,
    examples: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    correct: int,
    parse_failures: int,
    start: float,
) -> dict[str, Any]:
    by_gold_priority: dict[str, Counter] = {}
    by_answer_label: dict[str, Counter] = {}
    for example, prediction in zip(examples, predictions):
        _update_counter(by_gold_priority, str(example["gold_priority"]), prediction["is_correct"])
        _update_counter(by_answer_label, example["answer_label"], prediction["is_correct"])

    elapsed = time.monotonic() - start
    return {
        "model": args.model,
        "input": str(args.input),
        "output": str(args.output),
        "summary": str(args.summary),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "examples": len(examples),
        "correct": correct,
        "accuracy": correct / len(examples),
        "parse_failures": parse_failures,
        "elapsed_seconds": elapsed,
        "seconds_per_example": elapsed / len(examples),
        "by_gold_priority": _counter_summary(by_gold_priority),
        "by_answer_label": _counter_summary(by_answer_label),
    }


def _update_counter(groups: dict[str, Counter], key: str, is_correct: bool) -> None:
    counter = groups.setdefault(key, Counter())
    counter["total"] += 1
    counter["correct"] += int(is_correct)


def _counter_summary(groups: dict[str, Counter]) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            "total": counter["total"],
            "correct": counter["correct"],
            "accuracy": counter["correct"] / counter["total"],
        }
        for key, counter in sorted(groups.items())
    }


if __name__ == "__main__":
    raise SystemExit(main())
