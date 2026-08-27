"""Scoring helpers for the Tier 1 reverse-endpoint graph benchmark."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from typing import Any, Iterable

from ingest.derived.tier1_reverse_endpoint_benchmark import assemble_reverse_endpoint_model_prompt


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


def build_model_prompt(example: dict[str, Any]) -> str:
    """Return exactly what the downstream model should see for one example."""
    return assemble_reverse_endpoint_model_prompt(example)


def parse_model_response(response: str) -> tuple[dict[str, Any] | None, str]:
    """Parse the first JSON object from a model response."""
    json_text = extract_first_json_object(response)
    if not json_text:
        return None, "no_json_object"
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(parsed, dict):
        return None, "json_not_object"
    return parsed, "ok"


def extract_first_json_object(text: str) -> str | None:
    """Extract the first balanced JSON object, tolerating markdown wrappers."""
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


def score_prediction(parsed: dict[str, Any] | None, example: dict[str, Any]) -> dict[str, Any]:
    """Score a parsed model response against one benchmark example.

    The score is five points total:
    - 1 point for the target CVE ID
    - 1 point for each valid predicted edge hop in the provided graph

    Exact chain accuracy is tracked separately because a model can earn partial
    hop credit by finding a valid distractor path.
    """
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


def make_prediction_row(
    *,
    example: dict[str, Any],
    model: str,
    raw_response: str,
    prompt_tokens: int | None = None,
) -> dict[str, Any]:
    parsed, parse_status = parse_model_response(raw_response)
    score = score_prediction(parsed, example)
    return {
        "benchmark_id": example["benchmark_id"],
        "model": model,
        "parse_status": parse_status,
        "raw_response": raw_response,
        "parsed_response": parsed,
        "prompt_tokens": prompt_tokens,
        **score,
    }


def summarize_prediction_rows(
    rows: Iterable[dict[str, Any]],
    *,
    model: str,
    input_path: str,
    output_path: str,
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
        "input_path": input_path,
        "output_path": output_path,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
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


def _counter_rates(groups: dict[str, Counter[str]]) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            "total": counter["total"],
            "correct": counter["correct"],
            "accuracy": _safe_rate(counter["correct"], counter["total"]),
        }
        for key, counter in sorted(groups.items())
    }


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
