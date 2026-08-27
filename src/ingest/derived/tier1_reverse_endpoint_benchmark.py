"""Build graph-slice reverse-endpoint benchmark examples from Tier 1 chains.

Each example gives the model a target CVE description, a target Sigma rule, and
a shuffled local graph with distractor nodes and edges. The model must identify
the CVE node that matches the description and return the full
CVE -> CWE -> CAPEC -> ATT&CK -> Sigma chain as JSON.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any, Iterable, Sequence

import pyarrow.parquet as pq


_PRIORITY_COLUMNS = [
    "priority",
    "chain_record_id",
    "base_chain_id",
    "detection_chain_id",
    "cve_id",
    "nvd_node_id",
    "cwe_id",
    "cwe_title",
    "capec_id",
    "capec_title",
    "attack_technique_id",
    "attack_technique_title",
    "sigma_rule_id",
    "sigma_rule_title",
    "sigma_rule_level",
    "sigma_node_id",
    "path_node_ids",
    "path_relationships",
    "evidence_edge_ids",
    "content_length",
]

_EDGE_COLUMNS = [
    "edge_id",
    "source_node_id",
    "target_node_id",
    "relationship_type",
    "source_entity_type",
    "source_entity_id",
    "target_entity_type",
    "target_entity_id",
]

_NODE_COLUMNS = [
    "node_id",
    "entity_type",
    "entity_id",
    "source_id",
    "source_record_id",
    "record_id",
    "title",
    "content",
    "content_length",
    "license",
    "severity",
    "cvss_score",
    "exploited_in_wild",
    "attack_domains",
    "attack_tactics",
    "rule_level",
    "sigma_attack_tags",
]

_REQUIRED_PATH_RELATIONSHIPS = (
    "has_weakness",
    "related_attack_pattern",
    "maps_to_attack_technique",
    "detected_by_sigma_rule",
)

_CWE_TO_CAPEC_RELATIONSHIPS = {
    "related_attack_pattern",
    "related_weakness_attack_pattern",
}

_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", flags=re.IGNORECASE)
_TOKEN_ENCODER = None
_GENERATED_DISTRACTOR_META_TERMS = (
    "synthetic",
    "distractor",
    "decoy",
    "fake",
    "placeholder",
    "dummy",
)

_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["cve_id", "chain"],
    "properties": {
        "cve_id": {"type": "string"},
        "chain": {
            "type": "object",
            "additionalProperties": False,
            "required": ["sigma_rule", "attack_technique", "capec", "cwe", "cve"],
            "properties": {
                "sigma_rule": {"type": "string"},
                "attack_technique": {"type": "string"},
                "capec": {"type": "string"},
                "cwe": {"type": "string"},
                "cve": {"type": "string"},
            },
        },
    },
}

_SCORING_RUBRIC = {
    "total_points": 5,
    "partial_credit": {
        "cve_id": 1,
        "cve_to_cwe": 1,
        "cwe_to_capec": 1,
        "capec_to_attack": 1,
        "attack_to_sigma": 1,
    },
}

_RESPONSE_FORMAT_CONTRACT = """

Your response must be one JSON object and must match this exact format with no extra keys:
{
  "cve_id": "<CVE ID>",
  "chain": {
    "sigma_rule": "<sigma node_id>",
    "attack_technique": "<mitre-attack node_id>",
    "capec": "<capec node_id>",
    "cwe": "<mitre-cwe node_id>",
    "cve": "<nvd CVE node_id>"
  }
}

The chain fields must be ordered as sigma_rule, attack_technique, capec, cwe, cve.
Return JSON only.
""".strip()


@dataclass(frozen=True)
class ReverseEndpointBenchmarkSummary:
    """Summary of a graph-slice reverse-endpoint benchmark build."""

    output_path: Path
    summary_path: Path
    priority_chains_path: Path
    nodes_path: Path
    edges_path: Path
    scanned_rows: int
    unique_gold_candidates: int
    examples_written: int
    skipped_without_valid_gold_path: int
    skipped_without_unique_path: int
    skipped_without_enough_distractors: int
    skipped_by_diversity_cap: int
    seed: int
    gold_pool_multiplier: int
    min_distractor_nodes: int
    target_distractor_nodes: int
    max_distractor_bundles: int
    max_distractor_attempts: int
    max_examples_per_attack_technique: int | None
    max_examples_per_capec: int | None
    max_examples_per_cwe: int | None
    max_examples_per_cve: int | None
    unique_attack_techniques: int
    unique_capecs: int
    unique_cwes: int
    unique_cves: int
    synthetic_distractor_chains: int
    target_prompt_tokens: int | None
    target_prompt_token_tolerance: int
    synthetic_distractor_batch_size: int
    max_synthetic_distractor_chains: int | None
    prompt_token_count_min: int | None
    prompt_token_count_max: int | None
    prompt_token_count_mean: float | None
    query_mode: str
    openai_model: str | None
    openai_base_url: str | None
    openai_input_tokens: int
    openai_output_tokens: int
    openai_estimated_cost_usd: float
    openai_max_cost_usd: float | None
    hard_predictions_path: Path | None
    hard_prediction_failure_count: int
    hard_selection_candidates: int
    hard_min_selection_score: int | None
    hard_max_points: int | None
    excluded_benchmark_paths: tuple[Path, ...]
    excluded_existing_examples: int
    excluded_existing_cves: int
    skipped_by_existing_exclusion: int


@dataclass(frozen=True)
class _EdgeIndex:
    edge_by_id: dict[str, dict[str, Any]]
    outgoing: dict[str, tuple[dict[str, Any], ...]]
    incoming: dict[str, tuple[dict[str, Any], ...]]
    related_cwes: dict[str, tuple[str, ...]]
    related_capecs: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class _GraphSkeleton:
    benchmark_id: str
    gold_row: dict[str, Any]
    node_ids: tuple[str, ...]
    edges: tuple[dict[str, Any], ...]
    gold_path_edges: tuple[dict[str, Any], ...]
    distractor_types: tuple[str, ...]
    valid_paths: tuple[tuple[dict[str, Any], ...], ...]


@dataclass
class _OpenAICostTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class _SyntheticGenerationStats:
    openai_chain_count: int = 0
    local_template_chain_count: int = 0
    openai_batch_count: int = 0
    local_template_batch_count: int = 0
    local_template_fallback_error_count: int = 0
    local_template_fallback_error_types: list[str] | None = None

    def record_openai_batch(self, chain_count: int) -> None:
        self.openai_batch_count += 1
        self.openai_chain_count += chain_count

    def record_local_template_batch(
        self,
        chain_count: int,
        *,
        error: Exception | None = None,
    ) -> None:
        self.local_template_batch_count += 1
        self.local_template_chain_count += chain_count
        if error is not None:
            self.local_template_fallback_error_count += 1
            if self.local_template_fallback_error_types is None:
                self.local_template_fallback_error_types = []
            error_type = type(error).__name__
            if error_type not in self.local_template_fallback_error_types:
                self.local_template_fallback_error_types.append(error_type)


@dataclass(frozen=True)
class _HardSelectionProfile:
    predictions_path: Path
    failure_count: int
    cve_ids: frozenset[str]
    cwe_ids: frozenset[str]
    capec_ids: frozenset[str]
    attack_technique_ids: frozenset[str]
    sigma_node_ids: frozenset[str]
    min_selection_score: int
    max_points: int | None


@dataclass(frozen=True)
class _BenchmarkExclusionProfile:
    benchmark_paths: tuple[Path, ...]
    example_count: int
    benchmark_ids: frozenset[str]
    chain_record_ids: frozenset[str]
    detection_chain_ids: frozenset[str]
    cve_ids: frozenset[str]
    cve_node_ids: frozenset[str]

def build_reverse_endpoint_benchmark(
    priority_chains_path: Path,
    output_path: Path,
    *,
    nodes_path: Path | None = None,
    edges_path: Path | None = None,
    max_examples: int,
    seed: int = 13,
    batch_size: int = 10_000,
    priorities: Iterable[int] = (1, 2),
    summary_path: Path | None = None,
    gold_pool_multiplier: int = 50,
    min_distractor_nodes: int = 20,
    target_distractor_nodes: int = 40,
    max_distractor_bundles: int = 40,
    max_distractor_attempts: int = 400,
    max_examples_per_attack_technique: int | None = None,
    max_examples_per_capec: int | None = None,
    max_examples_per_cwe: int | None = None,
    max_examples_per_cve: int | None = None,
    query_mode: str = "deterministic",
    openai_model: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    openai_temperature: float | None = None,
    openai_max_completion_tokens: int = 8_000,
    synthetic_distractor_chains: int = 0,
    target_prompt_tokens: int | None = None,
    target_prompt_token_tolerance: int = 16_000,
    synthetic_distractor_batch_size: int = 20,
    max_synthetic_distractor_chains: int | None = None,
    openai_input_cost_per_million: float = 5.0,
    openai_output_cost_per_million: float = 30.0,
    openai_max_cost_usd: float | None = None,
    hard_predictions_path: Path | None = None,
    hard_min_selection_score: int = 1,
    hard_max_points: int | None = None,
    exclude_benchmark_paths: Sequence[Path] = (),
) -> ReverseEndpointBenchmarkSummary:
    """Build graph-slice Sigma-to-CVE path-finding examples."""
    priority_chains_path = Path(priority_chains_path)
    nodes_path = Path(nodes_path) if nodes_path else priority_chains_path.parent / "nodes.parquet"
    edges_path = Path(edges_path) if edges_path else priority_chains_path.parent / "edges.parquet"
    output_path = Path(output_path)
    summary_path = Path(summary_path) if summary_path else output_path.with_suffix(".summary.json")
    if max_examples <= 0:
        raise ValueError("max_examples must be positive.")
    if gold_pool_multiplier < 1:
        raise ValueError("gold_pool_multiplier must be at least 1.")
    if min_distractor_nodes < 0:
        raise ValueError("min_distractor_nodes cannot be negative.")
    if target_distractor_nodes < min_distractor_nodes:
        raise ValueError("target_distractor_nodes must be at least min_distractor_nodes.")
    if max_distractor_bundles < 0:
        raise ValueError("max_distractor_bundles cannot be negative.")
    if max_distractor_attempts < 0:
        raise ValueError("max_distractor_attempts cannot be negative.")
    for cap_name, cap_value in (
        ("max_examples_per_attack_technique", max_examples_per_attack_technique),
        ("max_examples_per_capec", max_examples_per_capec),
        ("max_examples_per_cwe", max_examples_per_cwe),
        ("max_examples_per_cve", max_examples_per_cve),
    ):
        if cap_value is not None and cap_value <= 0:
            raise ValueError(f"{cap_name} must be positive when provided.")
    if query_mode not in {"deterministic", "openai"}:
        raise ValueError("query_mode must be 'deterministic' or 'openai'.")
    if synthetic_distractor_chains < 0:
        raise ValueError("synthetic_distractor_chains cannot be negative.")
    if query_mode != "openai" and synthetic_distractor_chains:
        raise ValueError("synthetic_distractor_chains requires query_mode='openai'.")
    if synthetic_distractor_batch_size <= 0:
        raise ValueError("synthetic_distractor_batch_size must be positive.")
    if target_prompt_token_tolerance < 0:
        raise ValueError("target_prompt_token_tolerance cannot be negative.")
    if max_synthetic_distractor_chains is not None and max_synthetic_distractor_chains <= 0:
        raise ValueError("max_synthetic_distractor_chains must be positive when provided.")
    if target_prompt_tokens is not None:
        if target_prompt_tokens <= 0:
            raise ValueError("target_prompt_tokens must be positive when provided.")
        if query_mode != "openai":
            raise ValueError("target_prompt_tokens requires query_mode='openai'.")
        if max_synthetic_distractor_chains is None:
            max_synthetic_distractor_chains = max(800, synthetic_distractor_chains)
    if (
        max_synthetic_distractor_chains is not None
        and synthetic_distractor_chains > max_synthetic_distractor_chains
    ):
        raise ValueError("synthetic_distractor_chains cannot exceed max_synthetic_distractor_chains.")
    if hard_predictions_path is not None:
        hard_predictions_path = Path(hard_predictions_path)
        if hard_min_selection_score <= 0:
            raise ValueError("hard_min_selection_score must be positive.")
        if hard_max_points is not None and hard_max_points < 0:
            raise ValueError("hard_max_points cannot be negative.")
    if query_mode == "openai":
        openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        openai_base_url = (
            openai_base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
        )
        if not openai_api_key:
            raise ValueError("query_mode='openai' requires OPENAI_API_KEY or openai_api_key.")
        if not openai_model:
            raise ValueError("query_mode='openai' requires an explicit openai_model.")
        if openai_max_completion_tokens <= 0:
            raise ValueError("openai_max_completion_tokens must be positive.")
        if openai_input_cost_per_million < 0 or openai_output_cost_per_million < 0:
            raise ValueError("OpenAI token costs cannot be negative.")
        if openai_max_cost_usd is not None and openai_max_cost_usd <= 0:
            raise ValueError("openai_max_cost_usd must be positive when provided.")

    rng = random.Random(seed)
    requested_priorities = set(priorities)
    gold_pool_size = max_examples * gold_pool_multiplier
    edge_index = _load_edge_index(edges_path)
    hard_profile = (
        _load_hard_selection_profile(
            hard_predictions_path,
            min_selection_score=hard_min_selection_score,
            max_points=hard_max_points,
        )
        if hard_predictions_path is not None
        else None
    )
    exclusion_profile = _load_benchmark_exclusion_profile(exclude_benchmark_paths)

    selected_gold_rows: list[dict[str, Any]] = []
    seen_detection_ids: set[str] = set()
    scanned_rows = 0
    unique_gold_candidates = 0
    hard_selection_candidates = 0
    skipped_existing_exclusion = 0
    reservoir_seen_candidates = 0
    parquet_file = pq.ParquetFile(priority_chains_path)
    for batch in parquet_file.iter_batches(columns=_PRIORITY_COLUMNS, batch_size=batch_size):
        for row in batch.to_pylist():
            if row["priority"] not in requested_priorities or not row.get("sigma_rule_id"):
                continue
            if not _has_nvd_cve_endpoint(row):
                continue
            scanned_rows += 1
            row_id = row["detection_chain_id"] or row["chain_record_id"]
            if row_id in seen_detection_ids:
                continue
            seen_detection_ids.add(row_id)
            unique_gold_candidates += 1
            gold_row = _compact_gold_row(row)
            if exclusion_profile is not None and _excluded_by_existing_benchmark(
                gold_row,
                exclusion_profile,
            ):
                skipped_existing_exclusion += 1
                continue
            if hard_profile is not None:
                hard_score = _hard_selection_score(gold_row, hard_profile)
                if hard_score < hard_profile.min_selection_score:
                    continue
                gold_row["hard_selection_score"] = hard_score
                hard_selection_candidates += 1
            reservoir_seen_candidates += 1
            if len(selected_gold_rows) < gold_pool_size:
                selected_gold_rows.append(gold_row)
                continue
            replacement_index = rng.randrange(reservoir_seen_candidates)
            if replacement_index < gold_pool_size:
                selected_gold_rows[replacement_index] = gold_row

    if hard_profile is not None and not selected_gold_rows:
        raise ValueError(
            "The hard prediction profile did not match any gold candidates. "
            "Try lowering --hard-min-selection-score or using a different prediction file."
        )
    if hard_profile is not None:
        rng.shuffle(selected_gold_rows)
        selected_gold_rows.sort(
            key=lambda row: int(row.get("hard_selection_score") or 0),
            reverse=True,
        )

    skeletons: list[_GraphSkeleton] = []
    required_node_ids: set[str] = set()
    skipped_invalid_gold = 0
    skipped_nonunique = 0
    skipped_distractors = 0
    skipped_diversity = 0
    attack_counts: dict[str, int] = defaultdict(int)
    capec_counts: dict[str, int] = defaultdict(int)
    cwe_counts: dict[str, int] = defaultdict(int)
    cve_counts: dict[str, int] = defaultdict(int)
    for gold_row in selected_gold_rows:
        if len(skeletons) >= max_examples:
            break
        if not _within_diversity_caps(
            gold_row,
            attack_counts=attack_counts,
            capec_counts=capec_counts,
            cwe_counts=cwe_counts,
            cve_counts=cve_counts,
            max_examples_per_attack_technique=max_examples_per_attack_technique,
            max_examples_per_capec=max_examples_per_capec,
            max_examples_per_cwe=max_examples_per_cwe,
            max_examples_per_cve=max_examples_per_cve,
        ):
            skipped_diversity += 1
            continue
        skeleton, skip_reason = _make_graph_skeleton(
            gold_row,
            edge_index=edge_index,
            min_distractor_nodes=min_distractor_nodes,
            target_distractor_nodes=target_distractor_nodes,
            max_distractor_bundles=max_distractor_bundles,
            max_distractor_attempts=max_distractor_attempts,
            rng=rng,
        )
        if skeleton is None:
            if skip_reason == "invalid_gold_path":
                skipped_invalid_gold += 1
            elif skip_reason == "nonunique_path":
                skipped_nonunique += 1
            else:
                skipped_distractors += 1
            continue
        skeletons.append(skeleton)
        required_node_ids.update(skeleton.node_ids)
        _increment_diversity_counts(
            gold_row,
            attack_counts=attack_counts,
            capec_counts=capec_counts,
            cwe_counts=cwe_counts,
            cve_counts=cve_counts,
        )

    node_lookup = _load_nodes(nodes_path, required_node_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cost_tracker = _OpenAICostTracker()
    prompt_token_counts: list[int] = []
    with output_path.open("w") as f:
        for skeleton in skeletons:
            example = _materialize_example(
                skeleton,
                node_lookup=node_lookup,
                query_mode=query_mode,
                openai_model=openai_model,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
                openai_temperature=openai_temperature,
                openai_max_completion_tokens=openai_max_completion_tokens,
                synthetic_distractor_chains=synthetic_distractor_chains,
                target_prompt_tokens=target_prompt_tokens,
                target_prompt_token_tolerance=target_prompt_token_tolerance,
                synthetic_distractor_batch_size=synthetic_distractor_batch_size,
                max_synthetic_distractor_chains=max_synthetic_distractor_chains,
                openai_input_cost_per_million=openai_input_cost_per_million,
                openai_output_cost_per_million=openai_output_cost_per_million,
                openai_max_cost_usd=openai_max_cost_usd,
                cost_tracker=cost_tracker,
                rng=rng,
            )
            prompt_token_count = example["metadata"].get("model_prompt_token_count")
            if isinstance(prompt_token_count, int):
                prompt_token_counts.append(prompt_token_count)
            f.write(json.dumps(example) + "\n")

    summary = ReverseEndpointBenchmarkSummary(
        output_path=output_path,
        summary_path=summary_path,
        priority_chains_path=priority_chains_path,
        nodes_path=nodes_path,
        edges_path=edges_path,
        scanned_rows=scanned_rows,
        unique_gold_candidates=unique_gold_candidates,
        examples_written=len(skeletons),
        skipped_without_valid_gold_path=skipped_invalid_gold,
        skipped_without_unique_path=skipped_nonunique,
        skipped_without_enough_distractors=skipped_distractors,
        skipped_by_diversity_cap=skipped_diversity,
        seed=seed,
        gold_pool_multiplier=gold_pool_multiplier,
        min_distractor_nodes=min_distractor_nodes,
        target_distractor_nodes=target_distractor_nodes,
        max_distractor_bundles=max_distractor_bundles,
        max_distractor_attempts=max_distractor_attempts,
        max_examples_per_attack_technique=max_examples_per_attack_technique,
        max_examples_per_capec=max_examples_per_capec,
        max_examples_per_cwe=max_examples_per_cwe,
        max_examples_per_cve=max_examples_per_cve,
        unique_attack_techniques=len(attack_counts),
        unique_capecs=len(capec_counts),
        unique_cwes=len(cwe_counts),
        unique_cves=len(cve_counts),
        synthetic_distractor_chains=synthetic_distractor_chains,
        target_prompt_tokens=target_prompt_tokens,
        target_prompt_token_tolerance=target_prompt_token_tolerance,
        synthetic_distractor_batch_size=synthetic_distractor_batch_size,
        max_synthetic_distractor_chains=max_synthetic_distractor_chains,
        prompt_token_count_min=min(prompt_token_counts) if prompt_token_counts else None,
        prompt_token_count_max=max(prompt_token_counts) if prompt_token_counts else None,
        prompt_token_count_mean=(
            sum(prompt_token_counts) / len(prompt_token_counts)
            if prompt_token_counts
            else None
        ),
        query_mode=query_mode,
        openai_model=openai_model,
        openai_base_url=openai_base_url,
        openai_input_tokens=cost_tracker.input_tokens,
        openai_output_tokens=cost_tracker.output_tokens,
        openai_estimated_cost_usd=cost_tracker.estimated_cost_usd,
        openai_max_cost_usd=openai_max_cost_usd,
        hard_predictions_path=hard_profile.predictions_path if hard_profile else None,
        hard_prediction_failure_count=hard_profile.failure_count if hard_profile else 0,
        hard_selection_candidates=hard_selection_candidates,
        hard_min_selection_score=hard_profile.min_selection_score if hard_profile else None,
        hard_max_points=hard_profile.max_points if hard_profile else None,
        excluded_benchmark_paths=(
            exclusion_profile.benchmark_paths if exclusion_profile else ()
        ),
        excluded_existing_examples=(
            exclusion_profile.example_count if exclusion_profile else 0
        ),
        excluded_existing_cves=(
            len(exclusion_profile.cve_ids) if exclusion_profile else 0
        ),
        skipped_by_existing_exclusion=skipped_existing_exclusion,
    )
    summary_path.write_text(json.dumps(_summary_to_json(summary), indent=2) + "\n")
    return summary


def _compact_gold_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "priority": row["priority"],
        "chain_record_id": row["chain_record_id"],
        "base_chain_id": row["base_chain_id"],
        "detection_chain_id": row["detection_chain_id"],
        "cve_id": row["cve_id"],
        "nvd_node_id": row["nvd_node_id"],
        "cwe_id": row["cwe_id"],
        "cwe_title": row["cwe_title"],
        "capec_id": row["capec_id"],
        "capec_title": row["capec_title"],
        "attack_technique_id": row["attack_technique_id"],
        "attack_technique_title": row["attack_technique_title"],
        "sigma_rule_id": row["sigma_rule_id"],
        "sigma_rule_title": row["sigma_rule_title"],
        "sigma_rule_level": row["sigma_rule_level"],
        "sigma_node_id": row["sigma_node_id"],
        "path_node_ids": row["path_node_ids"],
        "path_relationships": row["path_relationships"],
        "evidence_edge_ids": row["evidence_edge_ids"],
        "source_content_length": row["content_length"],
    }


def _has_nvd_cve_endpoint(row: dict[str, Any]) -> bool:
    nvd_node_id = row.get("nvd_node_id")
    path_node_ids = row.get("path_node_ids")
    if not isinstance(nvd_node_id, str) or not nvd_node_id.startswith("nvd:CVE-"):
        return False
    if not isinstance(path_node_ids, list) or not path_node_ids:
        return False
    return path_node_ids[0] == nvd_node_id


def _within_diversity_caps(
    gold_row: dict[str, Any],
    *,
    attack_counts: dict[str, int],
    capec_counts: dict[str, int],
    cwe_counts: dict[str, int],
    cve_counts: dict[str, int],
    max_examples_per_attack_technique: int | None,
    max_examples_per_capec: int | None,
    max_examples_per_cwe: int | None,
    max_examples_per_cve: int | None,
) -> bool:
    if (
        max_examples_per_attack_technique is not None
        and attack_counts.get(gold_row["attack_technique_id"], 0) >= max_examples_per_attack_technique
    ):
        return False
    if max_examples_per_capec is not None and capec_counts.get(gold_row["capec_id"], 0) >= max_examples_per_capec:
        return False
    if max_examples_per_cwe is not None and cwe_counts.get(gold_row["cwe_id"], 0) >= max_examples_per_cwe:
        return False
    if max_examples_per_cve is not None and cve_counts.get(gold_row["cve_id"], 0) >= max_examples_per_cve:
        return False
    return True


def _increment_diversity_counts(
    gold_row: dict[str, Any],
    *,
    attack_counts: dict[str, int],
    capec_counts: dict[str, int],
    cwe_counts: dict[str, int],
    cve_counts: dict[str, int],
) -> None:
    attack_counts[gold_row["attack_technique_id"]] += 1
    capec_counts[gold_row["capec_id"]] += 1
    cwe_counts[gold_row["cwe_id"]] += 1
    cve_counts[gold_row["cve_id"]] += 1


def _load_benchmark_exclusion_profile(
    benchmark_paths: Sequence[Path],
) -> _BenchmarkExclusionProfile | None:
    paths = tuple(Path(path) for path in benchmark_paths)
    if not paths:
        return None

    benchmark_ids: set[str] = set()
    chain_record_ids: set[str] = set()
    detection_chain_ids: set[str] = set()
    cve_ids: set[str] = set()
    cve_node_ids: set[str] = set()
    example_count = 0

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Could not parse exclusion benchmark JSONL {path}:{line_number}"
                    ) from exc
                example_count += 1
                if isinstance(row.get("benchmark_id"), str):
                    benchmark_ids.add(row["benchmark_id"])
                metadata = row.get("metadata")
                if isinstance(metadata, dict):
                    _add_optional_string(metadata.get("gold_chain_record_id"), chain_record_ids)
                    _add_optional_string(metadata.get("gold_detection_chain_id"), detection_chain_ids)
                answer = row.get("answer")
                if isinstance(answer, dict):
                    cve_id = _normalize_cve_id(answer.get("cve_id"))
                    if cve_id:
                        cve_ids.add(cve_id)
                    chain = answer.get("chain")
                    if isinstance(chain, dict):
                        cve_node_id = chain.get("cve")
                        if isinstance(cve_node_id, str):
                            cve_node_ids.add(cve_node_id)
                        chain_cve_id = _normalize_cve_id(cve_node_id)
                        if chain_cve_id:
                            cve_ids.add(chain_cve_id)
    return _BenchmarkExclusionProfile(
        benchmark_paths=paths,
        example_count=example_count,
        benchmark_ids=frozenset(benchmark_ids),
        chain_record_ids=frozenset(chain_record_ids),
        detection_chain_ids=frozenset(detection_chain_ids),
        cve_ids=frozenset(cve_ids),
        cve_node_ids=frozenset(cve_node_ids),
    )


def _add_optional_string(value: Any, target: set[str]) -> None:
    if isinstance(value, str) and value:
        target.add(value)


def _excluded_by_existing_benchmark(
    gold_row: dict[str, Any],
    profile: _BenchmarkExclusionProfile,
) -> bool:
    return (
        gold_row.get("chain_record_id") in profile.chain_record_ids
        or gold_row.get("detection_chain_id") in profile.detection_chain_ids
        or gold_row.get("cve_id") in profile.cve_ids
        or gold_row.get("nvd_node_id") in profile.cve_node_ids
    )


def _load_hard_selection_profile(
    predictions_path: Path,
    *,
    min_selection_score: int,
    max_points: int | None,
) -> _HardSelectionProfile:
    cve_ids: set[str] = set()
    cwe_ids: set[str] = set()
    capec_ids: set[str] = set()
    attack_technique_ids: set[str] = set()
    sigma_node_ids: set[str] = set()
    failure_count = 0

    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            points = row.get("points")
            is_exact_failure = row.get("exact_chain_match") is False
            is_low_point_row = (
                max_points is not None
                and isinstance(points, int)
                and points <= max_points
            )
            if not is_exact_failure and not is_low_point_row:
                continue
            failure_count += 1
            _add_hard_chain_ids(row.get("gold_chain"), cve_ids, cwe_ids, capec_ids, attack_technique_ids, sigma_node_ids)
            _add_hard_chain_ids(row.get("predicted_chain"), cve_ids, cwe_ids, capec_ids, attack_technique_ids, sigma_node_ids)
            gold_cve_id = _normalize_cve_id(row.get("gold_cve_id"))
            predicted_cve_id = _normalize_cve_id(row.get("predicted_cve_id"))
            if gold_cve_id:
                cve_ids.add(gold_cve_id)
            if predicted_cve_id:
                cve_ids.add(predicted_cve_id)

    if failure_count == 0:
        raise ValueError(f"No hard rows found in prediction file: {predictions_path}")
    return _HardSelectionProfile(
        predictions_path=predictions_path,
        failure_count=failure_count,
        cve_ids=frozenset(cve_ids),
        cwe_ids=frozenset(cwe_ids),
        capec_ids=frozenset(capec_ids),
        attack_technique_ids=frozenset(attack_technique_ids),
        sigma_node_ids=frozenset(sigma_node_ids),
        min_selection_score=min_selection_score,
        max_points=max_points,
    )


def _add_hard_chain_ids(
    chain: Any,
    cve_ids: set[str],
    cwe_ids: set[str],
    capec_ids: set[str],
    attack_technique_ids: set[str],
    sigma_node_ids: set[str],
) -> None:
    if not isinstance(chain, dict):
        return
    cve_id = _normalize_cve_id(chain.get("cve"))
    cwe_id = _normalize_prefixed_id(chain.get("cwe"), prefix="mitre-cwe:")
    capec_id = _normalize_prefixed_id(chain.get("capec"), prefix="capec:")
    attack_id = _normalize_prefixed_id(chain.get("attack_technique"), prefix="mitre-attack:")
    sigma_node_id = _clean_string(chain.get("sigma_rule"))
    if cve_id:
        cve_ids.add(cve_id)
    if cwe_id:
        cwe_ids.add(cwe_id)
    if capec_id:
        capec_ids.add(capec_id)
    if attack_id:
        attack_technique_ids.add(attack_id)
    if sigma_node_id and sigma_node_id.startswith("sigma:"):
        sigma_node_ids.add(sigma_node_id)


def _hard_selection_score(
    gold_row: dict[str, Any],
    profile: _HardSelectionProfile,
) -> int:
    score = 0
    if gold_row["cve_id"] in profile.cve_ids:
        score += 16
    if gold_row["cwe_id"] in profile.cwe_ids:
        score += 8
    if gold_row["capec_id"] in profile.capec_ids:
        score += 8
    if gold_row["attack_technique_id"] in profile.attack_technique_ids:
        score += 4
    if gold_row["sigma_node_id"] in profile.sigma_node_ids:
        score += 2
    return score


def _normalize_cve_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = _CVE_RE.search(value)
    return match.group(0).upper() if match else None


def _normalize_prefixed_id(value: Any, *, prefix: str) -> str | None:
    text = _clean_string(value)
    if not text:
        return None
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _make_graph_skeleton(
    gold_row: dict[str, Any],
    *,
    edge_index: _EdgeIndex,
    min_distractor_nodes: int,
    target_distractor_nodes: int,
    max_distractor_bundles: int,
    max_distractor_attempts: int,
    rng: random.Random,
) -> tuple[_GraphSkeleton | None, str | None]:
    gold_path_edges = _resolve_gold_path_edges(gold_row, edge_index)
    if gold_path_edges is None:
        return None, "invalid_gold_path"

    selected_edges = {edge["edge_id"]: edge for edge in gold_path_edges}
    distractor_types: list[str] = []
    valid_paths = _find_valid_paths(selected_edges.values(), gold_row["sigma_node_id"])
    if not _has_exactly_one_gold_path(valid_paths, gold_row):
        return None, "nonunique_path"

    bundles = _candidate_distractor_bundles(gold_row, edge_index, rng)
    added_bundles = 0
    attempts = 0
    for distractor_type, bundle in bundles:
        if added_bundles >= max_distractor_bundles or attempts >= max_distractor_attempts:
            break
        attempts += 1
        if not bundle:
            continue
        trial_edges = dict(selected_edges)
        for edge in bundle:
            trial_edges[edge["edge_id"]] = edge
        trial_paths = _find_valid_paths(trial_edges.values(), gold_row["sigma_node_id"])
        if not _has_exactly_one_gold_path(trial_paths, gold_row):
            continue
        selected_edges = trial_edges
        distractor_types.append(distractor_type)
        added_bundles += 1
        if _distractor_node_count(selected_edges.values(), gold_row["path_node_ids"]) >= target_distractor_nodes:
            break

    valid_paths = _find_valid_paths(selected_edges.values(), gold_row["sigma_node_id"])
    if not _has_exactly_one_gold_path(valid_paths, gold_row):
        return None, "nonunique_path"
    if _distractor_node_count(selected_edges.values(), gold_row["path_node_ids"]) < min_distractor_nodes:
        return None, "not_enough_distractors"

    node_ids = _node_ids_from_edges(selected_edges.values())
    benchmark_id = _stable_id(
        "tier1-graph-reverse-endpoint",
        gold_row["detection_chain_id"],
        *(edge_id for edge_id in sorted(selected_edges)),
    )
    return (
        _GraphSkeleton(
            benchmark_id=benchmark_id,
            gold_row=gold_row,
            node_ids=tuple(sorted(node_ids)),
            edges=tuple(selected_edges.values()),
            gold_path_edges=gold_path_edges,
            distractor_types=tuple(distractor_types),
            valid_paths=tuple(tuple(path) for path in valid_paths),
        ),
        None,
    )


def _resolve_gold_path_edges(
    gold_row: dict[str, Any],
    edge_index: _EdgeIndex,
) -> tuple[dict[str, Any], ...] | None:
    path_nodes = gold_row["path_node_ids"]
    relationships = gold_row["path_relationships"]
    if len(path_nodes) != 5 or tuple(relationships) != _REQUIRED_PATH_RELATIONSHIPS:
        return None

    edges: list[dict[str, Any]] = []
    evidence_edges = [
        edge_index.edge_by_id[edge_id]
        for edge_id in gold_row["evidence_edge_ids"]
        if edge_id in edge_index.edge_by_id
    ]
    for source_node, target_node, relationship in zip(path_nodes, path_nodes[1:], relationships):
        match = _first_matching_edge(evidence_edges, source_node, target_node, relationship)
        if match is None:
            match = _first_matching_edge(
                edge_index.outgoing.get(source_node, ()),
                source_node,
                target_node,
                relationship,
            )
        if match is None:
            return None
        edges.append(match)
    return tuple(edges)


def _first_matching_edge(
    edges: Iterable[dict[str, Any]],
    source_node: str,
    target_node: str,
    relationship: str,
) -> dict[str, Any] | None:
    for edge in edges:
        if (
            edge["source_node_id"] == source_node
            and edge["target_node_id"] == target_node
            and edge["relationship_type"] == relationship
        ):
            return edge
    return None


def _candidate_distractor_bundles(
    gold_row: dict[str, Any],
    edge_index: _EdgeIndex,
    rng: random.Random,
) -> list[tuple[str, tuple[dict[str, Any], ...]]]:
    bundles: list[tuple[str, tuple[dict[str, Any], ...]]] = []
    gold_cwe_node = f"mitre-cwe:{gold_row['cwe_id']}"
    gold_capec_node = f"capec:{gold_row['capec_id']}"
    gold_attack_node = f"mitre-attack:{gold_row['attack_technique_id']}"
    target_sigma_node = gold_row["sigma_node_id"]
    avoid_nodes = set(gold_row["path_node_ids"])

    for edge in _sample_edges(
        [
            edge
            for edge in edge_index.outgoing.get(gold_attack_node, ())
            if edge["relationship_type"] == "detected_by_sigma_rule"
            and edge["target_node_id"] != target_sigma_node
        ],
        limit=3,
        rng=rng,
    ):
        bundles.append(("same_attack_wrong_sigma", (edge,)))

    for edge in _sample_edges(
        [
            edge
            for edge in edge_index.incoming.get(target_sigma_node, ())
            if edge["relationship_type"] == "detected_by_sigma_rule"
            and edge["source_node_id"] != gold_attack_node
        ],
        limit=8,
        rng=rng,
    ):
        bundle = _sample_complete_chain_to_attack(
            edge["source_node_id"],
            sigma_edge=edge,
            edge_index=edge_index,
            avoid_nodes=avoid_nodes,
            rng=rng,
        )
        if bundle:
            bundles.append(("same_sigma_wrong_complete_chain", bundle))

    for bundle in _sample_complete_chains_from_attack(
        gold_attack_node,
        edge_index=edge_index,
        target_sigma_node=target_sigma_node,
        avoid_nodes=avoid_nodes,
        limit=8,
        rng=rng,
    ):
        bundles.append(("same_attack_wrong_complete_chain", bundle))

    for cwe_node in _sample_values(edge_index.related_cwes.get(gold_cwe_node, ()), limit=20, rng=rng):
        bundle = _sample_chain_from_cwe(
            cwe_node,
            edge_index=edge_index,
            target_sigma_node=target_sigma_node,
            avoid_nodes=avoid_nodes,
            rng=rng,
        )
        if bundle:
            bundles.append(("related_cwe_wrong_path", bundle))

    for capec_node in _sample_values(edge_index.related_capecs.get(gold_capec_node, ()), limit=20, rng=rng):
        bundle = _sample_chain_from_capec(
            capec_node,
            edge_index=edge_index,
            target_sigma_node=target_sigma_node,
            avoid_nodes=avoid_nodes,
            rng=rng,
        )
        if bundle:
            bundles.append(("related_capec_wrong_path", bundle))

    for edge in _sample_edges(
        [
            edge
            for edge in edge_index.outgoing.get(gold_capec_node, ())
            if edge["relationship_type"] == "maps_to_attack_technique"
            and edge["target_node_id"] != gold_attack_node
        ],
        limit=5,
        rng=rng,
    ):
        bundle = [edge]
        wrong_sigma = _sample_detected_by_sigma_edge(
            edge["target_node_id"],
            edge_index=edge_index,
            target_sigma_node=target_sigma_node,
            rng=rng,
        )
        if wrong_sigma:
            bundle.append(wrong_sigma)
        bundles.append(("same_capec_wrong_attack", tuple(bundle)))

    rng.shuffle(bundles)
    return bundles


def _sample_chain_from_cwe(
    cwe_node: str,
    *,
    edge_index: _EdgeIndex,
    target_sigma_node: str,
    avoid_nodes: set[str],
    rng: random.Random,
) -> tuple[dict[str, Any], ...]:
    cve_edge = _sample_incoming_edge(
        cwe_node,
        edge_index=edge_index,
        relationship="has_weakness",
        source_type="cve",
        avoid_nodes=avoid_nodes,
        rng=rng,
    )
    capec_edge = _sample_outgoing_edge(
        cwe_node,
        edge_index=edge_index,
        relationships=_CWE_TO_CAPEC_RELATIONSHIPS,
        avoid_nodes=avoid_nodes,
        rng=rng,
    )
    if cve_edge is None or capec_edge is None:
        return ()
    bundle = [cve_edge, capec_edge]
    attack_edge = _sample_outgoing_edge(
        capec_edge["target_node_id"],
        edge_index=edge_index,
        relationships={"maps_to_attack_technique"},
        avoid_nodes=avoid_nodes,
        rng=rng,
    )
    if attack_edge is None:
        return tuple(bundle)
    bundle.append(attack_edge)
    sigma_edge = _sample_detected_by_sigma_edge(
        attack_edge["target_node_id"],
        edge_index=edge_index,
        target_sigma_node=target_sigma_node,
        rng=rng,
    )
    if sigma_edge is not None:
        bundle.append(sigma_edge)
    return tuple(bundle)


def _sample_complete_chains_from_attack(
    attack_node: str,
    *,
    edge_index: _EdgeIndex,
    target_sigma_node: str,
    avoid_nodes: set[str],
    limit: int,
    rng: random.Random,
) -> list[tuple[dict[str, Any], ...]]:
    bundles: list[tuple[dict[str, Any], ...]] = []
    sigma_edges = _sample_edges(
        [
            edge
            for edge in edge_index.outgoing.get(attack_node, ())
            if edge["relationship_type"] == "detected_by_sigma_rule"
            and edge["target_node_id"] != target_sigma_node
        ],
        limit=limit,
        rng=rng,
    )
    for sigma_edge in sigma_edges:
        bundle = _sample_complete_chain_to_attack(
            attack_node,
            sigma_edge=sigma_edge,
            edge_index=edge_index,
            avoid_nodes=avoid_nodes,
            rng=rng,
        )
        if bundle:
            bundles.append(bundle)
    return bundles


def _sample_complete_chain_to_attack(
    attack_node: str,
    *,
    sigma_edge: dict[str, Any],
    edge_index: _EdgeIndex,
    avoid_nodes: set[str],
    rng: random.Random,
) -> tuple[dict[str, Any], ...]:
    capec_edge = _sample_incoming_edge(
        attack_node,
        edge_index=edge_index,
        relationship="maps_to_attack_technique",
        source_type="capec",
        avoid_nodes=avoid_nodes,
        rng=rng,
    )
    if capec_edge is None:
        return ()
    cwe_edge = _sample_incoming_edge(
        capec_edge["source_node_id"],
        edge_index=edge_index,
        relationship="related_attack_pattern",
        source_type="cwe",
        avoid_nodes=avoid_nodes,
        rng=rng,
    )
    if cwe_edge is None:
        cwe_edge = _sample_incoming_edge(
            capec_edge["source_node_id"],
            edge_index=edge_index,
            relationship="related_weakness_attack_pattern",
            source_type="cwe",
            avoid_nodes=avoid_nodes,
            rng=rng,
        )
    if cwe_edge is None:
        return ()
    cve_edge = _sample_incoming_edge(
        cwe_edge["source_node_id"],
        edge_index=edge_index,
        relationship="has_weakness",
        source_type="cve",
        avoid_nodes=avoid_nodes,
        rng=rng,
    )
    if cve_edge is None:
        return ()
    return (cve_edge, cwe_edge, capec_edge, sigma_edge)


def _sample_chain_from_capec(
    capec_node: str,
    *,
    edge_index: _EdgeIndex,
    target_sigma_node: str,
    avoid_nodes: set[str],
    rng: random.Random,
) -> tuple[dict[str, Any], ...]:
    cwe_edge = _sample_incoming_edge(
        capec_node,
        edge_index=edge_index,
        relationship="related_attack_pattern",
        source_type="cwe",
        avoid_nodes=avoid_nodes,
        rng=rng,
    )
    if cwe_edge is None:
        cwe_edge = _sample_incoming_edge(
            capec_node,
            edge_index=edge_index,
            relationship="related_weakness_attack_pattern",
            source_type="cwe",
            avoid_nodes=avoid_nodes,
            rng=rng,
        )
    attack_edge = _sample_outgoing_edge(
        capec_node,
        edge_index=edge_index,
        relationships={"maps_to_attack_technique"},
        avoid_nodes=avoid_nodes,
        rng=rng,
    )
    if cwe_edge is None or attack_edge is None:
        return ()
    bundle = [cwe_edge, attack_edge]
    cve_edge = _sample_incoming_edge(
        cwe_edge["source_node_id"],
        edge_index=edge_index,
        relationship="has_weakness",
        source_type="cve",
        avoid_nodes=avoid_nodes,
        rng=rng,
    )
    if cve_edge is not None:
        bundle.insert(0, cve_edge)
    sigma_edge = _sample_detected_by_sigma_edge(
        attack_edge["target_node_id"],
        edge_index=edge_index,
        target_sigma_node=target_sigma_node,
        rng=rng,
    )
    if sigma_edge is not None:
        bundle.append(sigma_edge)
    return tuple(bundle)


def _sample_detected_by_sigma_edge(
    attack_node: str,
    *,
    edge_index: _EdgeIndex,
    target_sigma_node: str,
    rng: random.Random,
) -> dict[str, Any] | None:
    return _sample_outgoing_edge(
        attack_node,
        edge_index=edge_index,
        relationships={"detected_by_sigma_rule"},
        avoid_nodes={target_sigma_node},
        rng=rng,
    )


def _sample_incoming_edge(
    node_id: str,
    *,
    edge_index: _EdgeIndex,
    relationship: str,
    source_type: str,
    avoid_nodes: set[str],
    rng: random.Random,
) -> dict[str, Any] | None:
    candidates = [
        edge
        for edge in edge_index.incoming.get(node_id, ())
        if edge["relationship_type"] == relationship
        and edge["source_entity_type"] == source_type
        and edge["source_node_id"] not in avoid_nodes
    ]
    if not candidates:
        return None
    return candidates[rng.randrange(len(candidates))]


def _sample_outgoing_edge(
    node_id: str,
    *,
    edge_index: _EdgeIndex,
    relationships: set[str],
    avoid_nodes: set[str],
    rng: random.Random,
) -> dict[str, Any] | None:
    candidates = [
        edge
        for edge in edge_index.outgoing.get(node_id, ())
        if edge["relationship_type"] in relationships and edge["target_node_id"] not in avoid_nodes
    ]
    if not candidates:
        return None
    return candidates[rng.randrange(len(candidates))]


def _sample_edges(
    edges: list[dict[str, Any]],
    *,
    limit: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if len(edges) <= limit:
        rng.shuffle(edges)
        return edges
    return rng.sample(edges, limit)


def _sample_values(values: tuple[str, ...], *, limit: int, rng: random.Random) -> tuple[str, ...]:
    if len(values) <= limit:
        values = tuple(values)
        shuffled = list(values)
        rng.shuffle(shuffled)
        return tuple(shuffled)
    return tuple(rng.sample(values, limit))


def _find_valid_paths(
    edges: Iterable[dict[str, Any]],
    target_sigma_node: str,
) -> list[tuple[dict[str, Any], ...]]:
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        outgoing[edge["source_node_id"]].append(edge)

    paths: list[tuple[dict[str, Any], ...]] = []
    for first_edge_candidates in outgoing.values():
        for first_edge in first_edge_candidates:
            if first_edge["relationship_type"] != _REQUIRED_PATH_RELATIONSHIPS[0]:
                continue
            if first_edge["source_entity_type"] != "cve" or first_edge["target_entity_type"] != "cwe":
                continue
            _extend_path(
                (first_edge,),
                relationship_index=1,
                outgoing=outgoing,
                target_sigma_node=target_sigma_node,
                paths=paths,
            )
    return paths


def _extend_path(
    path: tuple[dict[str, Any], ...],
    *,
    relationship_index: int,
    outgoing: dict[str, list[dict[str, Any]]],
    target_sigma_node: str,
    paths: list[tuple[dict[str, Any], ...]],
) -> None:
    current_node = path[-1]["target_node_id"]
    if relationship_index >= len(_REQUIRED_PATH_RELATIONSHIPS):
        if current_node == target_sigma_node:
            paths.append(path)
        return
    required_relationship = _REQUIRED_PATH_RELATIONSHIPS[relationship_index]
    for edge in outgoing.get(current_node, ()):
        if edge["relationship_type"] != required_relationship:
            continue
        _extend_path(
            (*path, edge),
            relationship_index=relationship_index + 1,
            outgoing=outgoing,
            target_sigma_node=target_sigma_node,
            paths=paths,
        )


def _has_exactly_one_gold_path(paths: list[tuple[dict[str, Any], ...]], gold_row: dict[str, Any]) -> bool:
    if len(paths) != 1:
        return False
    path = paths[0]
    path_nodes = [path[0]["source_node_id"], *(edge["target_node_id"] for edge in path)]
    relationships = [edge["relationship_type"] for edge in path]
    return path_nodes == gold_row["path_node_ids"] and relationships == gold_row["path_relationships"]


def _distractor_node_count(edges: Iterable[dict[str, Any]], gold_path_node_ids: list[str]) -> int:
    return len(_node_ids_from_edges(edges) - set(gold_path_node_ids))


def _node_ids_from_edges(edges: Iterable[dict[str, Any]]) -> set[str]:
    node_ids: set[str] = set()
    for edge in edges:
        node_ids.add(edge["source_node_id"])
        node_ids.add(edge["target_node_id"])
    return node_ids


def _materialize_example(
    skeleton: _GraphSkeleton,
    *,
    node_lookup: dict[str, dict[str, Any]],
    query_mode: str,
    openai_model: str | None,
    openai_api_key: str | None,
    openai_base_url: str | None,
    openai_temperature: float | None,
    openai_max_completion_tokens: int,
    synthetic_distractor_chains: int,
    target_prompt_tokens: int | None,
    target_prompt_token_tolerance: int,
    synthetic_distractor_batch_size: int,
    max_synthetic_distractor_chains: int | None,
    openai_input_cost_per_million: float,
    openai_output_cost_per_million: float,
    openai_max_cost_usd: float | None,
    cost_tracker: _OpenAICostTracker,
    rng: random.Random,
) -> dict[str, Any]:
    gold_row = skeleton.gold_row
    target_cve_node = node_lookup[gold_row["nvd_node_id"]]
    target_sigma_node = node_lookup[gold_row["sigma_node_id"]]
    source_target_cve_description = _target_cve_description(target_cve_node, gold_row["cve_id"])
    target_cve_description = source_target_cve_description
    target_sigma_rule = _target_sigma_rule(target_sigma_node)
    gold_chain_context = _gold_chain_context(gold_row, node_lookup)

    nodes = [_node_for_prompt(node_lookup[node_id]) for node_id in skeleton.node_ids if node_id in node_lookup]
    edges = [_edge_for_prompt(edge) for edge in skeleton.edges]
    synthetic_nodes: list[dict[str, Any]] = []
    synthetic_edges: list[dict[str, Any]] = []
    synthetic_chain_count = 0
    synthetic_generation_stats = _SyntheticGenerationStats()
    prompt_token_count: int | None = None

    if query_mode == "openai":
        initial_synthetic_distractor_chains = 0 if target_prompt_tokens is not None else synthetic_distractor_chains
        generated_task_text = _generate_openai_task_text(
            source_target_cve_description=source_target_cve_description,
            target_sigma_rule=target_sigma_rule,
            synthetic_distractor_chains=initial_synthetic_distractor_chains,
            openai_model=openai_model,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            temperature=openai_temperature,
            max_completion_tokens=openai_max_completion_tokens,
            input_cost_per_million=openai_input_cost_per_million,
            output_cost_per_million=openai_output_cost_per_million,
            max_cost_usd=openai_max_cost_usd,
            cost_tracker=cost_tracker,
        )
        target_cve_description = generated_task_text["target_cve_description"]
        question = generated_task_text["question"]
        if initial_synthetic_distractor_chains:
            new_nodes, new_edges, new_chain_count = _synthetic_distractor_graph(
                benchmark_id=skeleton.benchmark_id,
                distractors=generated_task_text.get("synthetic_distractors", ()),
                avoid_node_ids=_prompt_node_ids(nodes),
                chain_index_offset=synthetic_chain_count,
            )
            nodes.extend(new_nodes)
            edges.extend(new_edges)
            synthetic_nodes.extend(new_nodes)
            synthetic_edges.extend(new_edges)
            synthetic_chain_count += new_chain_count
            synthetic_generation_stats.record_openai_batch(new_chain_count)
        if target_prompt_tokens is not None:
            prompt_token_count, synthetic_chain_count = _expand_synthetic_distractors_to_prompt_target(
                benchmark_id=skeleton.benchmark_id,
                question=question,
                target_cve_description=target_cve_description,
                target_sigma_rule=target_sigma_rule,
                gold_chain_context=gold_chain_context,
                nodes=nodes,
                edges=edges,
                synthetic_nodes=synthetic_nodes,
                synthetic_edges=synthetic_edges,
                synthetic_chain_count=synthetic_chain_count,
                min_synthetic_distractor_chains=synthetic_distractor_chains,
                target_prompt_tokens=target_prompt_tokens,
                target_prompt_token_tolerance=target_prompt_token_tolerance,
                synthetic_distractor_batch_size=synthetic_distractor_batch_size,
                max_synthetic_distractor_chains=max_synthetic_distractor_chains,
                openai_model=openai_model,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
                temperature=openai_temperature,
                max_completion_tokens=openai_max_completion_tokens,
                input_cost_per_million=openai_input_cost_per_million,
                output_cost_per_million=openai_output_cost_per_million,
                max_cost_usd=openai_max_cost_usd,
                cost_tracker=cost_tracker,
                generation_stats=synthetic_generation_stats,
            )
    else:
        question = _format_question(target_cve_description, target_sigma_rule)

    rng.shuffle(nodes)
    rng.shuffle(edges)
    if prompt_token_count is None:
        prompt_token_count = _count_model_prompt_tokens(
            question=question,
            target_cve_description=target_cve_description,
            target_sigma_rule=target_sigma_rule,
            nodes=nodes,
            edges=edges,
        )
    gold_path_edges = [_edge_for_prompt(edge) for edge in skeleton.gold_path_edges]
    real_distractor_node_count = _distractor_node_count(skeleton.edges, gold_row["path_node_ids"])
    real_distractor_edge_count = len(skeleton.edges) - len(skeleton.gold_path_edges)
    target_prompt_token_min = (
        max(0, target_prompt_tokens - target_prompt_token_tolerance)
        if target_prompt_tokens is not None
        else None
    )
    target_prompt_token_max = (
        target_prompt_tokens + target_prompt_token_tolerance
        if target_prompt_tokens is not None
        else None
    )
    return {
        "benchmark_id": skeleton.benchmark_id,
        "task_type": "tier1_sigma_to_cve_path_json",
        "version": "tier1-reverse-endpoint-graph-v1",
        "question": question,
        "input": {
            "target_cve_description": target_cve_description,
            "target_sigma_rule": target_sigma_rule,
            "nodes": nodes,
            "edges": edges,
        },
        "output_json_schema": _JSON_SCHEMA,
        "scoring": _SCORING_RUBRIC,
        "answer": {
            "cve_id": gold_row["cve_id"],
            "path_node_ids": gold_row["path_node_ids"],
            "reverse_path_node_ids": list(reversed(gold_row["path_node_ids"])),
            "path_relationships": gold_row["path_relationships"],
            "reverse_path_relationships": list(reversed(gold_row["path_relationships"])),
            "path_edge_ids": [edge["edge_id"] for edge in skeleton.gold_path_edges],
            "reverse_path_edge_ids": [edge["edge_id"] for edge in reversed(skeleton.gold_path_edges)],
            "chain": {
                "sigma_rule": gold_row["sigma_node_id"],
                "attack_technique": f"mitre-attack:{gold_row['attack_technique_id']}",
                "capec": f"capec:{gold_row['capec_id']}",
                "cwe": f"mitre-cwe:{gold_row['cwe_id']}",
                "cve": gold_row["nvd_node_id"],
            },
        },
        "metadata": {
            "gold_priority": gold_row["priority"],
            "gold_chain_record_id": gold_row["chain_record_id"],
            "gold_detection_chain_id": gold_row["detection_chain_id"],
            "source_content_length": gold_row["source_content_length"],
            "query_mode": query_mode,
            "hard_selection_score": gold_row.get("hard_selection_score"),
            "distractor_types": sorted(set(skeleton.distractor_types)),
            "synthetic_distractor_chains_requested": synthetic_distractor_chains,
            "synthetic_distractor_chain_count": synthetic_chain_count,
            "synthetic_distractor_openai_chain_count": synthetic_generation_stats.openai_chain_count,
            "synthetic_distractor_local_template_chain_count": (
                synthetic_generation_stats.local_template_chain_count
            ),
            "synthetic_distractor_openai_batch_count": synthetic_generation_stats.openai_batch_count,
            "synthetic_distractor_local_template_batch_count": (
                synthetic_generation_stats.local_template_batch_count
            ),
            "synthetic_distractor_local_template_fallback_error_count": (
                synthetic_generation_stats.local_template_fallback_error_count
            ),
            "synthetic_distractor_local_template_fallback_error_types": (
                sorted(synthetic_generation_stats.local_template_fallback_error_types or ())
            ),
            "synthetic_distractor_batch_size": synthetic_distractor_batch_size,
            "max_synthetic_distractor_chains": max_synthetic_distractor_chains,
            "target_prompt_tokens": target_prompt_tokens,
            "target_prompt_token_tolerance": target_prompt_token_tolerance,
            "target_prompt_token_min": target_prompt_token_min,
            "target_prompt_token_max": target_prompt_token_max,
            "model_prompt_token_count": prompt_token_count,
            "model_prompt_token_target_met": (
                True
                if target_prompt_tokens is None
                else (
                    target_prompt_token_min <= prompt_token_count <= target_prompt_token_max
                    if target_prompt_token_min is not None and target_prompt_token_max is not None
                    else False
                )
            ),
            "source_target_cve_description": source_target_cve_description,
        },
        "verification": {
            "gold_hops_verified": True,
            "exactly_one_valid_path_to_target_sigma": True,
            "valid_path_count_to_target_sigma": len(skeleton.valid_paths),
            "distractor_node_count": real_distractor_node_count + len(synthetic_nodes),
            "distractor_edge_count": real_distractor_edge_count + len(synthetic_edges),
            "real_distractor_node_count": real_distractor_node_count,
            "real_distractor_edge_count": real_distractor_edge_count,
            "synthetic_distractor_node_count": len(synthetic_nodes),
            "synthetic_distractor_edge_count": len(synthetic_edges),
        },
    }


def _target_cve_description(cve_node: dict[str, Any], cve_id: str) -> str:
    content = cve_node.get("content") or ""
    description = _redact_cve_ids(_snippet(content, max_chars=900), cve_id)
    return description or "Target vulnerability description unavailable."


def _target_sigma_rule(sigma_node: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": sigma_node["node_id"],
        "rule_id": sigma_node["entity_id"],
        "title": sigma_node.get("title"),
        "level": sigma_node.get("rule_level"),
        "description": _snippet(_strip_sigma_yaml(sigma_node.get("content") or ""), max_chars=500),
    }


def _gold_chain_context(
    gold_row: dict[str, Any],
    node_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return hidden gold concepts used only to generate harder false near-misses."""
    fields = [
        ("cve", gold_row["nvd_node_id"]),
        ("cwe", f"mitre-cwe:{gold_row['cwe_id']}"),
        ("capec", f"capec:{gold_row['capec_id']}"),
        ("attack_technique", f"mitre-attack:{gold_row['attack_technique_id']}"),
        ("sigma_rule", gold_row["sigma_node_id"]),
    ]
    context: dict[str, Any] = {}
    for key, node_id in fields:
        node = node_lookup.get(node_id, {})
        content = node.get("content") or ""
        if node.get("entity_type") == "sigma-rule":
            content = _strip_sigma_yaml(content)
        context[key] = {
            "id": node.get("entity_id") or node_id.split(":", 1)[-1],
            "title": node.get("title"),
            "description": _snippet(content, max_chars=500),
        }
    return context


def _node_for_prompt(node: dict[str, Any]) -> dict[str, Any]:
    content = node.get("content") or ""
    if node["entity_type"] == "sigma-rule":
        content = _strip_sigma_yaml(content)
    prompt_node = {
        "node_id": node["node_id"],
        "type": node["entity_type"],
        "id": node["entity_id"],
        "title": node.get("title"),
        "description": _snippet(content, max_chars=500),
    }
    metadata = {}
    for field in ("severity", "cvss_score", "exploited_in_wild", "attack_tactics", "rule_level"):
        value = node.get(field)
        if value not in (None, [], ""):
            metadata[field] = value
    if metadata:
        prompt_node["metadata"] = metadata
    return prompt_node


def _edge_for_prompt(edge: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": edge["source_node_id"],
        "relationship": edge["relationship_type"],
        "target": edge["target_node_id"],
    }


def assemble_reverse_endpoint_model_prompt(example: dict[str, Any]) -> str:
    """Return the downstream model-facing prompt for one benchmark example."""
    return _format_model_prompt(
        question=example["question"],
        target_cve_description=example["input"]["target_cve_description"],
        target_sigma_rule=example["input"]["target_sigma_rule"],
        nodes=example["input"]["nodes"],
        edges=example["input"]["edges"],
        output_json_schema=example.get("output_json_schema", _JSON_SCHEMA),
    )


def _format_model_prompt(
    *,
    question: str,
    target_cve_description: str,
    target_sigma_rule: dict[str, Any],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    output_json_schema: dict[str, Any],
) -> str:
    model_input = {
        "target_cve_description": target_cve_description,
        "target_sigma_rule": target_sigma_rule,
        "nodes": nodes,
        "edges": edges,
    }
    return (
        f"{question.strip()}\n\n"
        "Provided graph input:\n"
        f"{json.dumps(model_input, ensure_ascii=False, sort_keys=True)}\n\n"
        "Required output JSON schema:\n"
        f"{json.dumps(output_json_schema, ensure_ascii=False, sort_keys=True)}"
    )


def _count_model_prompt_tokens(
    *,
    question: str,
    target_cve_description: str,
    target_sigma_rule: dict[str, Any],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[int, int]:
    return _count_tokens(
        _format_model_prompt(
            question=question,
            target_cve_description=target_cve_description,
            target_sigma_rule=target_sigma_rule,
            nodes=nodes,
            edges=edges,
            output_json_schema=_JSON_SCHEMA,
        )
    )


def _count_tokens(text: str) -> int:
    global _TOKEN_ENCODER
    if _TOKEN_ENCODER is None:
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError("tiktoken is required to count benchmark prompt tokens.") from exc
        _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    return len(_TOKEN_ENCODER.encode(text))


def _expand_synthetic_distractors_to_prompt_target(
    *,
    benchmark_id: str,
    question: str,
    target_cve_description: str,
    target_sigma_rule: dict[str, Any],
    gold_chain_context: dict[str, Any] | None,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    synthetic_nodes: list[dict[str, Any]],
    synthetic_edges: list[dict[str, Any]],
    synthetic_chain_count: int,
    min_synthetic_distractor_chains: int,
    target_prompt_tokens: int,
    target_prompt_token_tolerance: int,
    synthetic_distractor_batch_size: int,
    max_synthetic_distractor_chains: int | None,
    openai_model: str | None,
    openai_api_key: str | None,
    openai_base_url: str | None,
    temperature: float | None,
    max_completion_tokens: int,
    input_cost_per_million: float,
    output_cost_per_million: float,
    max_cost_usd: float | None,
    cost_tracker: _OpenAICostTracker,
    generation_stats: _SyntheticGenerationStats,
) -> tuple[int, int]:
    target_min = max(0, target_prompt_tokens - target_prompt_token_tolerance)
    prompt_token_count = _count_model_prompt_tokens(
        question=question,
        target_cve_description=target_cve_description,
        target_sigma_rule=target_sigma_rule,
        nodes=nodes,
        edges=edges,
    )
    baseline_prompt_token_count = prompt_token_count
    batch_index = 0

    while prompt_token_count < target_min or synthetic_chain_count < min_synthetic_distractor_chains:
        if (
            max_synthetic_distractor_chains is not None
            and synthetic_chain_count >= max_synthetic_distractor_chains
        ):
            raise RuntimeError(
                "Could not reach target prompt length before hitting "
                f"max_synthetic_distractor_chains={max_synthetic_distractor_chains}. "
                f"Current prompt has {prompt_token_count:,} tokens; target minimum is "
                f"{target_min:,}."
            )

        remaining_min_chains = max(0, min_synthetic_distractor_chains - synthetic_chain_count)
        remaining_tokens = max(0, target_min - prompt_token_count)
        estimated_chains_for_tokens = synthetic_distractor_batch_size
        if synthetic_chain_count > 0 and prompt_token_count > baseline_prompt_token_count:
            tokens_per_chain = (
                (prompt_token_count - baseline_prompt_token_count)
                / synthetic_chain_count
            )
            if tokens_per_chain > 0:
                estimated_chains_for_tokens = max(1, math.ceil(remaining_tokens / tokens_per_chain))

        batch_count = min(
            synthetic_distractor_batch_size,
            max(remaining_min_chains, estimated_chains_for_tokens),
        )
        batch_count = max(1, batch_count)
        if max_synthetic_distractor_chains is not None:
            batch_count = min(batch_count, max_synthetic_distractor_chains - synthetic_chain_count)
        if batch_count <= 0:
            raise RuntimeError("Synthetic distractor expansion could not request another batch.")

        batch_index += 1
        generated_by_openai = False
        fallback_error: Exception | None = None
        try:
            distractors = _generate_openai_synthetic_distractors(
                target_cve_description=target_cve_description,
                target_sigma_rule=target_sigma_rule,
                gold_chain_context=gold_chain_context,
                synthetic_distractor_chains=batch_count,
                batch_index=batch_index,
                openai_model=openai_model,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                input_cost_per_million=input_cost_per_million,
                output_cost_per_million=output_cost_per_million,
                max_cost_usd=max_cost_usd,
                cost_tracker=cost_tracker,
            )
            generated_by_openai = True
        except Exception as exc:
            fallback_error = exc
            distractors = _local_synthetic_distractors(
                benchmark_id=benchmark_id,
                target_cve_description=target_cve_description,
                target_sigma_rule=target_sigma_rule,
                count=batch_count,
                batch_index=batch_index,
            )
        new_nodes, new_edges, new_chain_count = _synthetic_distractor_graph(
            benchmark_id=benchmark_id,
            distractors=distractors,
            avoid_node_ids=_prompt_node_ids(nodes),
            chain_index_offset=synthetic_chain_count,
        )
        if new_chain_count == 0:
            raise RuntimeError("OpenAI synthetic distractor batch produced no usable graph nodes.")
        nodes.extend(new_nodes)
        edges.extend(new_edges)
        synthetic_nodes.extend(new_nodes)
        synthetic_edges.extend(new_edges)
        synthetic_chain_count += new_chain_count
        if generated_by_openai:
            generation_stats.record_openai_batch(new_chain_count)
        else:
            generation_stats.record_local_template_batch(new_chain_count, error=fallback_error)
        prompt_token_count = _count_model_prompt_tokens(
            question=question,
            target_cve_description=target_cve_description,
            target_sigma_rule=target_sigma_rule,
            nodes=nodes,
            edges=edges,
        )

    return prompt_token_count, synthetic_chain_count


def _prompt_node_ids(nodes: Iterable[dict[str, Any]]) -> set[str]:
    return {node["node_id"] for node in nodes if isinstance(node.get("node_id"), str)}


def _local_synthetic_distractors(
    *,
    benchmark_id: str,
    target_cve_description: str,
    target_sigma_rule: dict[str, Any],
    count: int,
    batch_index: int,
) -> list[dict[str, Any]]:
    topics = [
        (
            "credential validation workflow",
            "improper rate limiting for authentication attempts",
            "credential guessing through repeated login requests",
            "valid account use after weak challenge handling",
            "multiple failed sign-in attempts followed by an interactive session",
        ),
        (
            "configuration parsing component",
            "improper handling of nested configuration values",
            "malformed configuration field processing",
            "configuration-driven execution path selection",
            "unexpected configuration utility launch after file parsing",
        ),
        (
            "file metadata processor",
            "insufficient validation of structured metadata",
            "crafted metadata value changes parser behavior",
            "user-driven processing of untrusted document content",
            "document viewer process spawned after unusual metadata input",
        ),
        (
            "administrative policy interface",
            "overly broad privilege assignment",
            "policy manipulation to widen account privileges",
            "account permission change through management tooling",
            "administrator group update from an uncommon parent process",
        ),
        (
            "web request router",
            "improper neutralization of request parameters",
            "request parameter manipulation changes backend query flow",
            "server-side request handling through script interpreter",
            "script host invoked with unusual request-derived arguments",
        ),
        (
            "service discovery endpoint",
            "exposure of internal resource information",
            "enumeration of backend service names",
            "remote service discovery through application messages",
            "network service listing from a nonstandard process lineage",
        ),
        (
            "archive extraction utility",
            "improper control of decompression resource use",
            "oversized archive expansion consumes application resources",
            "resource exhaustion through user-supplied compressed content",
            "archive tool execution with unusually large extracted output",
        ),
        (
            "template rendering engine",
            "improper separation of template data and directives",
            "template parameter manipulation changes rendered control flow",
            "template-driven command construction",
            "renderer process launched with unexpected directive-like input",
        ),
    ]
    sigma_title = target_sigma_rule.get("title") or "target rule"
    target_terms = [
        term
        for term in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{4,}", target_cve_description)
        if term.lower() not in {"before", "after", "through", "allows", "users", "version"}
    ][:8]
    if not target_terms:
        target_terms = ["application", "component", "request", "profile"]

    distractors: list[dict[str, Any]] = []
    for index in range(count):
        digest = sha256(
            f"{benchmark_id}|local|{batch_index}|{index}|{sigma_title}".encode("utf-8")
        ).hexdigest()
        topic = topics[_digest_int(digest, 0, 2) % len(topics)]
        anchor = target_terms[_digest_int(digest, 2, 4) % len(target_terms)]
        variant = 1 + _digest_int(digest, 4, 8) % 9_999
        component, weakness, capec, attack, sigma = topic
        distractors.append(
            {
                "cve_description": (
                    f"{anchor} scenario {variant}: a {component} mishandles external input "
                    "under a related but nonmatching condition."
                ),
                "cwe": weakness,
                "capec": capec,
                "attack_technique": attack,
                "sigma_rule": sigma,
            }
        )
    return distractors


def _synthetic_distractor_graph(
    *,
    benchmark_id: str,
    distractors: Iterable[dict[str, Any]],
    avoid_node_ids: set[str] | None = None,
    chain_index_offset: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    chain_count = 0
    used_node_ids = set(avoid_node_ids or ())
    for local_index, distractor in enumerate(distractors, start=1):
        if not isinstance(distractor, dict):
            continue
        raw_index = chain_index_offset + local_index
        fields = {
            "cve_description": _clean_generated_text(distractor.get("cve_description")),
            "cwe_title": _clean_generated_text(distractor.get("cwe_title") or distractor.get("cwe")),
            "cwe_description": _clean_generated_text(distractor.get("cwe_description")),
            "capec_title": _clean_generated_text(distractor.get("capec_title") or distractor.get("capec")),
            "capec_description": _clean_generated_text(distractor.get("capec_description")),
            "attack_technique_title": _clean_generated_text(
                distractor.get("attack_technique_title") or distractor.get("attack_technique")
            ),
            "attack_technique_description": _clean_generated_text(
                distractor.get("attack_technique_description")
            ),
            "sigma_rule_title": _clean_generated_text(
                distractor.get("sigma_rule_title") or distractor.get("sigma_rule")
            ),
            "sigma_rule_description": _clean_generated_text(distractor.get("sigma_rule_description")),
        }
        if not any(fields.values()):
            continue

        digest = sha256(
            json.dumps(
                {"benchmark_id": benchmark_id, "index": raw_index, "fields": fields},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        synthetic_ids = _plausible_synthetic_chain_ids(digest, used_node_ids=used_node_ids)
        used_node_ids.update(synthetic_ids["node_ids"])
        chain_count += 1

        cve_id = synthetic_ids["cve_id"]
        cwe_id = synthetic_ids["cwe_id"]
        capec_id = synthetic_ids["capec_id"]
        attack_id = synthetic_ids["attack_id"]
        sigma_rule_id = synthetic_ids["sigma_rule_id"]
        cve_node_id = synthetic_ids["cve_node_id"]
        cwe_node_id = synthetic_ids["cwe_node_id"]
        capec_node_id = synthetic_ids["capec_node_id"]
        attack_node_id = synthetic_ids["attack_node_id"]
        sigma_node_id = synthetic_ids["sigma_node_id"]

        cve_description = fields["cve_description"] or "A plausible vulnerability scenario."
        cwe_title = fields["cwe_title"] or "Weakness Pattern"
        capec_title = fields["capec_title"] or "Attack Pattern"
        attack_title = fields["attack_technique_title"] or "Technique"
        sigma_title = fields["sigma_rule_title"] or "Detection Pattern"

        nodes.extend(
            [
                _synthetic_node_for_prompt(
                    node_id=cve_node_id,
                    entity_type="cve",
                    entity_id=cve_id,
                    title=cve_id,
                    description=cve_description,
                ),
                _synthetic_node_for_prompt(
                    node_id=cwe_node_id,
                    entity_type="cwe",
                    entity_id=cwe_id,
                    title=cwe_title,
                    description=fields["cwe_description"] or cwe_title,
                ),
                _synthetic_node_for_prompt(
                    node_id=capec_node_id,
                    entity_type="capec",
                    entity_id=capec_id,
                    title=capec_title,
                    description=fields["capec_description"] or capec_title,
                ),
                _synthetic_node_for_prompt(
                    node_id=attack_node_id,
                    entity_type="attack-technique",
                    entity_id=attack_id,
                    title=attack_title,
                    description=fields["attack_technique_description"] or attack_title,
                ),
                _synthetic_node_for_prompt(
                    node_id=sigma_node_id,
                    entity_type="sigma-rule",
                    entity_id=sigma_rule_id,
                    title=sigma_title,
                    description=fields["sigma_rule_description"] or sigma_title,
                ),
            ]
        )
        edges.extend(
            [
                _synthetic_edge_for_prompt(
                    source=cve_node_id,
                    relationship="has_weakness",
                    target=cwe_node_id,
                ),
                _synthetic_edge_for_prompt(
                    source=cwe_node_id,
                    relationship="related_attack_pattern",
                    target=capec_node_id,
                ),
                _synthetic_edge_for_prompt(
                    source=capec_node_id,
                    relationship="maps_to_attack_technique",
                    target=attack_node_id,
                ),
                _synthetic_edge_for_prompt(
                    source=attack_node_id,
                    relationship="detected_by_sigma_rule",
                    target=sigma_node_id,
                ),
            ]
        )
    return nodes, edges, chain_count


def _plausible_synthetic_chain_ids(
    digest: str,
    *,
    used_node_ids: set[str],
) -> dict[str, Any]:
    attempt = 0
    candidate_digest = digest
    while True:
        cve_year = 2017 + _digest_int(candidate_digest, 0, 2) % 9
        cve_sequence = 1000 + _digest_int(candidate_digest, 2, 10) % 95_000
        cwe_id = f"CWE-{1 + _digest_int(candidate_digest, 10, 14) % 1_450}"
        capec_id = f"CAPEC-{1 + _digest_int(candidate_digest, 14, 18) % 700}"
        attack_base = 1001 + _digest_int(candidate_digest, 18, 22) % 660
        if _digest_int(candidate_digest, 22, 24) % 3 == 0:
            attack_id = f"T{attack_base}.{1 + _digest_int(candidate_digest, 24, 26) % 9:03d}"
        else:
            attack_id = f"T{attack_base}"
        cve_id = f"CVE-{cve_year}-{cve_sequence}"
        sigma_rule_id = _uuid_from_digest(candidate_digest[8:40])

        cve_node_id = f"nvd:{cve_id}"
        cwe_node_id = f"mitre-cwe:{cwe_id}"
        capec_node_id = f"capec:{capec_id}"
        attack_node_id = f"mitre-attack:{attack_id}"
        sigma_node_id = f"sigma:{sigma_rule_id}"
        node_ids = {cve_node_id, cwe_node_id, capec_node_id, attack_node_id, sigma_node_id}
        if not node_ids.intersection(used_node_ids):
            return {
                "cve_id": cve_id,
                "cwe_id": cwe_id,
                "capec_id": capec_id,
                "attack_id": attack_id,
                "sigma_rule_id": sigma_rule_id,
                "cve_node_id": cve_node_id,
                "cwe_node_id": cwe_node_id,
                "capec_node_id": capec_node_id,
                "attack_node_id": attack_node_id,
                "sigma_node_id": sigma_node_id,
                "node_ids": node_ids,
            }
        attempt += 1
        candidate_digest = sha256(f"{digest}|retry|{attempt}".encode("utf-8")).hexdigest()


def _digest_int(digest: str, start: int, end: int) -> int:
    return int(digest[start:end], 16)


def _synthetic_node_for_prompt(
    *,
    node_id: str,
    entity_type: str,
    entity_id: str,
    title: str,
    description: str,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "type": entity_type,
        "id": entity_id,
        "title": _snippet(title, max_chars=160),
        "description": _snippet(description, max_chars=500),
    }


def _synthetic_edge_for_prompt(
    *,
    source: str,
    relationship: str,
    target: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "relationship": relationship,
        "target": target,
    }


def _clean_generated_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return _strip_generated_meta_terms(_redact_cve_ids(value.strip(), ""))


def _strip_generated_meta_terms(text: str) -> str:
    cleaned = text
    replacements = {
        r"\bsynthetic\s+(weakness|pattern|technique|rule|vulnerability|chain|scenario)\s*:\s*": "",
        r"\bsynthetic\s+(weakness|pattern|technique|rule|vulnerability|chain|scenario)\b": r"\1",
        r"\bdistractor\s+(weakness|pattern|technique|rule|vulnerability|chain|scenario)\s*:\s*": "",
        r"\bdistractor\s+(weakness|pattern|technique|rule|vulnerability|chain|scenario)\b": r"\1",
        r"\bdecoy\s+(host|system|service|rule|endpoint|content|page|file|account)\b": r"lookalike \1",
        r"\bfake\s+(plugin|page|document|login|certificate|snippet|host|app)\b": r"lookalike \1",
        r"\bplaceholder\b": "stand-in",
        r"\bdummy\b": "test",
        r"\bsynthetic\b": "plausible",
        r"\bdistractor\b": "near-miss",
        r"\bdecoy\b": "lookalike",
        r"\bfake\b": "lookalike",
    }
    for pattern, replacement in replacements.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split())


def _uuid_from_digest(text: str) -> str:
    digest = (text + sha256(text.encode("utf-8")).hexdigest())[:32]
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def _strip_sigma_yaml(content: str) -> str:
    if "```yaml" in content:
        return content.split("```yaml", 1)[0].strip()
    return content


def _format_question(target_cve_description: str, target_sigma_rule: dict[str, Any]) -> str:
    question = (
        "You are given a target CVE description, a target Sigma rule, and a shuffled "
        "set of Tier 1 nodes and edges. Use only the provided graph. "
        "Starting with the target Sigma rule, find the valid "
        "Sigma -> ATT&CK -> CAPEC -> CWE -> CVE chain that maps to the target CVE "
        "description, and identify the final CVE ID.\n\n"
        f"Target CVE description:\n{target_cve_description}\n\n"
        "Target Sigma rule:\n"
        f"{json.dumps(target_sigma_rule, sort_keys=True)}\n\n"
        "Return JSON only using the provided schema."
    )
    return _append_response_format_contract(question)


def _append_response_format_contract(question: str) -> str:
    question = question.strip()
    if _RESPONSE_FORMAT_CONTRACT in question:
        return question
    return f"{question}\n\n{_RESPONSE_FORMAT_CONTRACT}"


def _generate_openai_task_text(
    *,
    source_target_cve_description: str,
    target_sigma_rule: dict[str, Any],
    synthetic_distractor_chains: int,
    openai_model: str | None,
    openai_api_key: str | None,
    openai_base_url: str | None,
    temperature: float | None,
    max_completion_tokens: int,
    input_cost_per_million: float,
    output_cost_per_million: float,
    max_cost_usd: float | None,
    cost_tracker: _OpenAICostTracker,
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is required for query_mode='openai'.") from exc

    client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
    if synthetic_distractor_chains:
        system = (
            "You write cybersecurity benchmark tasks for graph reasoning. "
            "Return only valid JSON. Do not reveal the target CVE ID. For the main task "
            "question and rewritten CVE description, do not add facts not present in the input. "
            "You may create only the requested synthetic false distractor chain descriptions; "
            "do not create real node IDs, edges, answers, or source-grounded claims."
        )
        key_list = "`target_cve_description`, `question`, and `synthetic_distractors`"
        synthetic_instruction = (
            "\n\n"
            f"`synthetic_distractors` must be an array with exactly {synthetic_distractor_chains} "
            "objects. Each object should describe a plausible but false near-miss chain that is "
            "semantically related to the target CVE description or target Sigma rule. These are "
            "false graph decoys, not source-grounded facts. Do not include real CVE IDs, real CWE "
            "IDs, real CAPEC IDs, real ATT&CK IDs, Sigma UUIDs, node IDs, or edges; code assigns "
            "plausible-looking IDs separately. Keep each "
            "field to one short phrase or sentence. Use exactly these object keys: "
            "`cve_description`, `cwe`, `capec`, `attack_technique`, and `sigma_rule`. "
            "Do not use meta-label words such as synthetic, distractor, decoy, fake, "
            "placeholder, or dummy in any returned field."
        )
    else:
        system = (
            "You write cybersecurity benchmark tasks for graph reasoning. "
            "Return only valid JSON. Do not reveal any CVE ID. Do not add facts not present "
            "in the input. Do not create nodes, edges, answers, or distractors."
        )
        key_list = "`target_cve_description` and `question`"
        synthetic_instruction = ""
    user = (
        f"Create one benchmark task as JSON with exactly these keys: {key_list}.\n\n"
        "`target_cve_description` must be a concise grounded rewrite of the source CVE "
        "description. Preserve the affected product/vendor, vulnerable component, "
        "vulnerability behavior, impact, attacker condition, and version range when "
        "present. Do not include a CVE ID.\n\n"
        "`question` must ask the model to start from the target Sigma rule and work "
        "backward through the provided shuffled graph to find the full chain that maps "
        "to the target CVE description. It must ask for this order: Sigma rule, ATT&CK "
        "technique, CAPEC, CWE, CVE ID. It must state that only the provided nodes and "
        "edges may be used. Do not mention distractors. It must tell "
        "the model to return JSON only using the provided schema. Do not include your own "
        "JSON schema, example answer, candidate answer, output format block, or gold ID; "
        "the code will append the exact response format separately."
        f"{synthetic_instruction}\n\n"
        f"Source CVE description:\n{source_target_cve_description}\n\n"
        f"Target Sigma rule:\n{json.dumps(target_sigma_rule, sort_keys=True)}"
    )
    request: dict[str, Any] = {
        "model": openai_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_completion_tokens": max_completion_tokens,
    }
    if temperature is not None:
        request["temperature"] = temperature
    response = client.chat.completions.create(**request)
    _update_openai_cost_tracker(
        cost_tracker,
        response=response,
        input_cost_per_million=input_cost_per_million,
        output_cost_per_million=output_cost_per_million,
        max_cost_usd=max_cost_usd,
    )
    content = response.choices[0].message.content
    if not content:
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        usage_json = usage.model_dump() if hasattr(usage, "model_dump") else usage
        raise RuntimeError(
            "OpenAI returned an empty rewritten CVE description. "
            f"finish_reason={getattr(choice, 'finish_reason', None)!r}; "
            f"usage={usage_json!r}. Try increasing --openai-max-completion-tokens."
        )
    parsed = _parse_json_object(content)
    if not isinstance(parsed, dict):
        raise RuntimeError("OpenAI returned a non-object benchmark task.")
    target_cve_description = parsed.get("target_cve_description")
    question = parsed.get("question")
    if not isinstance(target_cve_description, str) or not isinstance(question, str):
        raise RuntimeError("OpenAI task JSON must contain string target_cve_description and question.")
    target_cve_description = _redact_cve_ids(target_cve_description.strip(), "")
    question = _redact_cve_ids(question.strip(), "")
    synthetic_distractors = _parse_generated_synthetic_distractors(
        parsed.get("synthetic_distractors"),
        expected_count=synthetic_distractor_chains,
    )
    if "distractor" in question.lower() or "synthetic" in question.lower():
        question = _format_question(target_cve_description, target_sigma_rule)
    if target_sigma_rule["rule_id"] not in question and target_sigma_rule["node_id"] not in question:
        question = (
            f"Starting with this Sigma rule {json.dumps(target_sigma_rule, sort_keys=True)}, "
            f"find the full Sigma -> ATT&CK -> CAPEC -> CWE -> CVE chain that maps "
            f"to the following CVE description: {target_cve_description}. "
            "Return JSON only using the provided schema."
        )
    if "provided schema" not in question.lower():
        question = question.rstrip() + (
            " Return JSON only using the provided schema."
        )
    question = _append_response_format_contract(question)
    return {
        "target_cve_description": target_cve_description,
        "question": question,
        "synthetic_distractors": synthetic_distractors,
    }


def _generate_openai_synthetic_distractors(
    *,
    target_cve_description: str,
    target_sigma_rule: dict[str, Any],
    gold_chain_context: dict[str, Any] | None,
    synthetic_distractor_chains: int,
    batch_index: int,
    openai_model: str | None,
    openai_api_key: str | None,
    openai_base_url: str | None,
    temperature: float | None,
    max_completion_tokens: int,
    input_cost_per_million: float,
    output_cost_per_million: float,
    max_cost_usd: float | None,
    cost_tracker: _OpenAICostTracker,
) -> list[dict[str, Any]]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is required for query_mode='openai'.") from exc

    client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
    system = (
        "You write cybersecurity benchmark decoys for graph reasoning. "
        "Return only valid JSON. The decoys must be plausible but false. "
        "Do not create or include real CVE IDs, real CWE IDs, real CAPEC IDs, "
        "real ATT&CK IDs, Sigma UUIDs, node IDs, edges, answers, or source-grounded claims."
    )
    hidden_context = (
        "\n\nHidden gold-chain concepts for generating harder near misses. "
        "Use these only to create semantically adjacent false concepts; do not copy exact IDs, "
        "do not state these are gold, and do not repeat exact titles as the answer:\n"
        f"{json.dumps(gold_chain_context, sort_keys=True)}"
        if gold_chain_context
        else ""
    )
    user = (
        "Create synthetic false distractor chain descriptions as JSON with exactly one key: "
        "`synthetic_distractors`.\n\n"
        f"`synthetic_distractors` must be an array with exactly {synthetic_distractor_chains} "
        "objects. Make them adversarial near misses, not generic noise. Each object should "
        "be close enough that a capable model could plausibly confuse it with the target path "
        "unless it verifies every graph hop. Mix these styles across the batch: same vendor or "
        "product family but wrong component; same vulnerability class but wrong root cause; "
        "neighboring CWE concept; neighboring CAPEC behavior; same ATT&CK-style behavior but "
        "wrong operational precondition; and a Sigma-like detection description that resembles "
        "the target but should lead to a different endpoint. Prefer concrete security language "
        "over broad wording like 'improper handling' unless the target itself is broad. Keep "
        "every field to one short phrase or sentence. Use exactly these object keys: `cve_description`, "
        "`cwe`, `capec`, `attack_technique`, and `sigma_rule`. Do not use meta-label "
        "words such as synthetic, distractor, decoy, fake, placeholder, or dummy in any "
        "returned field.\n\n"
        f"Batch index for diversity: {batch_index}\n\n"
        f"Target CVE clue, without CVE ID:\n{target_cve_description}\n\n"
        f"Target Sigma rule:\n{json.dumps(target_sigma_rule, sort_keys=True)}"
        f"{hidden_context}"
    )
    request: dict[str, Any] = {
        "model": openai_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_completion_tokens": max_completion_tokens,
    }
    if temperature is not None:
        request["temperature"] = temperature
    response = client.chat.completions.create(**request)
    _update_openai_cost_tracker(
        cost_tracker,
        response=response,
        input_cost_per_million=input_cost_per_million,
        output_cost_per_million=output_cost_per_million,
        max_cost_usd=max_cost_usd,
    )
    content = response.choices[0].message.content
    if not content:
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        usage_json = usage.model_dump() if hasattr(usage, "model_dump") else usage
        raise RuntimeError(
            "OpenAI returned an empty synthetic distractor batch. "
            f"finish_reason={getattr(choice, 'finish_reason', None)!r}; "
            f"usage={usage_json!r}. Try increasing --openai-max-completion-tokens."
        )
    parsed = _parse_json_object(content)
    return _parse_generated_synthetic_distractors(
        parsed.get("synthetic_distractors"),
        expected_count=synthetic_distractor_chains,
    )


def _parse_generated_synthetic_distractors(
    value: Any,
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    if expected_count == 0:
        return []
    if not isinstance(value, list):
        raise RuntimeError("OpenAI task JSON must contain a synthetic_distractors list.")
    parsed: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            parsed.append(item)
    if len(parsed) < expected_count:
        raise RuntimeError(
            "OpenAI returned too few synthetic distractor chains: "
            f"{len(parsed)} < {expected_count}."
        )
    return parsed[:expected_count]


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenAI returned invalid JSON: {text[:200]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("OpenAI returned a non-object benchmark task.")
    return parsed


def _update_openai_cost_tracker(
    cost_tracker: _OpenAICostTracker,
    *,
    response: Any,
    input_cost_per_million: float,
    output_cost_per_million: float,
    max_cost_usd: float | None,
) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    cost_tracker.input_tokens += input_tokens
    cost_tracker.output_tokens += output_tokens
    cost_tracker.estimated_cost_usd += (
        input_tokens * input_cost_per_million / 1_000_000
        + output_tokens * output_cost_per_million / 1_000_000
    )
    if max_cost_usd is not None and cost_tracker.estimated_cost_usd > max_cost_usd:
        raise RuntimeError(
            "OpenAI generation exceeded the configured local cost cap: "
            f"${cost_tracker.estimated_cost_usd:.4f} > ${max_cost_usd:.4f}. "
            "The latest API call completed before the cap could be checked."
        )


def _load_edge_index(edges_path: Path) -> _EdgeIndex:
    edge_by_id: dict[str, dict[str, Any]] = {}
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    related_cwes: dict[str, set[str]] = defaultdict(set)
    related_capecs: dict[str, set[str]] = defaultdict(set)

    parquet_file = pq.ParquetFile(edges_path)
    for batch in parquet_file.iter_batches(columns=_EDGE_COLUMNS, batch_size=8192):
        for row in batch.to_pylist():
            edge_by_id[row["edge_id"]] = row
            outgoing[row["source_node_id"]].append(row)
            incoming[row["target_node_id"]].append(row)
            if (
                row["relationship_type"] == "related_weakness"
                and row["source_entity_type"] == "cwe"
                and row["target_entity_type"] == "cwe"
            ):
                related_cwes[row["source_node_id"]].add(row["target_node_id"])
                related_cwes[row["target_node_id"]].add(row["source_node_id"])
            elif (
                row["relationship_type"] == "related_attack_pattern"
                and row["source_entity_type"] == "capec"
                and row["target_entity_type"] == "capec"
            ):
                related_capecs[row["source_node_id"]].add(row["target_node_id"])
                related_capecs[row["target_node_id"]].add(row["source_node_id"])

    return _EdgeIndex(
        edge_by_id=edge_by_id,
        outgoing={key: tuple(value) for key, value in outgoing.items()},
        incoming={key: tuple(value) for key, value in incoming.items()},
        related_cwes={key: tuple(sorted(value)) for key, value in related_cwes.items()},
        related_capecs={key: tuple(sorted(value)) for key, value in related_capecs.items()},
    )


def _load_nodes(nodes_path: Path, node_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not node_ids:
        return {}
    remaining = set(node_ids)
    nodes: dict[str, dict[str, Any]] = {}
    parquet_file = pq.ParquetFile(nodes_path)
    for batch in parquet_file.iter_batches(columns=_NODE_COLUMNS, batch_size=8192):
        for row in batch.to_pylist():
            node_id = row["node_id"]
            if node_id not in remaining:
                continue
            nodes[node_id] = row
            remaining.remove(node_id)
        if not remaining:
            break
    if remaining:
        missing = ", ".join(sorted(remaining)[:10])
        raise ValueError(f"Missing node rows for {len(remaining)} node IDs, including: {missing}")
    return nodes


def _snippet(text: str, *, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _redact_cve_ids(text: str, gold_cve_id: str) -> str:
    redacted = _CVE_RE.sub("[REDACTED_CVE_ID]", text)
    if gold_cve_id:
        redacted = re.sub(re.escape(gold_cve_id), "[REDACTED_CVE_ID]", redacted, flags=re.IGNORECASE)
    return redacted


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest[:24]}"


def _summary_to_json(summary: ReverseEndpointBenchmarkSummary) -> dict[str, Any]:
    return {
        "output_path": str(summary.output_path),
        "summary_path": str(summary.summary_path),
        "priority_chains_path": str(summary.priority_chains_path),
        "nodes_path": str(summary.nodes_path),
        "edges_path": str(summary.edges_path),
        "scanned_rows": summary.scanned_rows,
        "unique_gold_candidates": summary.unique_gold_candidates,
        "examples_written": summary.examples_written,
        "skipped_without_valid_gold_path": summary.skipped_without_valid_gold_path,
        "skipped_without_unique_path": summary.skipped_without_unique_path,
        "skipped_without_enough_distractors": summary.skipped_without_enough_distractors,
        "skipped_by_diversity_cap": summary.skipped_by_diversity_cap,
        "seed": summary.seed,
        "gold_pool_multiplier": summary.gold_pool_multiplier,
        "min_distractor_nodes": summary.min_distractor_nodes,
        "target_distractor_nodes": summary.target_distractor_nodes,
        "max_distractor_bundles": summary.max_distractor_bundles,
        "max_distractor_attempts": summary.max_distractor_attempts,
        "max_examples_per_attack_technique": summary.max_examples_per_attack_technique,
        "max_examples_per_capec": summary.max_examples_per_capec,
        "max_examples_per_cwe": summary.max_examples_per_cwe,
        "max_examples_per_cve": summary.max_examples_per_cve,
        "unique_attack_techniques": summary.unique_attack_techniques,
        "unique_capecs": summary.unique_capecs,
        "unique_cwes": summary.unique_cwes,
        "unique_cves": summary.unique_cves,
        "synthetic_distractor_chains": summary.synthetic_distractor_chains,
        "target_prompt_tokens": summary.target_prompt_tokens,
        "target_prompt_token_tolerance": summary.target_prompt_token_tolerance,
        "synthetic_distractor_batch_size": summary.synthetic_distractor_batch_size,
        "max_synthetic_distractor_chains": summary.max_synthetic_distractor_chains,
        "prompt_token_count_min": summary.prompt_token_count_min,
        "prompt_token_count_max": summary.prompt_token_count_max,
        "prompt_token_count_mean": summary.prompt_token_count_mean,
        "query_mode": summary.query_mode,
        "openai_model": summary.openai_model,
        "openai_base_url": summary.openai_base_url,
        "openai_input_tokens": summary.openai_input_tokens,
        "openai_output_tokens": summary.openai_output_tokens,
        "openai_estimated_cost_usd": summary.openai_estimated_cost_usd,
        "openai_max_cost_usd": summary.openai_max_cost_usd,
        "hard_predictions_path": (
            str(summary.hard_predictions_path) if summary.hard_predictions_path else None
        ),
        "hard_prediction_failure_count": summary.hard_prediction_failure_count,
        "hard_selection_candidates": summary.hard_selection_candidates,
        "hard_min_selection_score": summary.hard_min_selection_score,
        "hard_max_points": summary.hard_max_points,
        "excluded_benchmark_paths": [
            str(path) for path in summary.excluded_benchmark_paths
        ],
        "excluded_existing_examples": summary.excluded_existing_examples,
        "excluded_existing_cves": summary.excluded_existing_cves,
        "skipped_by_existing_exclusion": summary.skipped_by_existing_exclusion,
    }
