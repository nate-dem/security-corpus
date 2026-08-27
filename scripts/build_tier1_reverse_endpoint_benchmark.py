#!/usr/bin/env python3
"""Build graph-slice reverse endpoint examples from Tier 1 priority chains."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))

from ingest.derived.tier1_reverse_endpoint_benchmark import (  # noqa: E402
    build_reverse_endpoint_benchmark,
)


def _path_arg(value: str) -> Path:
    return Path(value)


def _priority_arg(value: str) -> tuple[int, ...]:
    priorities = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    invalid = [priority for priority in priorities if priority not in {1, 2}]
    if invalid:
        raise argparse.ArgumentTypeError("reverse endpoint examples require priorities 1 and/or 2")
    return priorities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Sigma-to-CVE graph reasoning benchmark examples.",
    )
    parser.add_argument(
        "--priority-chains",
        type=_path_arg,
        default=Path("data/tier1-reasoning-clean-v2/priority_chains.parquet"),
        help="Clean-v2 priority_chains.parquet input.",
    )
    parser.add_argument(
        "--nodes",
        type=_path_arg,
        default=Path("data/tier1-reasoning-clean-v2/nodes.parquet"),
        help="Clean-v2 nodes.parquet input for graph node content.",
    )
    parser.add_argument(
        "--edges",
        type=_path_arg,
        default=Path("data/tier1-reasoning-clean-v2/edges.parquet"),
        help="Clean-v2 edges.parquet input for graph edges and distractors.",
    )
    parser.add_argument(
        "--output",
        type=_path_arg,
        required=True,
        help="JSONL benchmark output path.",
    )
    parser.add_argument(
        "--summary",
        type=_path_arg,
        help="Optional JSON summary path. Defaults to OUTPUT with .summary.json suffix.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        required=True,
        help="Number of examples to sample. Researcher-owned benchmark size.",
    )
    parser.add_argument(
        "--priorities",
        type=_priority_arg,
        default=(1, 2),
        help="Comma-separated priorities to sample from. Must be 1 and/or 2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Deterministic sampling seed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Rows per Parquet scan batch.",
    )
    parser.add_argument(
        "--gold-pool-multiplier",
        type=int,
        default=50,
        help=(
            "Reservoir size multiplier used before filtering for enough CWE/CAPEC "
            "distractors. Larger values improve the chance of exactly max-examples."
        ),
    )
    parser.add_argument(
        "--min-distractor-nodes",
        type=int,
        default=20,
        help="Minimum number of non-gold graph nodes required in each task.",
    )
    parser.add_argument(
        "--max-distractor-bundles",
        type=int,
        default=40,
        help="Maximum number of distractor edge bundles to add to each graph.",
    )
    parser.add_argument(
        "--max-distractor-attempts",
        type=int,
        default=400,
        help="Maximum distractor bundle attempts per example before skipping.",
    )
    parser.add_argument(
        "--target-distractor-nodes",
        type=int,
        default=40,
        help=(
            "Target number of real non-gold distractor nodes to try to add. "
            "The builder still only requires --min-distractor-nodes."
        ),
    )
    parser.add_argument(
        "--max-examples-per-attack-technique",
        type=int,
        help="Optional diversity cap for selected gold examples sharing an ATT&CK technique.",
    )
    parser.add_argument(
        "--max-examples-per-capec",
        type=int,
        help="Optional diversity cap for selected gold examples sharing a CAPEC.",
    )
    parser.add_argument(
        "--max-examples-per-cwe",
        type=int,
        help="Optional diversity cap for selected gold examples sharing a CWE.",
    )
    parser.add_argument(
        "--max-examples-per-cve",
        type=int,
        help="Optional diversity cap for selected gold examples sharing a CVE.",
    )
    parser.add_argument(
        "--hard-predictions",
        type=_path_arg,
        help=(
            "Optional model prediction JSONL used to bias gold sampling toward "
            "previous failure neighborhoods. The builder uses failed gold and "
            "predicted CVE/CWE/CAPEC/ATT&CK/Sigma IDs as hard-selection seeds."
        ),
    )
    parser.add_argument(
        "--exclude-benchmark-jsonl",
        type=_path_arg,
        action="append",
        default=[],
        help=(
            "Existing benchmark JSONL to exclude from this run. Can be passed "
            "multiple times. Excludes prior benchmark IDs, gold chain IDs, "
            "detection chain IDs, and CVEs."
        ),
    )
    parser.add_argument(
        "--hard-min-selection-score",
        type=int,
        default=1,
        help=(
            "Minimum hard-neighborhood score required when --hard-predictions is set. "
            "Higher values focus on rows sharing more failed-path concepts."
        ),
    )
    parser.add_argument(
        "--hard-max-points",
        type=int,
        help=(
            "Optional point threshold for adding extra rows from --hard-predictions. "
            "By default all exact-chain failures are used."
        ),
    )
    parser.add_argument(
        "--query-mode",
        choices=("deterministic", "openai"),
        default="deterministic",
        help="Use deterministic prompts or generate prompts through the OpenAI API.",
    )
    parser.add_argument(
        "--openai-model",
        help="OpenAI model to use when --query-mode openai is enabled.",
    )
    parser.add_argument(
        "--openai-base-url",
        help=(
            "Optional OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL "
            "or OPENAI_API_BASE when set."
        ),
    )
    parser.add_argument(
        "--openai-temperature",
        type=float,
        default=None,
        help=(
            "Optional prompt generation temperature. Omitted by default because "
            "some GPT-5.5 gateways only accept the provider default."
        ),
    )
    parser.add_argument(
        "--openai-max-completion-tokens",
        type=int,
        default=8000,
        help=(
            "Upper bound for visible output plus reasoning tokens. GPT-5.5 may "
            "return empty visible content if this is too small."
        ),
    )
    parser.add_argument(
        "--synthetic-distractor-chains",
        type=int,
        default=0,
        help=(
            "Number of GPT-generated false distractor chains to add to each example. "
            "Requires --query-mode openai. Gold paths remain verified from real data."
        ),
    )
    parser.add_argument(
        "--target-prompt-tokens",
        type=int,
        help=(
            "Optional target token count for the assembled downstream model prompt. "
            "When set, the builder generates synthetic distractor chains in batches "
            "until each example reaches the target range. Requires --query-mode openai."
        ),
    )
    parser.add_argument(
        "--target-prompt-token-tolerance",
        type=int,
        default=16_000,
        help=(
            "Allowed token-count tolerance around --target-prompt-tokens. The builder "
            "stops once the prompt is at least target minus this tolerance."
        ),
    )
    parser.add_argument(
        "--synthetic-distractor-batch-size",
        type=int,
        default=20,
        help="Number of additional GPT-generated distractor chains to request per expansion batch.",
    )
    parser.add_argument(
        "--max-synthetic-distractor-chains",
        type=int,
        help=(
            "Optional hard cap on total synthetic distractor chains per example. "
            "Defaults to 800 when --target-prompt-tokens is set."
        ),
    )
    parser.add_argument(
        "--openai-input-cost-per-million",
        type=float,
        default=5.0,
        help="Estimated input-token cost per 1M tokens. Defaults to GPT-5.5 standard pricing.",
    )
    parser.add_argument(
        "--openai-output-cost-per-million",
        type=float,
        default=30.0,
        help="Estimated output-token cost per 1M tokens. Defaults to GPT-5.5 standard pricing.",
    )
    parser.add_argument(
        "--openai-max-cost-usd",
        type=float,
        help=(
            "Optional local estimated cost cap for this run. The script checks after "
            "each API response, so one final request can exceed the cap slightly."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_reverse_endpoint_benchmark(
        args.priority_chains,
        args.output,
        nodes_path=args.nodes,
        edges_path=args.edges,
        summary_path=args.summary,
        max_examples=args.max_examples,
        seed=args.seed,
        batch_size=args.batch_size,
        priorities=args.priorities,
        gold_pool_multiplier=args.gold_pool_multiplier,
        min_distractor_nodes=args.min_distractor_nodes,
        target_distractor_nodes=args.target_distractor_nodes,
        max_distractor_bundles=args.max_distractor_bundles,
        max_distractor_attempts=args.max_distractor_attempts,
        max_examples_per_attack_technique=args.max_examples_per_attack_technique,
        max_examples_per_capec=args.max_examples_per_capec,
        max_examples_per_cwe=args.max_examples_per_cwe,
        max_examples_per_cve=args.max_examples_per_cve,
        query_mode=args.query_mode,
        openai_model=args.openai_model,
        openai_base_url=args.openai_base_url,
        openai_temperature=args.openai_temperature,
        openai_max_completion_tokens=args.openai_max_completion_tokens,
        synthetic_distractor_chains=args.synthetic_distractor_chains,
        target_prompt_tokens=args.target_prompt_tokens,
        target_prompt_token_tolerance=args.target_prompt_token_tolerance,
        synthetic_distractor_batch_size=args.synthetic_distractor_batch_size,
        max_synthetic_distractor_chains=args.max_synthetic_distractor_chains,
        openai_input_cost_per_million=args.openai_input_cost_per_million,
        openai_output_cost_per_million=args.openai_output_cost_per_million,
        openai_max_cost_usd=args.openai_max_cost_usd,
        hard_predictions_path=args.hard_predictions,
        hard_min_selection_score=args.hard_min_selection_score,
        hard_max_points=args.hard_max_points,
        exclude_benchmark_paths=args.exclude_benchmark_jsonl,
    )
    print(f"output: {summary.output_path}")
    print(f"summary: {summary.summary_path}")
    print(f"nodes: {summary.nodes_path}")
    print(f"edges: {summary.edges_path}")
    print(f"scanned priority rows: {summary.scanned_rows:,}")
    print(f"unique gold candidates: {summary.unique_gold_candidates:,}")
    print(f"examples written: {summary.examples_written:,}")
    print(f"skipped without valid gold path: {summary.skipped_without_valid_gold_path:,}")
    print(f"skipped without unique path: {summary.skipped_without_unique_path:,}")
    print(f"skipped without enough distractors: {summary.skipped_without_enough_distractors:,}")
    print(f"skipped by diversity cap: {summary.skipped_by_diversity_cap:,}")
    print(f"seed: {summary.seed}")
    print(f"gold pool multiplier: {summary.gold_pool_multiplier}")
    print(f"real distractor nodes: min {summary.min_distractor_nodes}, target {summary.target_distractor_nodes}")
    print(f"max real distractor bundles: {summary.max_distractor_bundles}")
    print(f"max real distractor attempts: {summary.max_distractor_attempts}")
    if summary.max_examples_per_attack_technique:
        print(f"max examples per ATT&CK technique: {summary.max_examples_per_attack_technique}")
    if summary.max_examples_per_capec:
        print(f"max examples per CAPEC: {summary.max_examples_per_capec}")
    if summary.max_examples_per_cwe:
        print(f"max examples per CWE: {summary.max_examples_per_cwe}")
    if summary.max_examples_per_cve:
        print(f"max examples per CVE: {summary.max_examples_per_cve}")
    if summary.excluded_benchmark_paths:
        print(
            "excluded benchmark files: "
            + ", ".join(str(path) for path in summary.excluded_benchmark_paths)
        )
        print(f"excluded existing examples: {summary.excluded_existing_examples:,}")
        print(f"excluded existing CVEs: {summary.excluded_existing_cves:,}")
        print(f"skipped by existing exclusion: {summary.skipped_by_existing_exclusion:,}")
    if summary.hard_predictions_path:
        print(f"hard predictions: {summary.hard_predictions_path}")
        print(f"hard prediction failures: {summary.hard_prediction_failure_count:,}")
        print(f"hard selection candidates: {summary.hard_selection_candidates:,}")
        print(f"hard min selection score: {summary.hard_min_selection_score}")
        if summary.hard_max_points is not None:
            print(f"hard max points: {summary.hard_max_points}")
    print(f"unique ATT&CK techniques: {summary.unique_attack_techniques:,}")
    print(f"unique CAPECs: {summary.unique_capecs:,}")
    print(f"unique CWEs: {summary.unique_cwes:,}")
    print(f"unique CVEs: {summary.unique_cves:,}")
    print(f"synthetic distractor chains per example: {summary.synthetic_distractor_chains}")
    if summary.target_prompt_tokens:
        print(f"target prompt tokens: {summary.target_prompt_tokens:,}")
        print(f"target prompt token tolerance: {summary.target_prompt_token_tolerance:,}")
        print(f"synthetic distractor batch size: {summary.synthetic_distractor_batch_size:,}")
        if summary.max_synthetic_distractor_chains:
            print(f"max synthetic distractor chains: {summary.max_synthetic_distractor_chains:,}")
    if summary.prompt_token_count_min is not None:
        print(
            "assembled prompt tokens: "
            f"min {summary.prompt_token_count_min:,}, "
            f"mean {summary.prompt_token_count_mean:,.1f}, "
            f"max {summary.prompt_token_count_max:,}"
        )
    print(f"query mode: {summary.query_mode}")
    if summary.openai_model:
        print(f"openai model: {summary.openai_model}")
    if summary.openai_base_url:
        print(f"openai base url: {summary.openai_base_url}")
    if summary.query_mode == "openai":
        print(f"openai input tokens: {summary.openai_input_tokens:,}")
        print(f"openai output tokens: {summary.openai_output_tokens:,}")
        print(f"openai estimated cost: ${summary.openai_estimated_cost_usd:.4f}")
        if summary.openai_max_cost_usd:
            print(f"openai local max cost: ${summary.openai_max_cost_usd:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
