#!/usr/bin/env python3
"""
Script 02 — Sample proxy run mixture configurations.

Reads token counts from buckets.yaml, samples N Dirichlet mixtures,
and writes them to experiments/mixtures.jsonl — one JSON line per run.

Also appends fixed reference mixtures (natural, uniform) for comparison.

Usage:
    python -m regmix.scripts.02_sample_mixtures \
        --config regmix/config/buckets.yaml \
        --experiment regmix/config/experiment.yaml \
        --output experiments/mixtures.jsonl
"""

import argparse
import json
import logging
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="regmix/config/buckets.yaml")
    parser.add_argument("--experiment", default="regmix/config/experiment.yaml")
    parser.add_argument("--output", default="experiments/mixtures.jsonl")
    parser.add_argument("--n-runs", type=int, default=None, help="Override n_runs from config")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    with open(repo_root / args.experiment) as f:
        exp_cfg = yaml.safe_load(f)

    from regmix.buckets.registry import BucketRegistry
    from regmix.mixtures.sampler import DirichletMixtureSampler

    registry = BucketRegistry(repo_root / args.config)
    proxy_cfg = exp_cfg["proxy"]

    n_runs = args.n_runs or proxy_cfg["n_runs"]
    alpha_scale = proxy_cfg["alpha_scale"]
    min_weight = proxy_cfg["min_bucket_weight"]
    token_budget = proxy_cfg["token_budget"]
    seed = args.seed or 42

    sampler = DirichletMixtureSampler(
        bucket_names=registry.names(),
        token_counts=registry.token_counts(),
        alpha_scale=alpha_scale,
        min_weight=min_weight,
        seed=seed,
    )

    mixtures = sampler.sample(n_runs)

    out_path = repo_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for i, mix in enumerate(mixtures):
        records.append({
            "mixture_id": f"m{i:04d}",
            "type": "sampled",
            "weights": mix.weights,
            "token_budget": token_budget,
        })

    # Add reference mixtures
    records.append({
        "mixture_id": "ref_natural",
        "type": "reference",
        "weights": registry.natural_weights(),
        "token_budget": token_budget,
    })
    records.append({
        "mixture_id": "ref_uniform",
        "type": "reference",
        "weights": registry.uniform_weights(),
        "token_budget": token_budget,
    })

    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    logger.info(f"Wrote {len(records)} mixture configs to {out_path}")
    logger.info(f"  {n_runs} sampled + 2 reference mixtures")
    logger.info(f"  Token budget per run: {token_budget:,}")

    # Print a few examples
    logger.info("\nExample sampled mixtures:")
    for rec in records[:3]:
        weights_str = "  ".join(f"{k}: {v:.3f}" for k, v in rec["weights"].items())
        logger.info(f"  [{rec['mixture_id']}]  {weights_str}")


if __name__ == "__main__":
    main()
