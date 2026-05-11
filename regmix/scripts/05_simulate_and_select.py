#!/usr/bin/env python3
"""
Script 05 — Simulate mixture candidates and select the best for target training.

Loads the fitted regressor, samples N candidate mixtures, predicts validation
loss for each, and selects the best using the configured strategy.

Outputs:
    experiments/selected_mixture.json   — chosen mixture weights
    experiments/top_k_summary.json      — top-k predicted mixtures
    experiments/simulation_stats.json   — distribution stats of predictions

Usage:
    python -m regmix.scripts.05_simulate_and_select \
        --experiment regmix/config/experiment.yaml \
        --model experiments/regression/model.pkl \
        --output-dir experiments
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="regmix/config/buckets.yaml")
    parser.add_argument("--experiment", default="regmix/config/experiment.yaml")
    parser.add_argument("--model", default="experiments/regression/model.pkl")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--n-simulations", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--strategy", default=None, choices=["best", "average_topk"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / args.output_dir

    with open(repo_root / args.experiment) as f:
        exp_cfg = yaml.safe_load(f)

    sim_cfg = exp_cfg["simulation"]
    proxy_cfg = exp_cfg["proxy"]

    n_simulations = args.n_simulations or sim_cfg["n_candidates"]
    top_k = args.top_k or sim_cfg["top_k"]
    strategy = args.strategy or sim_cfg["selection_strategy"]
    seed = args.seed if args.seed is not None else sim_cfg["random_seed"]

    from regmix.buckets.registry import BucketRegistry
    from regmix.mixtures.sampler import DirichletMixtureSampler
    from regmix.mixtures.simulator import MixtureSimulator

    registry = BucketRegistry(repo_root / args.config)

    with open(repo_root / args.model, "rb") as f:
        regressor = pickle.load(f)

    sampler = DirichletMixtureSampler(
        bucket_names=registry.names(),
        token_counts=registry.token_counts(),
        alpha_scale=proxy_cfg["alpha_scale"],
        min_weight=proxy_cfg["min_bucket_weight"],
        seed=seed,
    )

    simulator = MixtureSimulator(
        sampler=sampler,
        n_simulations=n_simulations,
    )

    logger.info(f"Simulating {n_simulations:,} candidate mixtures...")

    def predict_fn(mixtures):
        return regressor.predict(mixtures, model="lgbm")

    all_mixtures, all_preds = simulator.run(predict_fn)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    selected = simulator.select(all_mixtures, all_preds, top_k=top_k, strategy=strategy)
    top_k_rows = simulator.top_k_summary(all_mixtures, all_preds, k=top_k)

    # ------------------------------------------------------------------
    # Also compute reference mixture predictions for comparison
    # ------------------------------------------------------------------
    reference_mixtures = {
        "natural": sampler.natural_mixture(registry.token_counts()),
        "uniform": sampler.uniform_mixture(),
    }
    reference_preds = {
        name: float(regressor.predict([mix], model="lgbm")[0])
        for name, mix in reference_mixtures.items()
    }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    stats = {
        "n_simulations": n_simulations,
        "pred_min": float(all_preds.min()),
        "pred_max": float(all_preds.max()),
        "pred_mean": float(all_preds.mean()),
        "pred_p25": float(np.percentile(all_preds, 25)),
        "pred_p50": float(np.percentile(all_preds, 50)),
        "pred_p75": float(np.percentile(all_preds, 75)),
        "selected_pred_loss": float(regressor.predict([selected], model="lgbm")[0]),
        "selection_strategy": strategy,
        "top_k": top_k,
        "reference_preds": reference_preds,
    }

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    selected_path = out_dir / "selected_mixture.json"
    with open(selected_path, "w") as f:
        json.dump(
            {
                "weights": selected.weights,
                "predicted_loss": stats["selected_pred_loss"],
                "strategy": strategy,
                "top_k": top_k,
                "n_simulations": n_simulations,
            },
            f, indent=2,
        )

    with open(out_dir / "top_k_summary.json", "w") as f:
        json.dump(top_k_rows, f, indent=2)

    with open(out_dir / "simulation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*55}")
    logger.info("Selected mixture:")
    for name, w in sorted(selected.weights.items(), key=lambda x: -x[1]):
        bar = "█" * int(w * 40)
        logger.info(f"  {name:<30} {w:.3f}  {bar}")

    logger.info(f"\nPredicted composite loss: {stats['selected_pred_loss']:.4f}")
    logger.info("Reference predictions:")
    for name, loss in reference_preds.items():
        logger.info(f"  {name:<20} {loss:.4f}")

    logger.info(f"\nSaved → {selected_path}")


if __name__ == "__main__":
    main()
