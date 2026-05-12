#!/usr/bin/env python3
"""
Script 04 — Fit regression models and evaluate quality.

Loads all proxy run results, converts them to regression rows,
optionally cross-validates, then fits the final models and saves them.

Outputs:
    experiments/regression/model.pkl          — fitted MixtureRegressor
    experiments/regression/metrics.json       — evaluation metrics
    experiments/regression/importances.json   — feature importances

Usage:
    python -m regmix.scripts.04_fit_regression \
        --experiment regmix/config/experiment.yaml \
        --results-dir experiments/results \
        --output-dir experiments/regression
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="regmix/config/buckets.yaml")
    parser.add_argument("--experiment", default="regmix/config/experiment.yaml")
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--output-dir", default="experiments/regression")
    parser.add_argument("--target", default="y_composite",
                        help="Regression target column (see RegressionRow)")
    parser.add_argument("--cross-validate", action="store_true")
    parser.add_argument("--general-baseline", type=float, default=None,
                        help="Base model general language loss (overrides auto-detection)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(repo_root / args.experiment) as f:
        exp_cfg = yaml.safe_load(f)

    from regmix.buckets.registry import BucketRegistry
    from regmix.experiments.tracker import ExperimentTracker
    from regmix.regression.evaluator import (
        cross_validate_regression,
        evaluate_regression,
        print_metrics,
    )
    from regmix.regression.models import MixtureRegressor

    registry = BucketRegistry(repo_root / args.config)
    tracker = ExperimentTracker(repo_root / args.results_dir)

    obj_cfg = exp_cfg.get("objective", {})
    lambda_penalty = obj_cfg.get("lambda_general_penalty", 0.5)
    general_baseline = args.general_baseline or obj_cfg.get("general_baseline", 3.0)

    runs = tracker.load_all()
    if len(runs) < 10:
        logger.warning(f"Only {len(runs)} runs found — regression may be unreliable")
    logger.info(f"Loaded {len(runs)} proxy run results")

    rows = tracker.to_regression_rows(
        bucket_names=registry.names(),
        lambda_penalty=lambda_penalty,
        general_baseline=general_baseline,
    )

    # ------------------------------------------------------------------
    # Cross-validation (optional but recommended)
    # ------------------------------------------------------------------
    if args.cross_validate:
        logger.info("\nCross-validating regression quality...")
        cv_metrics = cross_validate_regression(
            rows=rows,
            bucket_names=registry.names(),
            target=args.target,
            n_folds=5,
        )
        print_metrics(cv_metrics, title="5-fold cross-validation")
        with open(out_dir / "cv_metrics.json", "w") as f:
            json.dump(cv_metrics, f, indent=2)

    # ------------------------------------------------------------------
    # Fit final model on all data
    # ------------------------------------------------------------------
    logger.info(f"\nFitting final model on {len(rows)} runs, target={args.target}")
    regressor = MixtureRegressor(registry.names())
    regressor.fit(rows, target=args.target)

    # Evaluate on training data (sanity check)
    train_metrics = evaluate_regression(regressor, rows, target=args.target)
    print_metrics(train_metrics, title="Train-set metrics (sanity check)")

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    model_path = out_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(regressor, f)
    logger.info(f"Model saved to {model_path}")

    # ------------------------------------------------------------------
    # Save metrics and importances
    # ------------------------------------------------------------------
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(train_metrics, f, indent=2)

    importances = regressor.feature_importance_table()
    with open(out_dir / "importances.json", "w") as f:
        json.dump(importances, f, indent=2)

    logger.info("\nFeature importances (LightGBM or Ridge):")
    for name, imp in importances:
        bar = "█" * int(imp / max(v for _, v in importances) * 30)
        logger.info(f"  {name:<30} {bar}  {imp:.1f}")

    # ------------------------------------------------------------------
    # Also save a readable summary DataFrame
    # ------------------------------------------------------------------
    df = tracker.to_dataframe()
    df.to_csv(out_dir / "runs_summary.csv", index=False)
    logger.info(f"\nRun summary CSV → {out_dir / 'runs_summary.csv'}")


if __name__ == "__main__":
    main()
