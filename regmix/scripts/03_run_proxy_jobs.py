#!/usr/bin/env python3
"""
Script 03 — Execute proxy midtraining jobs.

Reads experiments/mixtures.jsonl, launches one proxy training run per entry,
and saves a ProxyRunResult JSON for each to experiments/results/.

Can be run sequentially (default) or dispatched to a SLURM/job array
by setting --job-index and --total-jobs for a specific shard.

Usage:
    # Sequential (small PoC on one machine):
    python -m regmix.scripts.03_run_proxy_jobs

    # One SLURM array task (--array=0-63 in sbatch):
    python -m regmix.scripts.03_run_proxy_jobs --job-index $SLURM_ARRAY_TASK_ID --total-jobs 64

    # Dry run (skip training, write dummy losses — useful for pipeline testing):
    python -m regmix.scripts.03_run_proxy_jobs --dry-run
"""

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def make_dummy_losses(mixture_weights: dict[str, float]) -> dict[str, float]:
    """
    Synthetic losses for dry-run mode.
    Introduces a weak but detectable signal so regression can be sanity-checked.
    """
    rng = random.Random(sum(v * i for i, v in enumerate(mixture_weights.values())))
    base = 2.8
    cyber_signal = -0.3 * mixture_weights.get("mitre_cve", 0)
    cyber_signal -= 0.2 * mixture_weights.get("sigma_rules", 0)
    general_signal = 0.1 * (1.0 - mixture_weights.get("general_technical", 0))
    noise = lambda: rng.gauss(0, 0.04)  # noqa: E731
    return {
        "cyber_general": base + cyber_signal + noise(),
        "cloud_security": base + cyber_signal * 0.8 + noise(),
        "task_specific": base + cyber_signal * 1.2 + noise(),
        "general_language": base + general_signal + noise(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="regmix/config/buckets.yaml")
    parser.add_argument("--experiment", default="regmix/config/experiment.yaml")
    parser.add_argument("--mixtures", default="experiments/mixtures.jsonl")
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--job-index", type=int, default=None)
    parser.add_argument("--total-jobs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    with open(repo_root / args.experiment) as f:
        exp_cfg = yaml.safe_load(f)

    from regmix.buckets.registry import BucketRegistry
    from regmix.experiments.schemas import (
        MixtureWeights,
        ProxyRunResult,
        TrainingConfig,
        ValidationLosses,
    )
    from regmix.experiments.tracker import ExperimentTracker

    registry = BucketRegistry(repo_root / args.config)
    tracker = ExperimentTracker(repo_root / args.results_dir)
    train_cfg = exp_cfg["training"]
    val_cfg = exp_cfg["validation"]

    # Load all mixture configs
    with open(repo_root / args.mixtures) as f:
        all_mixtures = [json.loads(line) for line in f if line.strip()]

    # Shard for job arrays
    if args.job_index is not None and args.total_jobs is not None:
        all_mixtures = all_mixtures[args.job_index :: args.total_jobs]
        logger.info(f"Job {args.job_index}/{args.total_jobs}: {len(all_mixtures)} mixtures")

    # Build paths
    bucket_paths = {
        name: repo_root / bucket.path
        for name, bucket in registry.buckets.items()
    }
    validation_paths = {
        k: repo_root / v
        for k, v in val_cfg.items()
        if k != "tokens_per_set"
    }

    training_config = TrainingConfig(
        base_model=train_cfg["base_model"],
        learning_rate=train_cfg["learning_rate"],
        batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        max_steps=-1,   # token_budget controls termination
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        seed=train_cfg["seed"],
        bf16=train_cfg.get("bf16", True),
    )

    already_done = {r.run_id for r in tracker.load_all()}

    for entry in all_mixtures:
        run_id = entry["mixture_id"]
        if run_id in already_done:
            logger.info(f"[{run_id}] already complete, skipping")
            continue

        mixture = MixtureWeights(weights=entry["weights"])
        token_budget = entry["token_budget"]

        logger.info(f"\n[{run_id}] Starting  budget={token_budget:,}")

        if args.dry_run:
            val_losses_raw = make_dummy_losses(entry["weights"])
        else:
            from regmix.training.proxy_runner import run_proxy_job
            val_losses_raw = run_proxy_job(
                run_id=run_id,
                mixture_weights=entry["weights"],
                bucket_paths=bucket_paths,
                validation_paths=validation_paths,
                base_model=train_cfg["base_model"],
                token_budget=token_budget,
                training_cfg=train_cfg,
                keep_checkpoint=args.keep_checkpoints,
            )

        result = ProxyRunResult(
            run_id=run_id,
            timestamp=datetime.utcnow(),
            mixture=mixture,
            token_budget=token_budget,
            training_config=training_config,
            validation_losses=ValidationLosses(**val_losses_raw),
            metadata={"type": entry.get("type", "sampled"), "dry_run": args.dry_run},
        )
        saved_path = tracker.save(result)
        logger.info(f"[{run_id}] Saved → {saved_path}")

    logger.info(f"\nDone. Total results: {len(tracker.load_all())}")


if __name__ == "__main__":
    main()
