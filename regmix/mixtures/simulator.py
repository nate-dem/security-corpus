from __future__ import annotations

import numpy as np

from regmix.experiments.schemas import MixtureWeights
from regmix.mixtures.sampler import DirichletMixtureSampler


class MixtureSimulator:
    """
    Simulates a large candidate pool of mixtures and uses a fitted regressor
    to predict validation loss for each, then selects the best.

    Args:
        sampler:        DirichletMixtureSampler for generating candidates
        n_simulations:  total candidate mixtures to evaluate
        batch_size:     how many to generate/predict at a time (memory control)
    """

    def __init__(
        self,
        sampler: DirichletMixtureSampler,
        n_simulations: int = 100_000,
        batch_size: int = 10_000,
    ):
        self.sampler = sampler
        self.n_simulations = n_simulations
        self.batch_size = batch_size

    def run(self, predict_fn) -> tuple[list[MixtureWeights], np.ndarray]:
        """
        Generate all candidates and score them.

        Args:
            predict_fn: callable(list[MixtureWeights]) -> np.ndarray of loss predictions

        Returns:
            (all_mixtures, all_predictions)
        """
        all_mixtures: list[MixtureWeights] = []
        all_preds: list[np.ndarray] = []

        remaining = self.n_simulations
        while remaining > 0:
            n = min(self.batch_size, remaining)
            batch = self.sampler.sample(n)
            preds = predict_fn(batch)
            all_mixtures.extend(batch)
            all_preds.append(preds)
            remaining -= n

        return all_mixtures, np.concatenate(all_preds)

    # ------------------------------------------------------------------
    # Selection strategies
    # ------------------------------------------------------------------

    def select(
        self,
        mixtures: list[MixtureWeights],
        predictions: np.ndarray,
        top_k: int = 10,
        strategy: str = "average_topk",
    ) -> MixtureWeights:
        """
        Select a final mixture from the simulated pool.

        Strategies:
          best          — return the single mixture with lowest predicted loss
          average_topk  — average the top-k mixtures and renormalize (more robust)
        """
        top_idx = np.argsort(predictions)[:top_k]
        top_mixtures = [mixtures[i] for i in top_idx]

        if strategy == "best":
            return top_mixtures[0]

        if strategy == "average_topk":
            bucket_names = list(top_mixtures[0].weights.keys())
            avg = {
                name: float(np.mean([m.weights[name] for m in top_mixtures]))
                for name in bucket_names
            }
            total = sum(avg.values())
            return MixtureWeights(weights={k: v / total for k, v in avg.items()})

        raise ValueError(f"Unknown selection strategy: {strategy!r}")

    def top_k_summary(
        self,
        mixtures: list[MixtureWeights],
        predictions: np.ndarray,
        k: int = 10,
    ) -> list[dict]:
        """Return a human-readable summary of the top-k predicted mixtures."""
        top_idx = np.argsort(predictions)[:k]
        rows = []
        for rank, idx in enumerate(top_idx):
            row = {"rank": rank + 1, "predicted_loss": float(predictions[idx])}
            row.update(mixtures[idx].weights)
            rows.append(row)
        return rows
