from __future__ import annotations

from typing import Optional

import numpy as np

from regmix.experiments.schemas import MixtureWeights


class DirichletMixtureSampler:
    """
    Samples data mixture weight vectors from a Dirichlet distribution.

    The concentration parameter alpha_i for bucket i is set proportional to
    that bucket's share of total tokens, scaled by `alpha_scale * n_buckets`.
    This biases the prior toward the natural token distribution while still
    exploring the full simplex.

    Setting alpha_scale < 1 increases diversity (sparser mixtures).
    Setting alpha_scale > 1 concentrates samples near the natural weights.

    Args:
        bucket_names:   ordered list of bucket names
        token_counts:   {bucket_name: int}
        alpha_scale:    Dirichlet concentration scale factor
        min_weight:     clip all weights to at least this value, then renormalize
        seed:           RNG seed (None = non-deterministic)
    """

    def __init__(
        self,
        bucket_names: list[str],
        token_counts: dict[str, int],
        alpha_scale: float = 1.0,
        min_weight: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.bucket_names = bucket_names
        self.min_weight = min_weight
        self.rng = np.random.default_rng(seed)

        total = sum(token_counts[n] for n in bucket_names)
        if total == 0:
            shares = np.ones(len(bucket_names)) / len(bucket_names)
        else:
            shares = np.array([token_counts[n] / total for n in bucket_names])

        self.alpha = shares * alpha_scale * len(bucket_names)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n: int = 1) -> list[MixtureWeights]:
        """Draw n mixtures from the informed Dirichlet prior."""
        raw = self.rng.dirichlet(self.alpha, size=n)
        return self._postprocess(raw)

    def sample_uniform(self, n: int = 1) -> list[MixtureWeights]:
        """Draw n mixtures from a flat Dirichlet (alpha=1 for all buckets)."""
        raw = self.rng.dirichlet(np.ones(len(self.bucket_names)), size=n)
        return self._postprocess(raw)

    def _postprocess(self, raw: np.ndarray) -> list[MixtureWeights]:
        if self.min_weight > 0:
            raw = np.clip(raw, self.min_weight, 1.0)
            raw /= raw.sum(axis=1, keepdims=True)
        return [
            MixtureWeights(weights={n: float(w) for n, w in zip(self.bucket_names, row)})
            for row in raw
        ]

    # ------------------------------------------------------------------
    # Fixed reference mixtures
    # ------------------------------------------------------------------

    def natural_mixture(self, token_counts: dict[str, int]) -> MixtureWeights:
        total = sum(token_counts[n] for n in self.bucket_names)
        if total == 0:
            return self.uniform_mixture()
        return MixtureWeights(
            weights={n: token_counts[n] / total for n in self.bucket_names}
        )

    def uniform_mixture(self) -> MixtureWeights:
        n = len(self.bucket_names)
        return MixtureWeights(weights={name: 1.0 / n for name in self.bucket_names})
