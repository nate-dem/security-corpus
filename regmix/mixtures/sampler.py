from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import numpy as np

from regmix.experiments.schemas import MixtureWeights


class DirichletMixtureSampler:
    """
    Samples data mixture weight vectors from a Dirichlet distribution.

    REGMIX samples alpha = lambda * natural_token_distribution, where lambda
    is swept across a small range to produce both sparse and near-natural
    mixtures. Optional per-bucket floors reserve fixed mass before sampling
    the remaining simplex, and optional max-usage constraints reject mixtures
    that would require excessive repetition from small buckets.

    Smaller concentration values increase diversity (sparser mixtures).
    Larger concentration values concentrate samples near the natural weights.

    Args:
        bucket_names:   ordered list of bucket names
        token_counts:   {bucket_name: int}
        concentrations: Dirichlet concentration values to sweep
        bucket_floors:  optional {bucket_name: minimum mixture weight}
        token_budget:   proxy-run token budget, required for max_usage
        max_usage:      optional max repeat factor per bucket, as a scalar or dict
        seed:           RNG seed (None = non-deterministic)
    """

    def __init__(
        self,
        bucket_names: list[str],
        token_counts: dict[str, int],
        concentrations: list[float] | None = None,
        bucket_floors: dict[str, float] | None = None,
        token_budget: int | None = None,
        max_usage: float | dict[str, float] | None = None,
        alpha_scale: float | None = None,
        min_weight: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.bucket_names = bucket_names
        self.token_counts = {name: int(token_counts.get(name, 0)) for name in bucket_names}
        self.token_budget = token_budget
        self.max_usage = max_usage
        self.rng = np.random.default_rng(seed)

        if concentrations is None:
            concentrations = [alpha_scale if alpha_scale is not None else 1.0]
        self.concentrations = [float(c) for c in concentrations]
        if not self.concentrations or any(c <= 0 for c in self.concentrations):
            raise ValueError("Dirichlet concentrations must be positive")

        self.bucket_floors = self._validate_floors(bucket_floors, min_weight)
        self._floor_mass = sum(self.bucket_floors.values())
        if self._floor_mass >= 1.0:
            raise ValueError(f"Bucket floors must sum to < 1, got {self._floor_mass:.4f}")

        self._eligible_buckets = [name for name in bucket_names if self.token_counts[name] > 0]
        if not self._eligible_buckets:
            raise ValueError(
                "All bucket token counts are zero. Run 01_prepare_buckets.py before sampling."
            )
        for name, floor in self.bucket_floors.items():
            if floor > 0 and self.token_counts[name] <= 0:
                raise ValueError(f"Bucket {name!r} has a floor but zero tokens")

        total = sum(self.token_counts[n] for n in self._eligible_buckets)
        self._base_distribution = np.array(
            [self.token_counts[n] / total for n in self._eligible_buckets],
            dtype=float,
        )
        self._check_floor_feasibility()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n: int = 1, concentration: float | None = None) -> list[MixtureWeights]:
        """Draw n mixtures from the informed Dirichlet prior."""
        mixtures: list[MixtureWeights] = []
        for _ in range(n):
            value = float(concentration) if concentration is not None else self._choose_concentration()
            mixtures.append(self._sample_one(value))
        return mixtures

    def sample_uniform(self, n: int = 1) -> list[MixtureWeights]:
        """Draw n mixtures from a flat Dirichlet (alpha=1 for all buckets)."""
        mixtures: list[MixtureWeights] = []
        for _ in range(n):
            mixtures.append(self._sample_one_uniform())
        return mixtures

    def _sample_one(self, concentration: float) -> MixtureWeights:
        alpha = self._base_distribution * concentration
        return self._sample_with_rejection(alpha)

    def _sample_one_uniform(self) -> MixtureWeights:
        alpha = np.ones(len(self._eligible_buckets), dtype=float)
        return self._sample_with_rejection(alpha)

    def _sample_with_rejection(
        self,
        alpha: np.ndarray,
        max_attempts: int = 1_000,
    ) -> MixtureWeights:
        for _ in range(max_attempts):
            draw = self.rng.dirichlet(alpha)
            weights = self._apply_floors(draw)
            mixture = MixtureWeights(weights=weights)
            if self._within_usage_limits(mixture):
                return mixture
        raise ValueError(
            "Could not sample a mixture satisfying max_usage. "
            "Relax max_usage, lower bucket floors, or increase bucket token counts."
        )

    def _apply_floors(self, draw: np.ndarray) -> dict[str, float]:
        remaining_mass = 1.0 - self._floor_mass
        weights = {name: self.bucket_floors.get(name, 0.0) for name in self.bucket_names}
        for name, sampled_weight in zip(self._eligible_buckets, draw):
            weights[name] += remaining_mass * float(sampled_weight)

        total = sum(weights.values())
        return {name: value / total for name, value in weights.items()}

    def _choose_concentration(self) -> float:
        idx = int(self.rng.integers(0, len(self.concentrations)))
        return self.concentrations[idx]

    def _validate_floors(
        self,
        bucket_floors: dict[str, float] | None,
        min_weight: float,
    ) -> dict[str, float]:
        if bucket_floors is None:
            bucket_floors = {}
        if min_weight > 0:
            # Backward compatibility for old configs. New configs should use
            # explicit bucket_floors so the corpus policy is visible per bucket.
            bucket_floors = {name: min_weight for name in self.bucket_names} | bucket_floors

        unknown = set(bucket_floors) - set(self.bucket_names)
        if unknown:
            raise ValueError(f"Unknown bucket floor keys: {sorted(unknown)}")

        floors = {name: float(bucket_floors.get(name, 0.0)) for name in self.bucket_names}
        negative = {name: value for name, value in floors.items() if value < 0}
        if negative:
            raise ValueError(f"Bucket floors must be non-negative: {negative}")
        return floors

    def _usage_limit_for(self, bucket_name: str) -> float | None:
        if self.max_usage is None:
            return None
        if isinstance(self.max_usage, Mapping):
            value = self.max_usage.get(bucket_name)
            return None if value is None else float(value)
        return float(self.max_usage)

    def _check_floor_feasibility(self) -> None:
        if self.token_budget is None or self.max_usage is None:
            return
        for name, floor in self.bucket_floors.items():
            limit = self._usage_limit_for(name)
            if limit is None:
                continue
            allowed_tokens = self.token_counts[name] * limit
            requested_tokens = floor * self.token_budget
            if requested_tokens > allowed_tokens:
                raise ValueError(
                    f"Floor for {name!r} requests {requested_tokens:,.0f} tokens, "
                    f"but max_usage allows only {allowed_tokens:,.0f}"
                )

    def _within_usage_limits(self, mixture: MixtureWeights) -> bool:
        if self.token_budget is None or self.max_usage is None:
            return True
        for name, weight in mixture.weights.items():
            limit = self._usage_limit_for(name)
            if limit is None:
                continue
            requested_tokens = weight * self.token_budget
            allowed_tokens = self.token_counts[name] * limit
            if requested_tokens > allowed_tokens:
                return False
        return True

    # ------------------------------------------------------------------
    # Fixed reference mixtures
    # ------------------------------------------------------------------

    def natural_mixture(self, token_counts: dict[str, int]) -> MixtureWeights:
        total = sum(token_counts[n] for n in self.bucket_names)
        if total == 0:
            raise ValueError("Cannot build a natural mixture when all token counts are zero")
        return MixtureWeights(
            weights={n: token_counts[n] / total for n in self.bucket_names}
        )

    def uniform_mixture(self) -> MixtureWeights:
        remaining_mass = 1.0 - self._floor_mass
        weights = {name: self.bucket_floors.get(name, 0.0) for name in self.bucket_names}
        active_count = len(self._eligible_buckets)
        for name in self._eligible_buckets:
            weights[name] += remaining_mass / active_count
        return MixtureWeights(weights=weights)
