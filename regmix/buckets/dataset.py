from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

try:
    from torch.utils.data import IterableDataset
except ImportError:  # pragma: no cover - torch is an optional training dependency
    class IterableDataset:  # type: ignore[no-redef]
        pass


class WeightedBucketDataset(IterableDataset):
    """
    Streams tokenized examples from multiple data buckets with per-run
    token quotas.  Each bucket directory must contain one or more
    Parquet files with a column named 'input_ids' (list<int32>).

    A mixture weight of 0.10 means about 10% of the proxy run's token budget
    comes from that bucket. Buckets are cycled when exhausted, so small-bucket
    repetition is controlled upstream by the mixture sampler's max_usage guard.

    Args:
        bucket_paths:   {bucket_name: Path}
        weights:        {bucket_name: float}  — need not sum to 1
        token_budget:   stop after yielding this many tokens total
        seed:           RNG seed for reproducibility
    """

    def __init__(
        self,
        bucket_paths: dict[str, Path],
        weights: dict[str, float],
        token_budget: int,
        seed: int = 42,
    ):
        super().__init__()
        self.bucket_paths = bucket_paths
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Mixture weights must sum to a positive value")
        self.weights = {k: v / total for k, v in weights.items()}
        self.token_budget = token_budget
        self.rng = random.Random(seed)

        self._buckets = list(self.weights.keys())
        self._quotas = self._compute_token_quotas()

    def _iter_bucket(self, path: Path) -> Iterator[list[int]]:
        """Yields input_id lists from all parquet files in `path`, looping."""
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files in {path}")
        while True:
            self.rng.shuffle(files)
            for fpath in files:
                table = pq.read_table(fpath, columns=["input_ids"])
                indices = list(range(table.num_rows))
                self.rng.shuffle(indices)
                for i in indices:
                    yield table["input_ids"][i].as_py()

    def _compute_token_quotas(self) -> dict[str, int]:
        exact = {name: self.weights[name] * self.token_budget for name in self._buckets}
        quotas = {name: int(math.floor(value)) for name, value in exact.items()}

        remainder = self.token_budget - sum(quotas.values())
        by_fraction = sorted(
            self._buckets,
            key=lambda name: exact[name] - quotas[name],
            reverse=True,
        )
        for name in by_fraction[:remainder]:
            quotas[name] += 1
        return quotas

    @property
    def token_quotas(self) -> dict[str, int]:
        return dict(self._quotas)

    def __iter__(self) -> Iterator[dict]:
        remaining = dict(self._quotas)
        iterators = {
            name: self._iter_bucket(self.bucket_paths[name])
            for name, quota in remaining.items()
            if quota > 0
        }

        while any(tokens > 0 for tokens in remaining.values()):
            chosen = self._choose_bucket_with_remaining_quota(remaining)
            ids = next(iterators[chosen])
            quota = remaining[chosen]
            if len(ids) > quota:
                ids = ids[:quota]
            remaining[chosen] -= len(ids)
            yield {"input_ids": ids, "bucket": chosen}

    def _choose_bucket_with_remaining_quota(self, remaining: dict[str, int]) -> str:
        active = [(name, tokens) for name, tokens in remaining.items() if tokens > 0]
        total = sum(tokens for _, tokens in active)
        r = self.rng.random() * total
        running = 0.0
        for name, tokens in active:
            running += tokens
            if r <= running:
                return name
        return active[-1][0]
