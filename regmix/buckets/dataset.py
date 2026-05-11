from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq


class WeightedBucketDataset:
    """
    Streams tokenized examples from multiple data buckets with per-bucket
    sampling weights.  Each bucket directory must contain one or more
    Parquet files with a column named 'input_ids' (list<int32>).

    Sampling is done by rejection: pick a bucket according to `weights`,
    then draw one example from that bucket's iterator.  Buckets are
    cycled when exhausted.

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
        self.bucket_paths = bucket_paths
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}
        self.token_budget = token_budget
        self.rng = random.Random(seed)

        self._buckets = list(self.weights.keys())
        self._cum_weights = []
        running = 0.0
        for name in self._buckets:
            running += self.weights[name]
            self._cum_weights.append(running)

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

    def __iter__(self) -> Iterator[dict]:
        iterators = {name: self._iter_bucket(self.bucket_paths[name]) for name in self._buckets}
        tokens_seen = 0

        while tokens_seen < self.token_budget:
            # weighted bucket selection
            r = self.rng.random()
            chosen = self._buckets[-1]
            for name, cw in zip(self._buckets, self._cum_weights):
                if r <= cw:
                    chosen = name
                    break

            ids = next(iterators[chosen])
            tokens_seen += len(ids)
            yield {"input_ids": ids, "bucket": chosen}
