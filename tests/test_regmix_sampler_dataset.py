from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from regmix.buckets.dataset import WeightedBucketDataset
from regmix.mixtures.sampler import DirichletMixtureSampler


def test_dirichlet_sampler_applies_floors_and_zero_token_buckets():
    sampler = DirichletMixtureSampler(
        bucket_names=["small", "large", "empty"],
        token_counts={"small": 100, "large": 900, "empty": 0},
        concentrations=[1.0],
        bucket_floors={"small": 0.10},
        token_budget=1_000,
        max_usage={"small": 2.0, "large": 2.0},
        seed=7,
    )

    mixtures = sampler.sample(20)

    for mixture in mixtures:
        assert mixture.weights["small"] >= 0.10
        assert mixture.weights["empty"] == 0.0
        assert mixture.weights["small"] * 1_000 <= 200
        assert pytest.approx(sum(mixture.weights.values())) == 1.0


def test_dirichlet_sampler_rejects_impossible_floor_with_max_usage():
    with pytest.raises(ValueError, match="Floor for 'small'"):
        DirichletMixtureSampler(
            bucket_names=["small", "large"],
            token_counts={"small": 100, "large": 900},
            concentrations=[1.0],
            bucket_floors={"small": 0.50},
            token_budget=1_000,
            max_usage={"small": 2.0},
        )


def test_weighted_bucket_dataset_enforces_token_quotas(tmp_path: Path):
    bucket_a = tmp_path / "a"
    bucket_b = tmp_path / "b"
    bucket_a.mkdir()
    bucket_b.mkdir()
    pq.write_table(pa.table({"input_ids": [[1] * 7, [1] * 7]}), bucket_a / "a.parquet")
    pq.write_table(pa.table({"input_ids": [[2] * 7, [2] * 7, [2] * 7]}), bucket_b / "b.parquet")

    dataset = WeightedBucketDataset(
        bucket_paths={"a": bucket_a, "b": bucket_b},
        weights={"a": 0.25, "b": 0.75},
        token_budget=20,
        seed=1,
    )

    tokens_by_bucket = {"a": 0, "b": 0}
    for example in dataset:
        tokens_by_bucket[example["bucket"]] += len(example["input_ids"])

    assert dataset.token_quotas == {"a": 5, "b": 15}
    assert tokens_by_bucket == {"a": 5, "b": 15}
