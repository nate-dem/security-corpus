from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------


class MixtureWeights(BaseModel):
    """Simplex vector over named data buckets. Values must sum to ~1."""

    weights: dict[str, float]

    @model_validator(mode="after")
    def _check_simplex(self) -> "MixtureWeights":
        total = sum(self.weights.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Mixture weights must sum to 1, got {total:.4f}")
        return self

    def as_array(self, bucket_names: list[str]) -> list[float]:
        return [self.weights[n] for n in bucket_names]


class TrainingConfig(BaseModel):
    base_model: str
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    warmup_ratio: float
    weight_decay: float
    seed: int
    bf16: bool = True


class ValidationLosses(BaseModel):
    cyber_general: float
    cloud_security: float
    task_specific: float
    general_language: float

    def cyber_mean(self) -> float:
        return (self.cyber_general + self.cloud_security + self.task_specific) / 3.0

    def composite(self, lambda_penalty: float, general_baseline: float) -> float:
        """
        Primary objective for regression target.

        L = mean(cyber losses) + lambda * max(0, L_general - L_general_baseline)

        The penalty is zero when general language loss stays at or below the
        base model's pre-midtraining level.
        """
        degradation = max(0.0, self.general_language - general_baseline)
        return self.cyber_mean() + lambda_penalty * degradation

    def as_dict(self) -> dict[str, float]:
        return {
            "cyber_general": self.cyber_general,
            "cloud_security": self.cloud_security,
            "task_specific": self.task_specific,
            "general_language": self.general_language,
            "cyber_mean": self.cyber_mean(),
        }


# ---------------------------------------------------------------------------
# Top-level run record
# ---------------------------------------------------------------------------


class ProxyRunResult(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    mixture: MixtureWeights
    token_budget: int
    training_config: TrainingConfig
    validation_losses: ValidationLosses

    # composite score filled in after fitting the objective (not stored on write)
    composite_loss: Optional[float] = None

    # optional downstream benchmarks (e.g. CyberSecEval, MMLU subset)
    benchmark_scores: dict[str, float] = Field(default_factory=dict)

    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Regression dataset row (derived from runs)
# ---------------------------------------------------------------------------


class RegressionRow(BaseModel):
    run_id: str
    bucket_names: list[str]
    x: list[float]           # mixture weights in bucket_names order
    y_cyber_general: float
    y_cloud_security: float
    y_task_specific: float
    y_general_language: float
    y_composite: float

    @classmethod
    def from_run(
        cls,
        run: ProxyRunResult,
        bucket_names: list[str],
        lambda_penalty: float,
        general_baseline: float,
    ) -> "RegressionRow":
        return cls(
            run_id=run.run_id,
            bucket_names=bucket_names,
            x=run.mixture.as_array(bucket_names),
            y_cyber_general=run.validation_losses.cyber_general,
            y_cloud_security=run.validation_losses.cloud_security,
            y_task_specific=run.validation_losses.task_specific,
            y_general_language=run.validation_losses.general_language,
            y_composite=run.validation_losses.composite(lambda_penalty, general_baseline),
        )
