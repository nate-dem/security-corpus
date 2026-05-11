from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from regmix.experiments.schemas import ProxyRunResult, RegressionRow


class ExperimentTracker:
    """
    Persists proxy run results as one JSON file per run.

    Layout:
        results_dir/
            <run_id>.json
            ...
    """

    def __init__(self, results_dir: str | Path = "experiments/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, result: ProxyRunResult) -> Path:
        path = self.results_dir / f"{result.run_id}.json"
        path.write_text(result.model_dump_json(indent=2))
        return path

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_all(self) -> list[ProxyRunResult]:
        results = []
        for f in sorted(self.results_dir.glob("*.json")):
            results.append(ProxyRunResult.model_validate_json(f.read_text()))
        return results

    def load(self, run_id: str) -> ProxyRunResult:
        path = self.results_dir / f"{run_id}.json"
        return ProxyRunResult.model_validate_json(path.read_text())

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Flat DataFrame with one row per proxy run for EDA and regression."""
        records = []
        for r in self.load_all():
            row: dict = {
                "run_id": r.run_id,
                "timestamp": r.timestamp,
                "token_budget": r.token_budget,
            }
            for bucket, w in r.mixture.weights.items():
                row[f"w_{bucket}"] = w
            row.update({f"loss_{k}": v for k, v in r.validation_losses.as_dict().items()})
            row.update({f"bench_{k}": v for k, v in r.benchmark_scores.items()})
            records.append(row)
        return pd.DataFrame(records)

    def to_regression_rows(
        self,
        bucket_names: list[str],
        lambda_penalty: float,
        general_baseline: float,
    ) -> list[RegressionRow]:
        return [
            RegressionRow.from_run(r, bucket_names, lambda_penalty, general_baseline)
            for r in self.load_all()
        ]

    def summary(self) -> str:
        runs = self.load_all()
        if not runs:
            return "No runs recorded yet."
        df = self.to_dataframe()
        cyber_col = "loss_cyber_mean"
        lines = [
            f"Runs: {len(runs)}",
            f"Cyber mean loss — min: {df[cyber_col].min():.4f}  "
            f"max: {df[cyber_col].max():.4f}  "
            f"mean: {df[cyber_col].mean():.4f}",
        ]
        return "\n".join(lines)
