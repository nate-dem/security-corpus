from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from regmix.experiments.schemas import RegressionRow
from regmix.regression.models import MixtureRegressor


def evaluate_regression(
    regressor: MixtureRegressor,
    rows: list[RegressionRow],
    target: str = "y_composite",
    top_k: int = 10,
    model: str = "lgbm",
) -> dict:
    """
    Compute regression quality metrics on a held-out set of proxy runs.

    Returns:
        mse             — mean squared error
        rmse            — root mean squared error
        mae             — mean absolute error
        spearman_rho    — Spearman rank correlation
        spearman_p      — p-value for rank correlation
        top_k_overlap   — fraction of true top-k in predicted top-k
        n               — number of evaluation examples
    """
    y_true = np.array([getattr(r, target) for r in rows])
    y_pred = regressor.predict_rows(rows, model=model)

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rho, pval = spearmanr(y_true, y_pred)

    # top-k overlap (key always uses the requested top_k for stable cross-fold aggregation)
    k_actual = min(top_k, len(rows))
    true_topk = set(np.argsort(y_true)[:k_actual])
    pred_topk = set(np.argsort(y_pred)[:k_actual])
    overlap = len(true_topk & pred_topk) / k_actual

    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "spearman_rho": float(rho),
        "spearman_p": float(pval),
        f"top_{top_k}_overlap": overlap,
        "n": len(rows),
    }


def cross_validate_regression(
    rows: list[RegressionRow],
    bucket_names: list[str],
    target: str = "y_composite",
    n_folds: int = 5,
    model: str = "lgbm",
) -> dict:
    """
    K-fold cross-validation of regression quality.

    Returns mean ± std for each metric.
    """
    from regmix.regression.models import MixtureRegressor

    indices = np.arange(len(rows))
    np.random.default_rng(42).shuffle(indices)
    folds = np.array_split(indices, n_folds)

    fold_metrics: list[dict] = []
    for i, val_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        train_rows = [rows[j] for j in train_idx]
        val_rows = [rows[j] for j in val_idx]

        reg = MixtureRegressor(bucket_names)
        reg.fit(train_rows, target=target)
        metrics = evaluate_regression(reg, val_rows, target=target, model=model)
        fold_metrics.append(metrics)

    keys = list(fold_metrics[0].keys())
    summary = {}
    for k in keys:
        if k == "n":
            summary[k] = int(np.sum([m[k] for m in fold_metrics]))
            continue
        vals = [m[k] for m in fold_metrics]
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals))

    return summary


def print_metrics(metrics: dict, title: str = "Regression metrics") -> None:
    print(f"\n{title}")
    print("-" * 40)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<30} {v:.4f}")
        else:
            print(f"  {k:<30} {v}")
