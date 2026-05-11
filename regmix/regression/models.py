from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

from regmix.experiments.schemas import MixtureWeights, RegressionRow


def _to_X(rows: list[RegressionRow]) -> "pd.DataFrame":
    import pandas as pd
    return pd.DataFrame([r.x for r in rows], columns=rows[0].bucket_names)


def _to_y(rows: list[RegressionRow], target: str) -> np.ndarray:
    return np.array([getattr(r, target) for r in rows])


@dataclass
class FitResult:
    target: str
    ridge_coef: np.ndarray
    ridge_intercept: float
    lgbm_importances: Optional[np.ndarray] = None
    metrics: dict = field(default_factory=dict)  # populated by evaluator


class MixtureRegressor:
    """
    Fits mixture weights -> validation loss.

    Maintains one Ridge and one LightGBM model per target variable so callers
    can predict individual losses or the composite objective.

    Usage:
        reg = MixtureRegressor(bucket_names)
        reg.fit(rows, target="y_composite")
        preds = reg.predict(candidate_mixtures)
    """

    def __init__(self, bucket_names: list[str]):
        self.bucket_names = bucket_names
        self._ridge: Optional[Pipeline] = None
        self._lgbm = None
        self.target: Optional[str] = None
        self._fit_result: Optional[FitResult] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        rows: list[RegressionRow],
        target: str = "y_composite",
        lgbm_params: Optional[dict] = None,
    ) -> "MixtureRegressor":
        X = _to_X(rows)
        y = _to_y(rows, target)
        self.target = target

        # Ridge baseline
        self._ridge = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)),
        ])
        self._ridge.fit(X, y)

        # LightGBM
        if _LGBM_AVAILABLE:
            defaults = dict(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=max(3, len(rows) // 10),
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
            )
            if lgbm_params:
                defaults.update(lgbm_params)
            self._lgbm = lgb.LGBMRegressor(**defaults)
            # Use all data; at small n (< 64) early stopping hurts more than helps
            n_val = max(1, len(rows) // 5)
            X_tr, X_val = X[:-n_val], X[-n_val:]
            y_tr, y_val = y[:-n_val], y[-n_val:]
            self._lgbm.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )

        ridge_model = self._ridge.named_steps["ridge"]
        scaler = self._ridge.named_steps["scaler"]
        coef_original = ridge_model.coef_ / scaler.scale_

        self._fit_result = FitResult(
            target=target,
            ridge_coef=coef_original,
            ridge_intercept=float(ridge_model.intercept_),
            lgbm_importances=(
                self._lgbm.feature_importances_ if _LGBM_AVAILABLE and self._lgbm else None
            ),
        )
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        mixtures: list[MixtureWeights],
        model: str = "lgbm",
    ) -> np.ndarray:
        import pandas as pd
        X = pd.DataFrame(
            [[m.weights[n] for n in self.bucket_names] for m in mixtures],
            columns=self.bucket_names,
        )
        if model == "lgbm":
            if not _LGBM_AVAILABLE or self._lgbm is None:
                raise RuntimeError("LightGBM not available; use model='ridge'")
            return self._lgbm.predict(X)
        if model == "ridge":
            return self._ridge.predict(X)
        raise ValueError(f"Unknown model: {model!r}")

    def predict_rows(
        self,
        rows: list[RegressionRow],
        model: str = "lgbm",
    ) -> np.ndarray:
        import pandas as pd
        X = pd.DataFrame([r.x for r in rows], columns=rows[0].bucket_names)
        if model == "lgbm":
            if not _LGBM_AVAILABLE or self._lgbm is None:
                raise RuntimeError("LightGBM not available; use model='ridge'")
            return self._lgbm.predict(X)
        if model == "ridge":
            return self._ridge.predict(X)
        raise ValueError(f"Unknown model: {model!r}")

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance_table(self) -> list[tuple[str, float]]:
        if self._fit_result is None or self._fit_result.lgbm_importances is None:
            # Fall back to ridge coefficients (absolute value = rough importance)
            imps = np.abs(self._fit_result.ridge_coef)
        else:
            imps = self._fit_result.lgbm_importances
        pairs = sorted(zip(self.bucket_names, imps.tolist()), key=lambda x: -x[1])
        return pairs
