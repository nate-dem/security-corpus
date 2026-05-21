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
    lgbm_best_iteration: Optional[int] = None
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
        final_refit: bool = True,
        validation_fraction: float = 0.2,
        early_stopping_rounds: int = 50,
    ) -> "MixtureRegressor":
        X = _to_X(rows)
        y = _to_y(rows, target)
        self.target = target

        # Ridge baseline
        ridge_cv = min(5, len(rows)) if len(rows) >= 2 else None
        self._ridge = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(
                alphas=np.logspace(-3, 3, 7),
                cv=ridge_cv,
                scoring="neg_mean_squared_error" if ridge_cv else None,
            )),
        ])
        self._ridge.fit(X, y)

        # LightGBM
        best_iteration = None
        if _LGBM_AVAILABLE:
            defaults = dict(
                n_estimators=1000,
                learning_rate=0.01,
                random_state=42,
                verbose=-1,
            )
            if lgbm_params:
                defaults.update(lgbm_params)
            if len(rows) >= 5 and validation_fraction > 0 and early_stopping_rounds > 0:
                n_val = max(1, int(round(len(rows) * validation_fraction)))
                n_val = min(n_val, len(rows) - 1)
                rng = np.random.default_rng(42)
                indices = rng.permutation(len(rows))
                val_idx = indices[:n_val]
                train_idx = indices[n_val:]
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                tuned = lgb.LGBMRegressor(**defaults)
                tuned.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(-1),
                    ],
                )
                best_iteration = getattr(tuned, "best_iteration_", None) or defaults["n_estimators"]
                best_iteration = max(1, int(best_iteration))

                if final_refit:
                    final_params = {**defaults, "n_estimators": best_iteration}
                    self._lgbm = lgb.LGBMRegressor(**final_params)
                    self._lgbm.fit(X, y)
                else:
                    self._lgbm = tuned
            else:
                self._lgbm = lgb.LGBMRegressor(**defaults)
                self._lgbm.fit(X, y)
                best_iteration = defaults["n_estimators"]

        ridge_model = self._ridge.named_steps["ridge"]
        scaler = self._ridge.named_steps["scaler"]
        coef_original = ridge_model.coef_ / scaler.scale_
        intercept_original = ridge_model.intercept_ - np.dot(coef_original, scaler.mean_)

        self._fit_result = FitResult(
            target=target,
            ridge_coef=coef_original,
            ridge_intercept=float(intercept_original),
            lgbm_importances=(
                self._lgbm.feature_importances_ if _LGBM_AVAILABLE and self._lgbm else None
            ),
            lgbm_best_iteration=(
                int(best_iteration) if best_iteration is not None else None
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
