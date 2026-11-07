"""Rung 2: LightGBM, one model per quantile."""

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from gridforecast.config import QUANTILES
from gridforecast.models.base import QCOLS, sort_quantiles, time_split

DEFAULT_PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 100,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "reg_lambda": 1.0,
    "verbose": -1,
}


class LGBMQuantile:
    """Three LightGBM models (q10, q50, q90) trained with the quantile objective.

    fit() first trains with early stopping on the latest 8 weeks to find the
    right number of trees, then retrains on all the data with that many
    trees, so the most recent weeks are never thrown away.

    residual=True predicts `demand - lag_7d` instead of demand itself.
    """

    name = "lightgbm"

    def __init__(
        self,
        params: dict | None = None,
        max_trees: int = 3000,
        early_stopping: int = 100,
        residual: bool = False,
        seed: int = 42,
    ):
        self.params = DEFAULT_PARAMS | (params or {})
        self.max_trees = max_trees
        self.early_stopping = early_stopping
        self.residual = residual
        self.seed = seed

    def _target(self, X, y):
        return y - X["lag_7d"] if self.residual else y

    def fit(self, X: pd.DataFrame, y: pd.Series):
        keep = self._target(X, y).notna()
        X, y = X[keep], self._target(X, y)[keep]
        X_fit, y_fit, X_val, y_val = time_split(X, y)
        self.models_, self.best_iterations_ = {}, {}
        for q, col in zip(QUANTILES, QCOLS, strict=True):
            probe = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=self.max_trees,
                random_state=self.seed,
                **self.params,
            )
            probe.fit(
                X_fit,
                y_fit,
                eval_X=X_val,
                eval_y=y_val,
                callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)],
            )
            best = max(probe.best_iteration_ or self.max_trees, 50)
            final = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=best,
                random_state=self.seed,
                **self.params,
            )
            self.models_[col] = final.fit(X, y)
            self.best_iterations_[col] = best
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = pd.DataFrame({col: m.predict(X) for col, m in self.models_.items()}, index=X.index)
        if self.residual:
            preds = preds.add(X["lag_7d"], axis=0)
        return sort_quantiles(preds)

    # --- Saving and loading, used by the daily job ---
    def save(self, folder: Path, extra: dict | None = None):
        folder.mkdir(parents=True, exist_ok=True)
        for col, m in self.models_.items():
            m.booster_.save_model(folder / f"lgbm_{col}.txt")
        meta = {
            "residual": self.residual,
            "best_iterations": self.best_iterations_,
            "features": list(next(iter(self.models_.values())).feature_name_),
        }
        (folder / "metadata.json").write_text(
            json.dumps(meta | (extra or {}), indent=2, default=str)
        )


class LoadedLGBM:
    """A saved LGBMQuantile, read back from text files (no training needed)."""

    def __init__(self, folder: Path):
        self.meta = json.loads((folder / "metadata.json").read_text())
        self.boosters = {
            col: lgb.Booster(model_file=str(folder / f"lgbm_{col}.txt")) for col in QCOLS
        }
        self.features = self.meta["features"]
        self.conformal_q = self.meta.get("conformal_q", 0.0)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.features]
        preds = pd.DataFrame({col: b.predict(X) for col, b in self.boosters.items()}, index=X.index)
        if self.meta["residual"]:
            preds = preds.add(X["lag_7d"], axis=0)
        preds = sort_quantiles(preds)
        preds[QCOLS[0]] -= self.conformal_q  # conformal widening, see ensemble.Conformal
        preds[QCOLS[-1]] += self.conformal_q
        return preds
