"""Choose LightGBM settings with Optuna, using the tuning year only."""

import json

import lightgbm as lgb
import optuna
import pandas as pd

from gridforecast.config import REPORT_DIR, TUNING_END, TUNING_START
from gridforecast.features.build import FEATURES, TARGET
from gridforecast.models.gbm import DEFAULT_PARAMS

BEST_PARAMS = REPORT_DIR / "best_params.json"


def tune_lightgbm(origins: pd.DataFrame, n_trials: int = 30, seed: int = 42) -> dict:
    """Search a few LightGBM settings to minimise median (q50) error on the tuning year.

    Train: every row whose target is before the tuning year.
    Validate: forecasts issued during the tuning year.
    The test year is never looked at, so the backtest stays honest.
    """
    start = pd.Timestamp(TUNING_START, tz="UTC")
    end = pd.Timestamp(TUNING_END, tz="UTC") + pd.Timedelta(days=1)
    rows = origins[origins[TARGET].notna()]
    train = rows[rows["target_time"] < start]
    val = rows[(rows["issue_time"] >= start) & (rows["issue_time"] < end)]

    def objective(trial: optuna.Trial) -> float:
        params = DEFAULT_PARAMS | {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=0.5, n_estimators=3000, random_state=seed, **params
        )
        model.fit(
            train[FEATURES],
            train[TARGET],
            eval_X=val[FEATURES],
            eval_y=val[TARGET],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        return float((val[TARGET] - model.predict(val[FEATURES])).abs().mean())

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    BEST_PARAMS.parent.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS.write_text(json.dumps(study.best_params, indent=2))
    print(f"Best tuning-year MAE {study.best_value:,.0f} MW with {study.best_params}")
    return study.best_params


def load_best_params() -> dict:
    """The frozen tuned settings, or {} (defaults) if you haven't tuned yet."""
    return json.loads(BEST_PARAMS.read_text()) if BEST_PARAMS.exists() else {}
