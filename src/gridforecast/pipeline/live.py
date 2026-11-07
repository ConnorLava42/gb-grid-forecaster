"""The production loop: train the model, forecast each morning, score yesterday."""

import json
from datetime import date

import pandas as pd

from gridforecast.config import FORECAST_DIR, MODEL_DIR, REPORT_DIR, UK
from gridforecast.data import weather
from gridforecast.data.clean import build_model_table
from gridforecast.data.fixtures import load_matches
from gridforecast.data.store import load
from gridforecast.evaluation import metrics
from gridforecast.evaluation.backtest import rolling_origin_backtest
from gridforecast.features.build import FEATURES, TARGET, build_origin_table, issue_time_for
from gridforecast.models.base import QCOLS
from gridforecast.models.baseline import SeasonalNaive
from gridforecast.models.ensemble import Conformal
from gridforecast.models.gbm import LGBMQuantile, LoadedLGBM
from gridforecast.models.tune import load_best_params
from gridforecast.pipeline.ingest import DEMAND, LIVE_WEATHER
from gridforecast.pipeline.tables import ORIGINS

FORECAST_LOG = FORECAST_DIR / "forecast_log.parquet"
LIVE_SCORES = REPORT_DIR / "live_scores.parquet"
LIVE_SUMMARY = REPORT_DIR / "live_summary.json"


def production_factory():
    return Conformal(lambda: LGBMQuantile(params=load_best_params()))


def train_production(origins: pd.DataFrame | None = None, run_id: str | None = None) -> dict:
    """Fit the production model (LightGBM + conformal band) on all history and save it."""
    origins = load(ORIGINS) if origins is None else origins
    rows = origins[origins[TARGET].notna()]
    model = production_factory().fit(rows[FEATURES], rows[TARGET])
    naive = SeasonalNaive().fit(rows[FEATURES], rows[TARGET])  # kept as a fallback
    extra = {
        "conformal_q": model.q_,
        "naive_offsets": naive.offsets_.to_dict(orient="index"),
        "training_end": str(rows["target_time"].max()),
        "trained_at": str(pd.Timestamp.now(tz="UTC")),
        "mlflow_run_id": run_id,
        "version": pd.Timestamp.now(tz="UTC").strftime("lgbm-%Y%m%d"),
    }
    model.model_.save(MODEL_DIR, extra)
    print(f"Saved production model to {MODEL_DIR} (conformal widening {model.q_:,.0f} MW)")
    return extra


def _fallback(origins: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Seasonal-naive forecast, used when weather is missing."""
    offsets = pd.DataFrame(meta["naive_offsets"]).T
    offsets.index = offsets.index.astype(int)
    point = origins["lag_7d"].fillna(origins["lag_14d"])
    off = offsets.reindex(origins["hh"]).to_numpy()
    return pd.DataFrame(
        {QCOLS[0]: point + off[:, 0], QCOLS[1]: point, QCOLS[2]: point + off[:, 1]},
        index=origins.index,
    )


def forecast_today(issue_day: date | None = None, force: bool = False) -> pd.DataFrame:
    """Issue today's 06:00 forecast for D+1 and D+2 and append it to the log."""
    issue_day = issue_day or pd.Timestamp.now(tz=UK).date()
    issue_time = issue_time_for(issue_day)
    log = pd.read_parquet(FORECAST_LOG) if FORECAST_LOG.exists() else pd.DataFrame()
    if not force and len(log) and (log["issue_time"] == issue_time).any():
        print(f"Forecast for {issue_day} already logged; use --force to replace it.")
        return log[log["issue_time"] == issue_time]

    demand = load(DEMAND)
    demand = demand[demand["start_time"] < issue_time]  # never use data published later
    nat = weather.national_weather(load(LIVE_WEATHER))
    table = build_model_table(demand, nat)
    origins = build_origin_table(table, [issue_day], matches=load_matches())
    if origins.empty:
        raise RuntimeError("No target rows built: is the weather forecast long enough?")

    model = LoadedLGBM(MODEL_DIR)
    if origins["temperature_2m"].isna().mean() > 0.1:
        print("Weather missing for over 10% of targets: using the seasonal-naive fallback.")
        preds, version = _fallback(origins, model.meta), "fallback_naive"
    else:
        preds, version = model.predict(origins[FEATURES]), model.meta["version"]

    out = origins[["issue_time", "target_time", "lead_day"]].join(preds)
    out["model_version"] = version
    if len(log):
        log = log[log["issue_time"] != issue_time]
    log = pd.concat([log, out], ignore_index=True).sort_values(["issue_time", "target_time"])
    FORECAST_LOG.parent.mkdir(parents=True, exist_ok=True)
    log.to_parquet(FORECAST_LOG, index=False)
    print(f"Logged {len(out)} half-hours issued {issue_time:%Y-%m-%d %H:%M} UTC ({version})")
    return out


def score_live(settle_hours: int = 3) -> dict:
    """Join the forecast log with actuals and summarise live accuracy."""
    log = pd.read_parquet(FORECAST_LOG)
    actual = load(DEMAND).set_index("start_time")["indo_mw"]
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=settle_hours)
    scored = log[log["target_time"] < cutoff].copy()
    scored[TARGET] = actual.reindex(pd.DatetimeIndex(scored["target_time"])).to_numpy()
    scored = scored.dropna(subset=[TARGET])
    scored["abs_error"] = (scored[TARGET] - scored["q50"]).abs()
    LIVE_SCORES.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(LIVE_SCORES, index=False)

    now = pd.Timestamp.now(tz=UK)
    local_target = scored["target_time"].dt.tz_convert(UK)
    windows = {
        "last_7_days": scored[local_target >= now - pd.Timedelta(days=7)],
        "last_30_days": scored[local_target >= now - pd.Timedelta(days=30)],
        "month_to_date": scored[local_target >= now.normalize().replace(day=1)],
    }
    summary = {name: (metrics.score(df) if len(df) else None) for name, df in windows.items()}
    summary |= {
        "days_live": int(log["issue_time"].nunique()),
        "first_issue": str(log["issue_time"].min()),
        "last_issue": str(log["issue_time"].max()),
        "updated": str(pd.Timestamp.now(tz="UTC")),
    }
    LIVE_SUMMARY.write_text(json.dumps(summary, indent=2))
    return summary


def retrain_monthly(tolerance: float = 0.05) -> bool:
    """Retrain, but only replace the live model if it isn't clearly worse.

    Backtests the new configuration on last month and compares it with the
    current production model on the same forecasts. Returns True if replaced.
    """
    origins = load(ORIGINS)
    this_month = pd.Timestamp.now(tz="UTC").normalize().replace(day=1)
    month_start = this_month - pd.offsets.MonthBegin(1)  # first day of last month
    month_end = this_month - pd.Timedelta(days=1)  # last day of last month

    new = rolling_origin_backtest(
        production_factory, origins, str(month_start.date()), str(month_end.date()), verbose=False
    )
    new_mae = metrics.mae(new[TARGET], new["q50"])
    decision = {"month": f"{month_start:%Y-%m}", "new_mae": new_mae}
    if (MODEL_DIR / "metadata.json").exists():
        old = LoadedLGBM(MODEL_DIR)
        test = origins[origins["issue_time"].between(month_start, month_end + pd.Timedelta(days=1))]
        old_mae = metrics.mae(test[TARGET], old.predict(test[FEATURES])["q50"])
        decision["old_mae"] = old_mae
        replace = new_mae <= old_mae * (1 + tolerance)
    else:
        replace = True
    decision["replaced"] = replace
    if replace:
        train_production(origins)
    meta_path = MODEL_DIR / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["last_retrain_decision"] = decision
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(decision)
    return replace
