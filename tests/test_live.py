"""The daily job, end to end and offline: train, forecast, log once, score."""

import fake_apis
import pandas as pd
import pytest
from conftest import make_table

from gridforecast.config import UK
from gridforecast.data import weather
from gridforecast.features.build import build_origin_table
from gridforecast.models.ensemble import Conformal
from gridforecast.models.gbm import LGBMQuantile
from gridforecast.pipeline import live


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point every file the live pipeline touches at a temporary folder."""
    for name in ("DEMAND", "LIVE_WEATHER", "FORECAST_LOG", "LIVE_SCORES"):
        monkeypatch.setattr(live, name, tmp_path / f"{name.lower()}.parquet")
    monkeypatch.setattr(live, "LIVE_SUMMARY", tmp_path / "live_summary.json")
    monkeypatch.setattr(live, "MODEL_DIR", tmp_path / "model")
    monkeypatch.setattr(
        live, "production_factory", lambda: Conformal(lambda: LGBMQuantile(max_trees=60))
    )

    # 30 days of synthetic demand up to now, and a live weather file around today
    now = pd.Timestamp.now(tz="UTC").floor("30min")
    t = pd.date_range(now - pd.Timedelta(days=30), now - pd.Timedelta(hours=1), freq="30min")
    pd.DataFrame({"start_time": t, "indo_mw": fake_apis.synthetic_demand(t)}).to_parquet(
        live.DEMAND
    )
    hours = pd.date_range(now.normalize() - pd.Timedelta(days=30), periods=34 * 24, freq="h")
    weather.parse_response(fake_apis._weather_payload(hours)).to_parquet(live.LIVE_WEATHER)
    return tmp_path


def test_daily_job_forecasts_once_and_scores(workspace):
    live.train_production(build_origin_table(make_table("2024-01-01", "2024-06-30")))
    assert (live.MODEL_DIR / "lgbm_q50.txt").exists()

    issue_day = (pd.Timestamp.now(tz=UK) - pd.Timedelta(days=3)).date()
    out = live.forecast_today(issue_day)
    assert len(out) in (94, 96, 98)  # 96, or 94/98 around a clock change
    assert out["model_version"].str.startswith("lgbm-").all()

    live.forecast_today(issue_day)  # a second run the same day must not duplicate
    assert len(pd.read_parquet(live.FORECAST_LOG)) == len(out)

    summary = live.score_live()
    assert summary["days_live"] == 1
    assert summary["last_7_days"]["n"] > 0
    assert summary["last_7_days"]["mape"] < 25  # sane, not a unit or alignment bug


def test_fallback_when_weather_is_missing(workspace):
    live.train_production(build_origin_table(make_table("2024-01-01", "2024-06-30")))
    w = pd.read_parquet(live.LIVE_WEATHER)
    w.loc[w["time"] > pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2), "temperature_2m"] = None
    w.to_parquet(live.LIVE_WEATHER)

    out = live.forecast_today((pd.Timestamp.now(tz=UK) - pd.Timedelta(days=1)).date())
    assert (out["model_version"] == "fallback_naive").all()
    assert out[["q10", "q50", "q90"]].notna().all().all()
