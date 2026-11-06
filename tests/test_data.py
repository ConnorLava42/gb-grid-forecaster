"""API clients and storage, all offline: requests go to tests/fake_apis.py."""

from datetime import date

import fake_apis
import numpy as np
import pandas as pd
import responses

from gridforecast.data import elexon, weather
from gridforecast.data.store import upsert


@responses.activate
def test_elexon_client_parses_and_dedupes():
    fake_apis.register(responses.mock)
    df = elexon.fetch_demand(date(2024, 3, 30), date(2024, 4, 2), chunk_days=2)
    assert list(df.columns) == ["start_time", "settlement_date", "settlement_period", "indo_mw"]
    assert str(df["start_time"].dt.tz) == "UTC"
    assert df["start_time"].is_unique
    # 31 March 2024 is the spring clock change: 46 settlement periods that day
    assert (df["settlement_date"] == "2024-03-31").sum() == 46


@responses.activate
def test_weather_client_returns_every_city():
    fake_apis.register(responses.mock)
    long = weather.fetch_weather(date(2024, 1, 1), date(2024, 1, 3), chunk_days=2)
    assert long["city"].nunique() == 8
    assert long.groupby("city").size().eq(72).all()  # 3 days x 24 hours per city


def test_national_weather_is_population_weighted():
    times = pd.to_datetime(["2024-01-01 00:00"], utc=True)
    long = pd.DataFrame({"time": times.repeat(8), "city": list(weather.CITIES)})
    for var in weather.WEATHER_VARS:
        long[var] = 0.0
    long.loc[long["city"] == "London", "temperature_2m"] = 10.0
    nat = weather.national_weather(long)
    total = sum(w for _, _, w in weather.CITIES.values())
    expected = 10.0 * weather.CITIES["London"][2] / total
    assert np.isclose(nat["temperature_2m"].iloc[0], expected)


def test_upsert_dedupes_and_keeps_newest(tmp_path):
    path = tmp_path / "x.parquet"
    t = pd.date_range("2024-01-01", periods=4, freq="30min", tz="UTC")
    upsert(pd.DataFrame({"start_time": t[:3], "v": [1, 2, 3]}), path, "start_time")
    out = upsert(pd.DataFrame({"start_time": t[2:], "v": [30, 4]}), path, "start_time")
    assert out["start_time"].is_unique
    assert out["v"].tolist() == [1, 2, 30, 4]
