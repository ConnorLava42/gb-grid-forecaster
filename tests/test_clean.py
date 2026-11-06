"""Cleaning: complete grids, gap filling and impossible values."""

import pandas as pd

from gridforecast.config import WEATHER_VARS
from gridforecast.data.clean import build_model_table, clean_demand, gap_report


def test_clean_demand_fills_single_gaps_only():
    t = pd.date_range("2024-01-01", periods=10, freq="30min", tz="UTC")
    vals = [30000.0] * 10
    df = pd.DataFrame({"start_time": t, "indo_mw": vals}).drop(index=[2, 5, 6])  # gaps of 1 and 2
    s = clean_demand(df)
    assert len(s) == 10
    assert s.iloc[2] == 30000.0  # single gap filled
    assert s.iloc[5:7].isna().all()  # two-period gap left missing
    gaps = gap_report(s)
    assert gaps["periods"].tolist() == [2]


def test_gap_report_is_empty_without_gaps():
    t = pd.date_range("2024-01-01", periods=4, freq="30min", tz="UTC")
    assert gap_report(pd.Series(1.0, index=t)).empty


def test_clean_demand_removes_implausible_values():
    t = pd.date_range("2024-01-01", periods=5, freq="30min", tz="UTC")
    df = pd.DataFrame({"start_time": t, "indo_mw": [30000, 31000, 500, 900000, 30500]})
    s = clean_demand(df)
    assert s.isna().sum() == 2  # two bad values in a row: removed, and too long a gap to fill
    assert s.max() <= 60000 and s.min() >= 10000


def test_model_table_has_a_complete_grid():
    t = pd.date_range("2024-01-01", "2024-01-03", freq="30min", tz="UTC")
    demand = pd.DataFrame({"start_time": t, "indo_mw": 30000.0})
    hours = pd.date_range("2024-01-01", "2024-01-04", freq="h", tz="UTC")
    nat = pd.DataFrame({"time": hours} | {v: 1.0 for v in WEATHER_VARS})
    table = build_model_table(demand, nat)
    assert table.index.to_series().diff().dropna().eq(pd.Timedelta("30min")).all()
    assert table["valid"].sum() == len(t)
    assert table["temperature_2m"].notna().all()
