"""Turn raw downloads into one clean, gap-aware half-hourly table."""

import pandas as pd

from gridforecast.config import WEATHER_VARS

MIN_PLAUSIBLE_MW, MAX_PLAUSIBLE_MW = 10_000, 60_000


def half_hour_grid(start, end) -> pd.DatetimeIndex:
    """Every half-hour between start and end, in UTC.

    Indexing on UTC means every day has 48 periods. Settlement days have 46
    or 50 on clock-change days, which is why we never index on them.
    """
    return pd.date_range(start, end, freq="30min", tz="UTC", name="start_time")


def clean_demand(demand: pd.DataFrame) -> pd.Series:
    """INDO on a complete grid: impossible values removed, single gaps filled."""
    s = demand.set_index("start_time")["indo_mw"].astype(float)
    s = s[~s.index.duplicated(keep="last")]
    grid = half_hour_grid(s.index.min(), s.index.max())
    s = s.reindex(grid)
    implausible = (s < MIN_PLAUSIBLE_MW) | (s > MAX_PLAUSIBLE_MW)
    if implausible.any():
        print(f"Removing {implausible.sum()} implausible demand values")
    s[implausible] = float("nan")
    # Fill isolated single missing periods; leave longer gaps as NaN.
    # (interpolate(limit=1) alone would also fill the first half-hour of long gaps.)
    missing = s.isna()
    run_length = missing.groupby((missing != missing.shift()).cumsum()).transform("size")
    single = missing & (run_length == 1)
    s[single] = s.interpolate(limit_area="inside")[single]
    return s


def gap_report(s: pd.Series) -> pd.DataFrame:
    """Each run of consecutive missing half-hours: where it starts and how long it is."""
    missing = s.isna()
    run_id = (missing != missing.shift()).cumsum()  # a new id every time missing flips
    runs = pd.DataFrame(
        {
            "start": s.index.to_series().groupby(run_id).first(),
            "periods": missing.groupby(run_id).size(),
            "is_gap": missing.groupby(run_id).first(),
        }
    )
    return runs.loc[runs["is_gap"], ["start", "periods"]].reset_index(drop=True)


def upsample_weather(nat: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Hourly national weather interpolated onto the half-hour grid."""
    w = nat.set_index("time")[WEATHER_VARS].sort_index()
    w = w[~w.index.duplicated(keep="last")]
    both = w.index.union(grid)
    w = w.reindex(both).interpolate(method="time", limit=2, limit_area="inside")
    return w.reindex(grid)


def build_model_table(demand: pd.DataFrame, weather_nat: pd.DataFrame) -> pd.DataFrame:
    """One row per half-hour: demand, a `valid` flag and national weather."""
    indo = clean_demand(demand)
    # Extend the grid a few days past the last actual so forecast-day weather fits
    end = max(indo.index.max(), weather_nat["time"].max())
    grid = half_hour_grid(indo.index.min(), end)
    table = upsample_weather(weather_nat, grid)
    table.insert(0, "indo_mw", indo.reindex(grid))
    table.insert(1, "valid", table["indo_mw"].notna())
    return table
