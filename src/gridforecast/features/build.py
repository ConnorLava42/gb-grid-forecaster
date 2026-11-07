"""The forecast-origin table: one row per (forecast issue, target half-hour).

A forecast is issued at 06:00 UK time on day D for every half-hour of D+1
and D+2. Each row only uses demand that was published before 06:00 on D,
so training rows look exactly like the rows the live model will see.
"""

from datetime import date

import numpy as np
import pandas as pd

from gridforecast.config import HEATING_BASE_C, ISSUE_HOUR, LEAD_DAYS, UK
from gridforecast.data.calendar import bank_holidays
from gridforecast.data.fixtures import match_features

LAG_FEATURES = ["lag_d1_same_tod", "lag_7d", "lag_14d", "mean_d1", "mean_7d", "last_obs"]
CALENDAR_FEATURES = [
    "lead_day",
    "hh",
    "dow",
    "is_weekend",
    "is_holiday",
    "is_bridge",
    "is_xmas",
    "doy_sin",
    "doy_cos",
    "is_bst",
]
WEATHER_FEATURES = [
    "temperature_2m",
    "apparent_temperature",
    "heating_degrees",
    "temp_24h_mean",
    "temp_72h_mean",
    "shortwave_radiation",
    "cloud_cover",
    "wind_speed_10m",
    "precipitation",
]
MATCH_FEATURES = ["match_phase", "match_tier"]
FEATURES = LAG_FEATURES + CALENDAR_FEATURES + WEATHER_FEATURES + MATCH_FEATURES
TARGET = "indo_mw"


def issue_time_for(day) -> pd.Timestamp:
    """The UTC moment of the 06:00 UK issue on a given local date."""
    local = pd.Timestamp(day) + pd.Timedelta(hours=ISSUE_HOUR)
    return local.tz_localize(UK).tz_convert("UTC")


def _lookup(series: pd.Series, times) -> np.ndarray:
    """Values of `series` at `times`, NaN where a time isn't in the index."""
    return series.reindex(pd.DatetimeIndex(times)).to_numpy()


def _target_rows(index: pd.DatetimeIndex, issue_dates) -> pd.DataFrame:
    """Every (issue date, target time) pair, for lead days 1 and 2."""
    local_date = index.tz_convert(UK).normalize().tz_localize(None)
    frames = []
    for lead in LEAD_DAYS:
        frames.append(
            pd.DataFrame(
                {
                    "target_time": index,
                    "lead_day": lead,
                    "issue_date": local_date - pd.Timedelta(days=lead),
                }
            )
        )
    rows = pd.concat(frames, ignore_index=True)
    if issue_dates is not None:
        wanted = pd.to_datetime(pd.Index(issue_dates))
        rows = rows[rows["issue_date"].isin(wanted)]
    return rows.reset_index(drop=True)


def _calendar(target: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features from the target's UK local time."""
    local = target.tz_convert(UK)
    days = local.normalize().tz_localize(None)
    hols = bank_holidays(range(days.year.min() - 1, days.year.max() + 2))
    hol_days = pd.DatetimeIndex(sorted(hols))

    is_holiday = days.isin(hol_days)
    dow = local.day_of_week
    weekend = dow >= 5
    prev_off = (days - pd.Timedelta(days=1)).isin(hol_days) | ((dow - 1) % 7 >= 5)
    next_off = (days + pd.Timedelta(days=1)).isin(hol_days) | ((dow + 1) % 7 >= 5)
    # A "bridge" is a lone working day squeezed between a holiday and a weekend
    is_bridge = ~weekend & ~is_holiday & prev_off & next_off
    md = local.month * 100 + local.day
    doy = local.day_of_year
    utc_offset = local.tz_localize(None) - target.tz_convert(None)  # 0 in GMT, 1 hour in BST

    return pd.DataFrame(
        {
            "hh": local.hour * 2 + local.minute // 30,
            "dow": dow,
            "is_weekend": weekend.astype(int),
            "is_holiday": is_holiday.astype(int),
            "is_bridge": is_bridge.astype(int),
            "is_xmas": ((md >= 1224) | (md <= 102)).astype(int),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "is_bst": (utc_offset > pd.Timedelta(0)).astype(int),
        }
    )


def _weather_table(table: pd.DataFrame) -> pd.DataFrame:
    """Weather at each half-hour, plus heating degrees and recent averages."""
    w = table[
        [
            "temperature_2m",
            "apparent_temperature",
            "shortwave_radiation",
            "cloud_cover",
            "wind_speed_10m",
            "precipitation",
        ]
    ].copy()
    w["heating_degrees"] = (HEATING_BASE_C - w["temperature_2m"]).clip(lower=0)
    # Buildings respond to the last day or three of temperature, not just this hour
    w["temp_24h_mean"] = w["temperature_2m"].rolling(48, min_periods=24).mean()
    w["temp_72h_mean"] = w["temperature_2m"].rolling(144, min_periods=72).mean()
    return w


def build_origin_table(
    table: pd.DataFrame, issue_dates=None, matches: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Build features for every forecast issued on `issue_dates`.

    If `issue_dates` is None, every date with a full history and full targets is used.

    `table` is the output of data.clean.build_model_table: a UTC half-hourly
    index with `indo_mw`, `valid` and national weather columns.
    """
    demand = table["indo_mw"]
    if issue_dates is None:
        issue_dates = all_issue_dates(table)
    rows = _target_rows(table.index, issue_dates)
    rows["issue_time"] = (
        (rows["issue_date"] + pd.Timedelta(hours=ISSUE_HOUR))
        .dt.tz_localize(UK)
        .dt.tz_convert("UTC")
    )
    target = pd.DatetimeIndex(rows["target_time"])
    day = pd.Timedelta(days=1)

    # --- Lags, all anchored so they are published before the issue time ---
    rows["lag_7d"] = _lookup(demand, target - 7 * day)
    rows["lag_14d"] = _lookup(demand, target - 14 * day)
    # Same time of day on D-1: two days back for D+1 targets, three for D+2
    back = pd.to_timedelta(rows["lead_day"] + 1, unit="D")
    rows["lag_d1_same_tod"] = _lookup(demand, target - pd.TimedeltaIndex(back))
    # Latest published period: the one starting an hour before the issue time
    rows["last_obs"] = _lookup(demand, pd.DatetimeIndex(rows["issue_time"]) - pd.Timedelta(hours=1))

    local_day = demand.index.tz_convert(UK).normalize().tz_localize(None)
    daily = demand.groupby(local_day).mean()
    rows["mean_d1"] = daily.reindex(rows["issue_date"] - day).to_numpy()
    mean_7d = daily.rolling(7, min_periods=5).mean()  # days D-7 ... D-1
    rows["mean_7d"] = mean_7d.reindex(rows["issue_date"] - day).to_numpy()

    # --- Calendar and weather at the target time ---
    rows = pd.concat([rows, _calendar(target)], axis=1)
    weather = _weather_table(table)
    for col in WEATHER_FEATURES:
        rows[col] = _lookup(weather[col], target)

    # --- Football ---
    if matches is not None and len(matches):
        mf = match_features(pd.DatetimeIndex(target.unique()), matches)
        rows = rows.join(mf, on="target_time")
    else:
        rows["match_phase"] = 0
        rows["match_tier"] = 0

    # --- Target (NaN for future half-hours in a live forecast) ---
    rows[TARGET] = _lookup(demand.where(table["valid"]), target)

    cols = ["issue_time", "target_time"] + FEATURES + [TARGET]
    return rows[cols].sort_values(["issue_time", "target_time"]).reset_index(drop=True)


def all_issue_dates(table: pd.DataFrame, warmup_days: int = 15) -> list[date]:
    """Issue dates with 15 days of history before them and both target days after."""
    local = table.index.tz_convert(UK)
    first = local.min().normalize() + pd.Timedelta(days=warmup_days)
    last = local.max().normalize() - pd.Timedelta(days=3)
    return [
        d.date() for d in pd.date_range(first.tz_localize(None), last.tz_localize(None), freq="D")
    ]
