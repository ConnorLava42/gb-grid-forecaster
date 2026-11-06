"""Elexon Insights API: GB demand outturn (our target) and NESO's own forecast."""

import time
from datetime import date, timedelta

import pandas as pd

from gridforecast.data.http import get_json

BASE = "https://data.elexon.co.uk/bmrs/api/v1"


def _date_chunks(start: date, end: date, days: int):
    """Split [start, end] into consecutive windows of at most `days` days."""
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=days - 1), end)
        yield cur, stop
        cur = stop + timedelta(days=1)


def fetch_demand(start: date, end: date, chunk_days: int = 14) -> pd.DataFrame:
    """Initial National Demand Outturn (INDO), half-hourly, in MW.

    Returns one row per settlement period with a UTC `start_time`.
    """
    frames = []
    for lo, hi in _date_chunks(start, end, chunk_days):
        payload = get_json(
            f"{BASE}/demand/outturn",
            {
                "settlementDateFrom": lo.isoformat(),
                "settlementDateTo": hi.isoformat(),
                "format": "json",
            },
        )
        frames.append(pd.DataFrame(payload["data"]))
        time.sleep(0.5)  # be polite to a free service
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return pd.DataFrame(
            columns=["start_time", "settlement_date", "settlement_period", "indo_mw"]
        )

    df["start_time"] = pd.to_datetime(df["startTime"], utc=True)
    df = df.rename(
        columns={
            "settlementDate": "settlement_date",
            "settlementPeriod": "settlement_period",
            "initialDemandOutturn": "indo_mw",
        }
    )
    return (
        df[["start_time", "settlement_date", "settlement_period", "indo_mw"]]
        .dropna(subset=["indo_mw"])
        .drop_duplicates("start_time", keep="last")
        .sort_values("start_time")
        .reset_index(drop=True)
    )


def fetch_neso_forecast(start: date, end: date, chunk_days: int = 7) -> pd.DataFrame:
    """NESO's day-ahead National Demand Forecast (NDF), used only as a benchmark.

    Uses the "earliest" view: the first forecast NESO published for each
    half-hour, which is the fair comparison for our day-ahead forecast.
    Parameter and field names are taken from the Elexon developer portal;
    if Elexon changes them, this function raises a clear error.
    """
    frames = []
    for lo, hi in _date_chunks(start, end, chunk_days):
        payload = get_json(
            f"{BASE}/forecast/demand/day-ahead/earliest",
            {"from": lo.isoformat(), "to": hi.isoformat(), "format": "json"},
        )
        rows = payload["data"] if isinstance(payload, dict) else payload
        frames.append(pd.DataFrame(rows))
        time.sleep(0.5)
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return pd.DataFrame(columns=["start_time", "neso_mw"])

    value_col = next((c for c in ("nationalDemand", "demand") if c in df), None)
    if value_col is None or "startTime" not in df:
        raise KeyError(
            f"Unexpected NDF columns {list(df.columns)}. Check the endpoint on "
            "developer.data.elexon.co.uk and update fetch_neso_forecast."
        )
    df["start_time"] = pd.to_datetime(df["startTime"], utc=True)
    df = df.rename(columns={value_col: "neso_mw"})
    return (
        df[["start_time", "neso_mw"]]
        .drop_duplicates("start_time", keep="first")
        .sort_values("start_time")
        .reset_index(drop=True)
    )


def fetch_fuelinst(start: str, end: str) -> pd.DataFrame:
    """5-minute generation by fuel type (FUELINST), summed to a GB total.

    Only used for the optional sharp-resolution look at TV pickups.
    `start` and `end` are ISO datetimes, e.g. "2024-07-10T18:00Z"; keep the
    window to a day or two per call.
    """
    payload = get_json(
        f"{BASE}/datasets/FUELINST",
        {"publishDateTimeFrom": start, "publishDateTimeTo": end, "format": "json"},
    )
    df = pd.DataFrame(payload["data"])
    df["start_time"] = pd.to_datetime(df["startTime"], utc=True)
    total = df.groupby("start_time")["generation"].sum().rename("generation_mw")
    return total.reset_index()
