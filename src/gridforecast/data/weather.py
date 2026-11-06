"""Open-Meteo weather for the population centres in config.CITIES.

Three sources share one response format:
- "historical_forecast": archived forecasts (from 2021). Use for training and backtests.
- "archive": ERA5 reanalysis, i.e. what actually happened. Used for the oracle experiment.
- "forecast": the live forecast. Used by the daily job.
"""

import time
from datetime import date, timedelta

import pandas as pd

from gridforecast.config import CITIES, WEATHER_VARS
from gridforecast.data.http import get_json

URLS = {
    "historical_forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
    "archive": "https://archive-api.open-meteo.com/v1/archive",
    "forecast": "https://api.open-meteo.com/v1/forecast",
}


def _base_params() -> dict:
    lats, lons = zip(*[(lat, lon) for lat, lon, _ in CITIES.values()], strict=True)
    return {
        "latitude": ",".join(str(x) for x in lats),
        "longitude": ",".join(str(x) for x in lons),
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "GMT",  # timestamps in UTC, matching Elexon
        "wind_speed_unit": "ms",
    }


def parse_response(payload) -> pd.DataFrame:
    """Turn an Open-Meteo response into long format: one row per city per hour.

    With several locations Open-Meteo returns a list in the same order as the
    request, so the list position tells us which city each block belongs to.
    """
    blocks = payload if isinstance(payload, list) else [payload]
    frames = []
    for city, block in zip(CITIES, blocks, strict=True):
        hourly = pd.DataFrame(block["hourly"])
        hourly["time"] = pd.to_datetime(hourly["time"]).dt.tz_localize("UTC")
        hourly["city"] = city
        frames.append(hourly)
    return pd.concat(frames, ignore_index=True)


def fetch_weather(
    start: date, end: date, source: str = "historical_forecast", chunk_days: int = 90
) -> pd.DataFrame:
    """Hourly weather for every city between start and end (inclusive)."""
    frames, cur = [], start
    while cur <= end:
        stop = min(cur + timedelta(days=chunk_days - 1), end)
        params = _base_params() | {"start_date": cur.isoformat(), "end_date": stop.isoformat()}
        frames.append(parse_response(get_json(URLS[source], params)))
        cur = stop + timedelta(days=1)
        time.sleep(1)  # long multi-city requests count as several API calls
    return pd.concat(frames, ignore_index=True)


def fetch_live_weather(past_days: int = 16, forecast_days: int = 3) -> pd.DataFrame:
    """The latest forecast plus the recent past, in one request."""
    params = _base_params() | {"past_days": past_days, "forecast_days": forecast_days}
    return parse_response(get_json(URLS["forecast"], params))


def national_weather(long: pd.DataFrame) -> pd.DataFrame:
    """Population-weighted GB average of each variable, one row per hour."""
    weights = pd.Series({city: w for city, (_, _, w) in CITIES.items()}, name="w")
    df = long.join(weights, on="city")
    out = {}
    for var in WEATHER_VARS:
        valid = df[var].notna()
        num = (df.loc[valid, var] * df.loc[valid, "w"]).groupby(df.loc[valid, "time"]).sum()
        den = df.loc[valid, "w"].groupby(df.loc[valid, "time"]).sum()
        out[var] = num / den  # re-normalise if a city is missing for an hour
    nat = pd.DataFrame(out)
    nat.index.name = "time"
    return nat.reset_index()
