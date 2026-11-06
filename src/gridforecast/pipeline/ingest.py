"""Download everything the project needs into data/raw/."""

from datetime import date, timedelta

import pandas as pd

from gridforecast.config import RAW_DIR, UK
from gridforecast.data import elexon, weather
from gridforecast.data.store import upsert

DEMAND = RAW_DIR / "demand.parquet"
NESO = RAW_DIR / "neso_forecast.parquet"
LIVE_WEATHER = RAW_DIR / "weather_live.parquet"


def weather_path(source: str):
    """data/raw/weather_<source>.parquet, e.g. weather_historical_forecast.parquet."""
    return RAW_DIR / f"weather_{source}.parquet"


def ingest_history(start: date, end: date, sources=("historical_forecast",)) -> None:
    """Backfill demand, NESO's forecast and weather between two dates."""
    print(f"Demand {start} to {end}...")
    upsert(elexon.fetch_demand(start, end), DEMAND, "start_time")
    print("NESO day-ahead forecast...")
    try:
        upsert(elexon.fetch_neso_forecast(start, end), NESO, "start_time")
    except Exception as exc:  # the benchmark is optional; never block the main pipeline
        print(f"  Skipped NESO benchmark: {exc}")
    for source in sources:
        print(f"Weather ({source})...")
        upsert(weather.fetch_weather(start, end, source), weather_path(source), ["time", "city"])


def ingest_recent(days: int = 16) -> None:
    """What the daily job needs: recent demand and the latest weather forecast."""
    today = pd.Timestamp.now(tz=UK).date()
    start = today - timedelta(days=days)
    upsert(elexon.fetch_demand(start, today), DEMAND, "start_time")
    upsert(
        weather.fetch_live_weather(past_days=days, forecast_days=4), LIVE_WEATHER, ["time", "city"]
    )
    try:
        upsert(elexon.fetch_neso_forecast(start, today + timedelta(days=2)), NESO, "start_time")
    except Exception as exc:
        print(f"Skipped NESO benchmark: {exc}")
