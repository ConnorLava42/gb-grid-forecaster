"""Raw downloads -> the processed tables every model trains on."""

import pandas as pd

from gridforecast.config import PROCESSED_DIR
from gridforecast.data import weather
from gridforecast.data.clean import build_model_table
from gridforecast.data.fixtures import load_matches
from gridforecast.data.store import load
from gridforecast.features.build import build_origin_table
from gridforecast.pipeline.ingest import DEMAND, weather_path

MODEL_TABLE = PROCESSED_DIR / "model_table.parquet"
ORIGINS = PROCESSED_DIR / "origins.parquet"


def origins_path(source: str = "historical_forecast"):
    """The training table. ERA5 ("archive") weather gets its own file for the oracle test."""
    if source == "historical_forecast":
        return ORIGINS
    return PROCESSED_DIR / f"origins_{source}.parquet"


def build_tables(source: str = "historical_forecast") -> pd.DataFrame:
    """Raw files -> model_table.parquet -> origins.parquet (the training table)."""
    nat = weather.national_weather(load(weather_path(source)))
    table = build_model_table(load(DEMAND), nat)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(MODEL_TABLE)
    origins = build_origin_table(table, matches=load_matches())
    out = origins_path(source)
    origins.to_parquet(out, index=False)
    print(
        f"{len(table):,} half-hours, {origins['issue_time'].nunique():,} forecast issues -> {out}"
    )
    return origins
