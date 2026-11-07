"""Command-line entry point: `uv run gridforecast --help` lists every command."""

from datetime import date

import pandas as pd
import typer

from gridforecast.config import HISTORY_START, UK

app = typer.Typer(
    help="GB Grid Forecaster: 48-hour probabilistic GB demand forecasts.", no_args_is_help=True
)


@app.command()
def ingest(start: str = HISTORY_START, end: str = "", oracle: bool = False):
    """Backfill raw data. --oracle also downloads ERA5 actual weather."""
    from gridforecast.pipeline.ingest import ingest_history

    end_day = date.fromisoformat(end) if end else pd.Timestamp.now(tz=UK).date()
    sources = ("historical_forecast", "archive") if oracle else ("historical_forecast",)
    ingest_history(date.fromisoformat(start), end_day, sources)


@app.command("ingest-recent")
def ingest_recent(days: int = 16):
    """Recent demand and the latest weather forecast, for the daily job."""
    from gridforecast.pipeline.ingest import ingest_recent as run

    run(days)


@app.command()
def build(source: str = "historical_forecast"):
    """Build model_table.parquet and the forecast-origin table from raw files."""
    from gridforecast.pipeline.tables import build_tables

    build_tables(source)
