"""Command-line entry point: `uv run gridforecast --help` lists every command."""

import json
from datetime import date

import pandas as pd
import typer

from gridforecast.config import HISTORY_START, REPORT_DIR, TEST_END, TEST_START, UK

app = typer.Typer(
    help="GB Grid Forecaster: 48-hour probabilistic GB demand forecasts.", no_args_is_help=True
)
BACKTEST_DIR = REPORT_DIR / "backtests"


def _factories(params: dict):
    """Every model in the ladder, by name. Imports inside so torch stays optional."""
    from gridforecast.models.baseline import SeasonalNaive
    from gridforecast.models.ensemble import Average, Conformal
    from gridforecast.models.gbm import LGBMQuantile

    def mlp():
        from gridforecast.models.nn import MLPQuantile

        return MLPQuantile()

    return {
        "naive": SeasonalNaive,
        "lgbm": lambda: LGBMQuantile(params=params),
        "lgbm_resid": lambda: LGBMQuantile(params=params, residual=True),
        "mlp": mlp,
        "lgbm_mlp": lambda: Average([lambda: LGBMQuantile(params=params), mlp]),
        "lgbm_cqr": lambda: Conformal(lambda: LGBMQuantile(params=params)),
    }


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


@app.command()
def tune(trials: int = 30):
    """Tune LightGBM on the tuning year and save reports/best_params.json."""
    from gridforecast.data.store import load
    from gridforecast.models.tune import tune_lightgbm
    from gridforecast.pipeline.tables import ORIGINS

    tune_lightgbm(load(ORIGINS), trials)


@app.command()
def backtest(
    model: str = "lgbm",
    source: str = "historical_forecast",
    football: bool = True,
    track: bool = True,
    start: str = TEST_START,
    end: str = TEST_END,
    event: str = "",
):
    """Rolling-origin backtest of one model. Saves reports/backtests/<name>.parquet.

    --event NAME saves to reports/backtests/events/ instead, for extra windows
    (e.g. a past tournament) that shouldn't appear in the model comparison.
    """
    from gridforecast.data.store import load
    from gridforecast.evaluation import metrics
    from gridforecast.evaluation.backtest import rolling_origin_backtest
    from gridforecast.features.build import FEATURES, MATCH_FEATURES
    from gridforecast.models.tune import load_best_params
    from gridforecast.pipeline.tables import origins_path

    origins = load(origins_path(source))
    features = FEATURES if football else [f for f in FEATURES if f not in MATCH_FEATURES]
    params = load_best_params()
    name = model + ("" if football else "_nofootball")
    if source != "historical_forecast":
        name += f"_{source}"

    preds = rolling_origin_backtest(_factories(params)[model], origins, start, end, features)
    folder = BACKTEST_DIR / "events" if event else BACKTEST_DIR
    name = f"{name}_{event}" if event else name
    folder.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(folder / f"{name}.parquet", index=False)
    print(json.dumps(metrics.score(preds), indent=2))
    print(metrics.score_by(preds, "lead_day")[["mae", "mape", "coverage_80"]].round(3))
    if track:
        from gridforecast.tracking import log_backtest

        run_id = log_backtest(name, preds, params, features, source)
        print(f"MLflow run {run_id}")


@app.command()
def compare(production: str = "lgbm_cqr"):
    """Table of every saved backtest, plus NESO's forecast, -> reports/model_comparison.md."""
    from gridforecast.evaluation import metrics
    from gridforecast.pipeline.ingest import NESO

    runs = {p.stem: pd.read_parquet(p) for p in sorted(BACKTEST_DIR.glob("*.parquet"))}
    if "naive" not in runs:
        raise typer.BadParameter("Run `gridforecast backtest --model naive` first.")
    naive_mae = metrics.mae(runs["naive"]["indo_mw"], runs["naive"]["q50"])
    rows = []
    for name, p in runs.items():
        s, d1 = metrics.score(p), p[p["lead_day"] == 1]
        rows.append(
            {
                "model": name,
                "MAE (MW)": s["mae"],
                "MAPE (%)": s["mape"],
                "skill vs naive": metrics.skill(s["mae"], naive_mae),
                "pinball": s["pinball"],
                "coverage 10-90": s["coverage_80"],
                "MAE D+1": metrics.mae(d1["indo_mw"], d1["q50"]),
            }
        )
    if NESO.exists():
        neso = pd.read_parquet(NESO).set_index("start_time")["neso_mw"]
        d1 = runs["naive"][runs["naive"]["lead_day"] == 1]
        bench = neso.reindex(pd.DatetimeIndex(d1["target_time"])).to_numpy()
        rows.append(
            {
                "model": "NESO day-ahead (benchmark)",
                "MAE D+1": metrics.mae(d1["indo_mw"], bench),
                "MAPE (%)": metrics.mape(d1["indo_mw"], bench),
            }
        )
    table = pd.DataFrame(rows).set_index("model").round(3)
    print(table)
    (REPORT_DIR / "model_comparison.md").write_text(table.to_markdown(missingval=""))
    if production in runs:
        summary = metrics.score(runs[production])
        (REPORT_DIR / "backtest_summary.json").write_text(json.dumps(summary, indent=2))
