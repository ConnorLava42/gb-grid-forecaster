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


@app.command()
def explain():
    """SHAP plots for LightGBM and the TV pickup event study."""
    from gridforecast.data.fixtures import load_matches
    from gridforecast.data.store import load
    from gridforecast.explain import plot_event_study, shap_plots, tv_pickup_event_study
    from gridforecast.features.build import FEATURES, TARGET
    from gridforecast.models.gbm import LGBMQuantile
    from gridforecast.models.tune import load_best_params
    from gridforecast.pipeline.tables import ORIGINS

    origins = load(ORIGINS)
    train = origins[
        (origins["target_time"] < pd.Timestamp(TEST_END, tz="UTC")) & origins[TARGET].notna()
    ]
    model = LGBMQuantile(params=load_best_params()).fit(train[FEATURES], train[TARGET])
    for p in shap_plots(model, train[FEATURES]):
        print(f"Saved {p}")

    # TV pickup: every no-football backtest we have, test year plus any tournament windows
    files = [BACKTEST_DIR / "lgbm_nofootball.parquet"]
    files += sorted((BACKTEST_DIR / "events").glob("lgbm_nofootball_*.parquet"))
    files = [f for f in files if f.exists()]
    if not files:
        print(
            "For the TV pickup study, first run: gridforecast backtest --model lgbm --no-football"
        )
        return
    preds = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    preds = preds.drop_duplicates(["issue_time", "target_time"])
    study = tv_pickup_event_study(preds, load_matches())
    if study.empty:
        print("No England matches fall inside these backtests; add a tournament window (Phase 7).")
        return
    study.round(1).to_csv(REPORT_DIR / "tv_pickup.csv")
    print(study.round(0))
    print(f"Saved {plot_event_study(study)}")


@app.command()
def train():
    """Train the production model on all history and save it to models/production/."""
    from gridforecast.pipeline.live import train_production

    train_production()


@app.command()
def forecast(day: str = "", force: bool = False):
    """Issue the 06:00 forecast for today (or --day YYYY-MM-DD)."""
    from gridforecast.pipeline.live import forecast_today

    forecast_today(date.fromisoformat(day) if day else None, force)


@app.command()
def score():
    """Score every logged forecast whose target has passed."""
    from gridforecast.pipeline.live import score_live

    print(json.dumps(score_live(), indent=2))


@app.command("retrain-monthly")
def retrain_monthly():
    """Retrain the production model if it doesn't get worse."""
    from gridforecast.pipeline.live import retrain_monthly as run

    run()


@app.command()
def report():
    """Regenerate docs/index.html."""
    from gridforecast.report import build_dashboard

    build_dashboard()
    print("Wrote docs/index.html")
