"""Log every backtest to MLflow so runs are comparable and reproducible."""

import hashlib
import subprocess
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from gridforecast.config import ROOT
from gridforecast.evaluation import metrics

TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
EXPERIMENT = "gb-demand-backtest"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def log_backtest(
    name: str,
    preds: pd.DataFrame,
    params: dict,
    features: list[str],
    weather_source: str,
    stage: str = "final",
    figures: list[Path] = (),
) -> str:
    """Log one backtest: settings, overall and monthly metrics, and files. Returns the run ID."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"stage": stage, "model": name, "git_sha": _git_sha()})
        mlflow.log_params(
            params
            | {
                "weather_source": weather_source,
                "n_features": len(features),
                "feature_hash": hashlib.md5(",".join(features).encode()).hexdigest()[:8],
            }
        )
        overall = metrics.score(preds)
        mlflow.log_metrics({k: v for k, v in overall.items() if k != "n"})
        mlflow.log_metric("peak_mae", metrics.peak_mae(preds))
        for lead, s in metrics.score_by(preds, "lead_day").iterrows():
            mlflow.log_metric(f"mae_d{lead}", s["mae"])
        for step, (_, s) in enumerate(metrics.score_by(preds, "fold").iterrows()):
            mlflow.log_metric("monthly_mae", s["mae"], step=step)
            mlflow.log_metric("monthly_coverage", s["coverage_80"], step=step)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{name}_predictions.parquet"
            preds.to_parquet(path, index=False)
            mlflow.log_artifact(str(path))
        for fig in figures:
            mlflow.log_artifact(str(fig))
        return run.info.run_id
