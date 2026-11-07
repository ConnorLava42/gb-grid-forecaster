"""The dashboard builds from a forecast log and live scores."""

import json

import pandas as pd

from gridforecast import report


def test_dashboard_builds(tmp_path, monkeypatch):
    issue = pd.Timestamp("2026-11-01 06:00", tz="UTC")
    targets = pd.date_range("2026-11-02", periods=96, freq="30min", tz="UTC")
    log = pd.DataFrame(
        {
            "issue_time": issue,
            "target_time": targets,
            "lead_day": 1,
            "q10": 25_000.0,
            "q50": 27_000.0,
            "q90": 29_000.0,
        }
    )
    scores = log.assign(indo_mw=27_500.0, abs_error=500.0)
    summary = {
        "month_to_date": {"mae": 500.0, "mape": 1.8},
        "last_30_days": {"coverage_80": 0.81},
        "days_live": 1,
        "updated": "2026-11-05",
    }

    paths = {
        "FORECAST_LOG": "log.parquet",
        "LIVE_SCORES": "scores.parquet",
        "LIVE_SUMMARY": "summary.json",
        "NESO": "neso.parquet",
        "BACKTEST_SUMMARY": "backtest.json",
    }
    for name, file in paths.items():
        monkeypatch.setattr(report, name, tmp_path / file)
    monkeypatch.setattr(report, "DOCS_DIR", tmp_path / "docs")
    log.to_parquet(report.FORECAST_LOG)
    scores.to_parquet(report.LIVE_SCORES)
    report.LIVE_SUMMARY.write_text(json.dumps(summary))

    html = report.build_dashboard()
    assert (tmp_path / "docs" / "index.html").exists()
    assert "Next 48 hours" in html and "500 MW" in html and "81%" in html
