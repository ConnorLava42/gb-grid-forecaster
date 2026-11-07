"""Build the static dashboard at docs/index.html for GitHub Pages."""

import json
from html import escape

import pandas as pd
import plotly.graph_objects as go

from gridforecast.config import DOCS_DIR, REPORT_DIR, UK
from gridforecast.pipeline.ingest import NESO
from gridforecast.pipeline.live import FORECAST_LOG, LIVE_SCORES, LIVE_SUMMARY

BACKTEST_SUMMARY = REPORT_DIR / "backtest_summary.json"
REPO_URL = "https://github.com/ConnorLava42/gb-grid-forecaster"
BLUE, GREY, INK = "#2563eb", "#6b7280", "#111827"


def _layout(fig: go.Figure, title: str, ytitle: str = "Demand (MW)", first: bool = False) -> str:
    """Shared styling. The first chart on the page also loads plotly.js from its CDN."""
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=380,
        margin=dict(l=50, r=20, t=50, b=40),
        yaxis_title=ytitle,
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
    )
    return fig.to_html(include_plotlyjs="cdn" if first else False, full_html=False)


def _local(s: pd.Series) -> pd.Series:
    return s.dt.tz_convert(UK).dt.tz_localize(None)


def next_48h(log: pd.DataFrame) -> str:
    latest = log[log["issue_time"] == log["issue_time"].max()]
    x = _local(latest["target_time"])
    fig = go.Figure(
        [
            go.Scatter(
                x=x, y=latest["q90"], line=dict(width=0), showlegend=False, hoverinfo="skip"
            ),
            go.Scatter(
                x=x,
                y=latest["q10"],
                fill="tonexty",
                fillcolor="rgba(37,99,235,0.18)",
                line=dict(width=0),
                name="10-90% range",
            ),
            go.Scatter(
                x=x, y=latest["q50"], line=dict(color=BLUE, width=2.5), name="Forecast (median)"
            ),
        ]
    )
    if NESO.exists():
        neso = pd.read_parquet(NESO)
        neso = neso[neso["start_time"].isin(latest["target_time"])]
        if len(neso):
            fig.add_scatter(
                x=_local(neso["start_time"]),
                y=neso["neso_mw"],
                name="NESO forecast",
                line=dict(color=GREY, dash="dash"),
            )
    issued = latest["issue_time"].iloc[0].tz_convert(UK)
    return _layout(fig, f"Next 48 hours (issued {issued:%a %d %b %H:%M} UK time)", first=True)


def last_14_days(scores: pd.DataFrame) -> str:
    recent = scores[
        (scores["lead_day"] == 1)
        & (scores["target_time"] >= scores["target_time"].max() - pd.Timedelta(days=14))
    ]
    x = _local(recent["target_time"])
    fig = go.Figure(
        [
            go.Scatter(x=x, y=recent["q50"], name="Day-ahead forecast", line=dict(color=BLUE)),
            go.Scatter(
                x=x, y=recent["indo_mw"], name="Actual", line=dict(color=INK, width=1, dash="dot")
            ),
        ]
    )
    return _layout(fig, "Last 14 days: day-ahead forecast vs actual")


def daily_mae(scores: pd.DataFrame) -> str:
    d1 = scores[scores["lead_day"] == 1]
    by_day = d1.groupby(_local(d1["target_time"]).dt.date)["abs_error"].mean()
    fig = go.Figure(
        go.Bar(x=list(by_day.index), y=by_day.values, marker_color=BLUE, name="Daily MAE")
    )
    if BACKTEST_SUMMARY.exists():
        ref = json.loads(BACKTEST_SUMMARY.read_text())["mae"]
        fig.add_hline(
            y=ref, line_dash="dash", line_color=GREY, annotation_text=f"Backtest MAE {ref:,.0f} MW"
        )
    return _layout(fig, "Day-ahead error per day since go-live", "MAE (MW)")


def _tile(label: str, value: str) -> str:
    return (
        f'<div class="tile"><div class="v">{escape(value)}</div>'
        f'<div class="l">{escape(label)}</div></div>'
    )


def build_dashboard() -> str:
    log = pd.read_parquet(FORECAST_LOG)
    scores = pd.read_parquet(LIVE_SCORES) if LIVE_SCORES.exists() else pd.DataFrame()
    summary = json.loads(LIVE_SUMMARY.read_text()) if LIVE_SUMMARY.exists() else {}
    mtd = summary.get("month_to_date") or {}
    last30 = summary.get("last_30_days") or {}

    tiles = "".join(
        [
            _tile("MAE, month to date", f"{mtd['mae']:,.0f} MW" if mtd else "-"),
            _tile("MAPE, month to date", f"{mtd['mape']:.1f}%" if mtd else "-"),
            _tile(
                "10-90% coverage, last 30 days", f"{last30['coverage_80']:.0%}" if last30 else "-"
            ),
            _tile("Days forecasting", str(summary.get("days_live", log["issue_time"].nunique()))),
        ]
    )
    charts = [next_48h(log)]
    if len(scores):
        charts += [last_14_days(scores), daily_mae(scores)]

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GB Grid Forecaster</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 0 auto; max-width: 1000px;
        padding: 16px; color: {INK}; }}
 .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }}
 .tile {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; }}
 .v {{ font-size: 1.6rem; font-weight: 600; }} .l {{ color: {GREY}; font-size: 0.9rem; }}
 footer {{ color: {GREY}; font-size: 0.85rem; margin-top: 24px; }}
</style></head><body>
<h1>GB Grid Forecaster</h1>
<p>A machine-learning model that forecasts Great Britain's electricity demand for the next
48 hours every morning at 06:00, with an 80% uncertainty range, and scores itself against
what actually happened.</p>
<div class="tiles">{tiles}</div>
{"".join(charts)}
<footer>Updated {escape(summary.get("updated", "")[:16])} UTC.
Demand data: Elexon BMRS (Insights Solution). Weather: Open-Meteo.com (CC BY 4.0).
Code: <a href="{REPO_URL}">GitHub</a>.</footer>
</body></html>"""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
    return html
