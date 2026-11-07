"""Point and probabilistic forecast metrics.

Every function takes the actuals `y` and predictions as arrays or Series and
ignores rows where the actual is missing.
"""

import numpy as np
import pandas as pd


def _clean(y, *preds):
    y = np.asarray(y, dtype=float)
    preds = [np.asarray(p, dtype=float) for p in preds]
    keep = ~np.isnan(y)
    for p in preds:
        keep &= ~np.isnan(p)
    return (y[keep], *[p[keep] for p in preds])


def mae(y, pred) -> float:
    """Mean absolute error, in MW."""
    y, pred = _clean(y, pred)
    return float(np.mean(np.abs(y - pred)))


def mape(y, pred) -> float:
    """Mean absolute percentage error, in %."""
    y, pred = _clean(y, pred)
    return float(100 * np.mean(np.abs(y - pred) / np.abs(y)))


def rmse(y, pred) -> float:
    y, pred = _clean(y, pred)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def pinball(y, pred, q: float) -> float:
    """Pinball (quantile) loss for quantile q: lower is better."""
    y, pred = _clean(y, pred)
    err = y - pred
    return float(np.mean(np.maximum(q * err, (q - 1) * err)))


def mean_pinball(y, q10, q50, q90) -> float:
    """Average pinball loss over the three quantiles we forecast."""
    return float(np.mean([pinball(y, q10, 0.1), pinball(y, q50, 0.5), pinball(y, q90, 0.9)]))


def coverage(y, lower, upper) -> float:
    """Share of actuals inside [lower, upper]. Our 10-90 band should give 0.8."""
    y, lower, upper = _clean(y, lower, upper)
    return float(np.mean((y >= lower) & (y <= upper)))


def width(lower, upper) -> float:
    """Average width of the prediction band, in MW: narrower is better, if calibrated."""
    return float(np.nanmean(np.asarray(upper) - np.asarray(lower)))


def score(df: pd.DataFrame, y="indo_mw") -> dict:
    """Every metric for one set of predictions with columns q10, q50, q90."""
    df = df[df[y].notna()]
    return {
        "mae": mae(df[y], df["q50"]),
        "mape": mape(df[y], df["q50"]),
        "rmse": rmse(df[y], df["q50"]),
        "pinball": mean_pinball(df[y], df["q10"], df["q50"], df["q90"]),
        "coverage_80": coverage(df[y], df["q10"], df["q90"]),
        "below_q10": float(np.mean(df[y] < df["q10"])),
        "above_q90": float(np.mean(df[y] > df["q90"])),
        "width_80": width(df["q10"], df["q90"]),
        "n": len(df),
    }


def score_by(df: pd.DataFrame, by: str | list[str] | pd.Series) -> pd.DataFrame:
    """score() for each group, e.g. by lead day or by month."""
    return pd.DataFrame({k: score(g) for k, g in df.groupby(by)}).T


def peak_mae(df: pd.DataFrame, y="indo_mw") -> float:
    """MAE at each day's peak half-hour (UK local day)."""
    df = df[df[y].notna()]
    local_day = df["target_time"].dt.tz_convert("Europe/London").dt.date
    peaks = df.loc[df.groupby([df["issue_time"], local_day])[y].idxmax()]
    return mae(peaks[y], peaks["q50"])


def skill(model_mae: float, baseline_mae: float) -> float:
    """Improvement over the baseline: 0.2 means 20% lower MAE."""
    return 1 - model_mae / baseline_mae
