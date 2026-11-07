"""What every model shares: the quantiles it predicts and a time-based split."""

import numpy as np
import pandas as pd

from gridforecast.config import QUANTILES

QCOLS = [f"q{int(q * 100)}" for q in QUANTILES]  # ["q10", "q50", "q90"]
ROWS_PER_WEEK = 7 * 96  # one issue a day, 96 target half-hours each


def time_split(X: pd.DataFrame, y: pd.Series, weeks: int = 8):
    """Hold out the most recent `weeks` of training rows for validation.

    Rows arrive in issue-time order, so the last rows are the latest weeks.
    Never split time series randomly: neighbouring half-hours are nearly
    identical, so a random split leaks the answer into training.
    """
    n_val = min(weeks * ROWS_PER_WEEK, len(X) // 4)
    return X.iloc[:-n_val], y.iloc[:-n_val], X.iloc[-n_val:], y.iloc[-n_val:]


def sort_quantiles(preds: pd.DataFrame) -> pd.DataFrame:
    """Fix quantile crossing: force q10 <= q50 <= q90 on every row."""
    out = preds.copy()
    out[QCOLS] = np.sort(out[QCOLS].to_numpy(), axis=1)
    return out
