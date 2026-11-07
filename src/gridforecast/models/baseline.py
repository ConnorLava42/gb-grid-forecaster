"""Rung 1: seasonal naive. Tomorrow looks like the same time last week."""

import pandas as pd

from gridforecast.models.base import QCOLS


class SeasonalNaive:
    """Forecast = demand at the same time 7 days before the target.

    The 10-90 band comes from last year's naive errors at each half-hour of
    the day, so the band is wider at the evening peak than overnight.
    """

    name = "seasonal_naive"

    def __init__(self, history_days: int = 365):
        self.history_days = history_days

    @staticmethod
    def _point(X: pd.DataFrame) -> pd.Series:
        # Fall back to two weeks ago if last week's value is missing
        return X["lag_7d"].fillna(X["lag_14d"])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        recent = slice(-self.history_days * 96, None)
        resid = (y - self._point(X)).iloc[recent]
        hh = X["hh"].iloc[recent]
        self.offsets_ = resid.groupby(hh).quantile([0.1, 0.9]).unstack()
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        point = self._point(X).to_numpy()
        off = self.offsets_.reindex(X["hh"].to_numpy()).to_numpy()
        return pd.DataFrame(
            {QCOLS[0]: point + off[:, 0], QCOLS[1]: point, QCOLS[2]: point + off[:, 1]},
            index=X.index,
        )
