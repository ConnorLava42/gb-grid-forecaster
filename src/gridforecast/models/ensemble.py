"""Combining models: a simple average, and a conformal correction for the band."""

import numpy as np
import pandas as pd

from gridforecast.models.base import QCOLS, sort_quantiles, time_split


class Average:
    """Average the quantiles of several models (e.g. LightGBM and the MLP)."""

    def __init__(self, factories):
        self.factories = factories
        self.name = "+".join(f().name for f in factories)

    def fit(self, X, y):
        self.models_ = [f().fit(X, y) for f in self.factories]
        return self

    def predict(self, X):
        preds = [m.predict(X)[QCOLS] for m in self.models_]
        return sort_quantiles(sum(preds) / len(preds))


class Conformal:
    """Conformalised quantile regression (CQR) around any quantile model.

    1. Fit the model on all but the latest 8 weeks.
    2. On those 8 weeks, measure how far each actual falls outside the band:
       E = max(q10 - y, y - q90). Negative E means it fell inside.
    3. Q = the (80%)-quantile of E. Moving both edges out by Q would have
       covered 80% of the calibration weeks.
    4. Refit on everything and widen every future band by Q.

    Q can be negative, which narrows a band that was too cautious.
    """

    def __init__(self, factory, coverage: float = 0.8, weeks: int = 8):
        self.factory, self.coverage, self.weeks = factory, coverage, weeks
        self.name = f"{factory().name}_cqr"

    def fit(self, X, y):
        keep = y.notna()
        X, y = X[keep], y[keep]
        X_fit, y_fit, X_cal, y_cal = time_split(X, y, self.weeks)
        cal = self.factory().fit(X_fit, y_fit).predict(X_cal)
        scores = np.maximum(cal[QCOLS[0]] - y_cal, y_cal - cal[QCOLS[-1]]).to_numpy()
        n = len(scores)
        level = min(np.ceil((n + 1) * self.coverage) / n, 1.0)  # finite-sample correction
        self.q_ = float(np.quantile(scores, level))
        self.model_ = self.factory().fit(X, y)
        return self

    def predict(self, X) -> pd.DataFrame:
        preds = self.model_.predict(X).copy()
        preds[QCOLS[0]] -= self.q_
        preds[QCOLS[-1]] += self.q_
        return sort_quantiles(preds)
