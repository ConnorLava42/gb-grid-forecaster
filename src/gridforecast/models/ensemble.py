"""Combining models: a simple average (and, in Phase 6, a conformal correction)."""

from gridforecast.models.base import QCOLS, sort_quantiles


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
