"""The backtest harness: monthly refits that never see their own test month."""

import pandas as pd

from gridforecast.evaluation.backtest import rolling_origin_backtest
from gridforecast.features.build import TARGET, build_origin_table


class LastWeek:
    """A dummy model: forecast = same time last week, with a +/-5% band."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        point = X["lag_7d"]
        return pd.DataFrame({"q10": point * 0.95, "q50": point, "q90": point * 1.05})


def test_backtest_never_trains_on_its_test_month(table):
    origins = build_origin_table(table)
    origins = origins[origins[TARGET].notna()]
    seen = []

    class Spy(LastWeek):
        def fit(self, X, y):
            seen.append(origins.loc[X.index, "target_time"].max())
            return self

    preds = rolling_origin_backtest(Spy, origins, "2024-03-01", "2024-04-15", verbose=False)
    assert preds["fold"].unique().tolist() == ["2024-03", "2024-04"]
    assert seen[0] < pd.Timestamp("2024-03-01", tz="UTC")
    assert seen[1] < pd.Timestamp("2024-04-01", tz="UTC")
    assert set(preds["issue_time"].dt.month) == {3, 4}
    assert preds["issue_time"].max() < pd.Timestamp("2024-04-16", tz="UTC")
