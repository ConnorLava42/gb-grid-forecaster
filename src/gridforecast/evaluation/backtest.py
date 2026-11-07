"""Rolling-origin backtest: refit monthly, forecast every day of the next month."""

from collections.abc import Callable

import pandas as pd

from gridforecast.features.build import FEATURES, TARGET

KEEP = ["issue_time", "target_time", "lead_day", "hh", TARGET]


def rolling_origin_backtest(
    model_factory: Callable,
    origins: pd.DataFrame,
    test_start: str,
    test_end: str,
    features: list[str] = FEATURES,
    verbose: bool = True,
) -> pd.DataFrame:
    """Predictions for every forecast issued between test_start and test_end.

    model_factory() must return a fresh object with fit(X, y) and predict(X),
    where predict returns a DataFrame with columns q10, q50 and q90.

    For each month, the model is trained only on rows whose *target* is
    before the month starts, so no actual from the test month can leak in.
    """
    months = pd.date_range(test_start, test_end, freq="MS", tz="UTC")
    end = pd.Timestamp(test_end, tz="UTC") + pd.Timedelta(days=1)
    results = []
    for month_start in months:
        month_end = min(month_start + pd.offsets.MonthBegin(1), end)
        train = origins[(origins["target_time"] < month_start) & origins[TARGET].notna()]
        test = origins[(origins["issue_time"] >= month_start) & (origins["issue_time"] < month_end)]
        if test.empty:
            continue
        model = model_factory()
        model.fit(train[features], train[TARGET])
        preds = model.predict(test[features])
        out = test[KEEP].reset_index(drop=True).join(preds.reset_index(drop=True))
        out["fold"] = month_start.strftime("%Y-%m")
        results.append(out)
        if verbose:
            print(f"{month_start:%Y-%m}: trained on {len(train):,} rows, forecast {len(test):,}")
    return pd.concat(results, ignore_index=True)
