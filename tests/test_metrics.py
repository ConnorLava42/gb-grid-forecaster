"""Metrics, checked against numbers worked out by hand."""

import numpy as np
import pandas as pd

from gridforecast.evaluation import metrics


def test_pinball_loss_hand_example():
    # y=10, q=8, tau=0.9: under-forecast by 2 costs 0.9 * 2 = 1.8
    # y=10, q=13, tau=0.9: over-forecast by 3 costs 0.1 * 3 = 0.3
    assert np.isclose(metrics.pinball([10, 10], [8, 13], 0.9), (1.8 + 0.3) / 2)


def test_coverage_is_eighty_percent_on_a_built_case():
    y = np.arange(10)
    lower, upper = np.full(10, 1), np.full(10, 8)  # 1..8 inside, 0 and 9 outside
    assert metrics.coverage(y, lower, upper) == 0.8


def test_mae_and_mape_ignore_missing_actuals():
    assert metrics.mae([100, np.nan, 300], [110, 5, 270]) == 20
    assert np.isclose(metrics.mape([100, 200], [110, 180]), 10.0)


def test_skill():
    assert np.isclose(metrics.skill(80, 100), 0.2)


def test_score_by_groups():
    df = pd.DataFrame(
        {
            "indo_mw": [1.0, 2, 3, 4],
            "q10": 0.0,
            "q50": [1.0, 2, 3, 5],
            "q90": 10.0,
            "lead_day": [1, 1, 2, 2],
        }
    )
    by = metrics.score_by(df, "lead_day")
    assert by.loc[1, "mae"] == 0 and by.loc[2, "mae"] == 0.5


def test_score_ignores_rows_without_actuals():
    df = pd.DataFrame(
        {"indo_mw": [10.0, np.nan], "q10": [5.0, 0], "q50": [10.0, 0], "q90": [20.0, 0]}
    )
    s = metrics.score(df)
    assert s["n"] == 1 and s["mae"] == 0 and s["below_q10"] == 0 and s["coverage_80"] == 1
