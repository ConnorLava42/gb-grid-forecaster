"""The most important tests in the repo: the features never see the future."""

import numpy as np
import pandas as pd

from gridforecast.data.fixtures import load_matches, match_features
from gridforecast.features.build import FEATURES, build_origin_table, issue_time_for


def test_no_leakage(table):
    """Scrambling every demand value at or after the issue time must not change any feature."""
    issue_day = pd.Timestamp("2024-03-10").date()
    issue = issue_time_for(issue_day)
    before = build_origin_table(table, [issue_day])

    scrambled = table.copy()
    future = scrambled.index >= issue
    scrambled.loc[future, "indo_mw"] = np.random.default_rng(1).uniform(0, 1e6, future.sum())
    after = build_origin_table(scrambled, [issue_day])

    pd.testing.assert_frame_equal(before[FEATURES], after[FEATURES])
    assert (before["target_time"] > before["issue_time"]).all()


def test_clock_change_days(table):
    """Spring 2024 clock change (31 March) has 46 local half-hours.

    So forecasts issued on 29 and 30 March have 94 rows instead of 96.
    """
    origins = build_origin_table(
        table,
        [
            pd.Timestamp("2024-03-29").date(),
            pd.Timestamp("2024-03-30").date(),
            pd.Timestamp("2024-03-20").date(),
        ],
    )
    counts = origins.groupby(origins["issue_time"].dt.date).size()
    assert counts.loc[pd.Timestamp("2024-03-20").date()] == 96
    assert counts.loc[pd.Timestamp("2024-03-29").date()] == 94
    assert counts.loc[pd.Timestamp("2024-03-30").date()] == 94


def test_issue_time_is_six_am_uk():
    assert issue_time_for(pd.Timestamp("2024-01-15").date()) == pd.Timestamp("2024-01-15 06:00Z")
    assert issue_time_for(pd.Timestamp("2024-07-15").date()) == pd.Timestamp("2024-07-15 05:00Z")


def test_bank_holiday_flag(table):
    origins = build_origin_table(table, [pd.Timestamp("2024-03-28").date()])  # Good Friday 29 Mar
    good_friday = origins[origins["target_time"].dt.tz_convert("Europe/London").dt.day == 29]
    assert good_friday["is_holiday"].eq(1).all()


def test_match_phases_for_a_20_00_kickoff():
    """Euro 2024 final, 20:00 BST = 19:00 UTC."""
    matches = load_matches()
    times = pd.date_range("2024-07-14 18:30", "2024-07-14 21:30", freq="30min", tz="UTC")
    phases = match_features(times, matches)["match_phase"].tolist()
    #           18:30 19:00 19:30 20:00 20:30 21:00 21:30
    assert phases == [0, 1, 2, 3, 4, 0, 0]


def test_bst_flag(table):
    origins = build_origin_table(table, [pd.Timestamp("2024-03-29").date()])
    local_day = origins["target_time"].dt.tz_convert("Europe/London").dt.day
    assert origins.loc[local_day == 30, "is_bst"].eq(0).all()  # still GMT
    assert origins.loc[local_day == 31, "is_bst"].iloc[-1] == 1  # BST from 01:00 on 31 March
