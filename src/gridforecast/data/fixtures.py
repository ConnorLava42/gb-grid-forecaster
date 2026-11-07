"""Big football matches: when they happen and which half-hours they cover."""

import numpy as np
import pandas as pd

from gridforecast.config import REFERENCE_DIR

MATCHES_CSV = REFERENCE_DIR / "england_matches.csv"

PHASES = {0: "none", 1: "first_half", 2: "half_time", 3: "second_half", 4: "full_time"}
TIERS = {
    "group": 1,
    "round_of_32": 2,
    "round_of_16": 2,
    "quarter_final": 2,
    "third_place": 2,
    "semi_final": 3,
    "final": 3,
}


def load_matches(path=MATCHES_CSV) -> pd.DataFrame:
    """Read the hand-made match list and work out when each match ended."""
    m = pd.read_csv(path)
    m["kickoff_utc"] = pd.to_datetime(m["kickoff_utc"], utc=True)
    m["tier"] = m["stage"].map(TIERS).fillna(1).astype(int)
    # 45 + 15 (half-time) + 45 + ~7 stoppage = 112 minutes to the final whistle
    minutes = 112 + 35 * m["extra_time"].astype(int) + 15 * m["penalties"].astype(int)
    m["end_utc"] = m["kickoff_utc"] + pd.to_timedelta(minutes, unit="min")
    return m


def match_features(times: pd.DatetimeIndex, matches: pd.DataFrame) -> pd.DataFrame:
    """For each half-hour starting at `times`, the match phase and tier.

    A half-hour is labelled, in priority order:
      half_time   if it contains the middle of the half-time break
      full_time   if it contains the final whistle
      first_half  if it overlaps the first 47 minutes
      second_half if it overlaps the second half (and any extra time)
    """
    phase = np.zeros(len(times), dtype=int)
    tier = np.zeros(len(times), dtype=int)
    starts = times.tz_convert(None).to_numpy()
    ends = (times + pd.Timedelta(minutes=30)).tz_convert(None).to_numpy()

    def overlaps(a, b):
        return (starts < np.datetime64(b)) & (ends > np.datetime64(a))

    def contains(t):
        return (starts <= np.datetime64(t)) & (ends > np.datetime64(t))

    for m in matches.itertuples():
        k = m.kickoff_utc.tz_convert("UTC").tz_localize(None)
        end = m.end_utc.tz_convert("UTC").tz_localize(None)
        labels = [
            (1, overlaps(k, k + pd.Timedelta(minutes=47))),
            (3, overlaps(k + pd.Timedelta(minutes=62), end)),
            (4, contains(end)),
            (2, contains(k + pd.Timedelta(minutes=56))),
        ]
        for code, mask in labels:  # later entries overwrite earlier ones = higher priority
            phase[mask] = code
            tier[mask] = m.tier
    return pd.DataFrame({"match_phase": phase, "match_tier": tier}, index=times)


def premier_league_kickoffs(season: str) -> pd.DataFrame:
    """Premier League kick-offs from football-data.co.uk, e.g. season="2425".

    The Date column is UK local, dd/mm/yyyy, and Time is the local kick-off.
    """
    url = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
    df = pd.read_csv(url, usecols=["Date", "Time", "HomeTeam", "AwayTeam"])
    local = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
    df["kickoff_utc"] = local.dt.tz_localize("Europe/London").dt.tz_convert("UTC")
    return df[["kickoff_utc", "HomeTeam", "AwayTeam"]]
