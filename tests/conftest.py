"""Shared test fixtures: a small, realistic-looking synthetic dataset.

Tests never touch the network. Synthetic data with a daily shape, a weekly
cycle and a temperature effect is enough to check that the code is correct.
"""

import numpy as np
import pandas as pd
import pytest

from gridforecast.config import UK, WEATHER_VARS


def make_table(start="2024-01-01", end="2024-04-30", seed=0) -> pd.DataFrame:
    """A model table like data.clean.build_model_table returns."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="30min", tz="UTC", name="start_time")
    local = idx.tz_convert(UK)
    hour = local.hour + local.minute / 60
    doy = local.day_of_year.to_numpy()

    temp = 8 + 6 * np.sin(2 * np.pi * (doy - 100) / 365) + 3 * np.sin(2 * np.pi * (hour - 9) / 24)
    temp = temp + rng.normal(0, 1, len(idx))
    daily_shape = 6000 * np.sin(np.pi * np.clip(hour - 6, 0, 16) / 16)
    weekend = np.isin(local.day_of_week, [5, 6]) * -3000
    demand = 24000 + daily_shape + weekend + 400 * np.clip(15.5 - temp, 0, None)
    demand = demand + rng.normal(0, 300, len(idx))

    table = pd.DataFrame(index=idx)
    table["indo_mw"] = demand
    table["valid"] = True
    table["temperature_2m"] = temp
    table["apparent_temperature"] = temp - 2
    table["shortwave_radiation"] = np.clip(400 * np.sin(np.pi * (hour - 7) / 10), 0, None)
    table["cloud_cover"] = rng.uniform(0, 100, len(idx))
    table["wind_speed_10m"] = rng.uniform(0, 12, len(idx))
    table["precipitation"] = rng.exponential(0.1, len(idx))
    assert set(WEATHER_VARS) <= set(table.columns)
    return table


@pytest.fixture
def table() -> pd.DataFrame:
    return make_table()
