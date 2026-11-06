"""The package imports and its settings make sense."""

from gridforecast import config


def test_settings_are_consistent():
    assert config.ROOT.joinpath("pyproject.toml").exists()
    assert config.QUANTILES == (0.1, 0.5, 0.9)
    assert all(weight > 0 for _, _, weight in config.CITIES.values())
    assert config.TUNING_END < config.TEST_START  # never tune on the test year
