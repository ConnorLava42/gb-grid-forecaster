"""Project settings: paths, dates, locations and modelling choices.

Every value you might want to change lives here, so you never hunt
through the code for a hard-coded date or file name.
"""

from pathlib import Path
from zoneinfo import ZoneInfo

# Folders, built relative to this file so they work wherever you run from
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # API downloads (git-ignored)
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned tables (git-ignored)
REFERENCE_DIR = DATA_DIR / "reference"  # small hand-made files (committed)
FORECAST_DIR = ROOT / "forecasts"  # live forecast log (committed)
REPORT_DIR = ROOT / "reports"  # backtests, metrics, live scores
FIG_DIR = REPORT_DIR / "figures"
MODEL_DIR = ROOT / "models" / "production"  # model used by the daily job
DOCS_DIR = ROOT / "docs"  # GitHub Pages dashboard

UK = ZoneInfo("Europe/London")

# History starts when Open-Meteo's archive of past forecasts begins
HISTORY_START = "2021-01-01"

# Forecast design: issued at 06:00 UK time on day D, covering D+1 and D+2
ISSUE_HOUR = 6
LEAD_DAYS = (1, 2)
QUANTILES = (0.1, 0.5, 0.9)

# Backtest design: tune on one year, test on the next, never tune on the test year
TUNING_START, TUNING_END = "2024-11-01", "2025-10-31"
TEST_START, TEST_END = "2025-11-01", "2026-10-31"

# Weather locations: (latitude, longitude, weight). Each city stands in for a
# region and is weighted by that region's population in millions. These are
# approximate ONS mid-year estimates; check and update them from ons.gov.uk.
CITIES = {
    "London": (51.51, -0.13, 24.7),  # London + South East + East of England
    "Bristol": (51.45, -2.59, 5.8),  # South West
    "Birmingham": (52.48, -1.90, 10.9),  # West + East Midlands
    "Manchester": (53.48, -2.24, 7.5),  # North West
    "Leeds": (53.80, -1.55, 5.5),  # Yorkshire and the Humber
    "Newcastle": (54.98, -1.61, 2.7),  # North East
    "Cardiff": (51.48, -3.18, 3.1),  # Wales
    "Glasgow": (55.86, -4.25, 5.4),  # Scotland
}

# Hourly weather variables requested from Open-Meteo
WEATHER_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "shortwave_radiation",
    "cloud_cover",
    "wind_speed_10m",
    "precipitation",
]

# Temperature below which homes and offices start heating (degrees C)
HEATING_BASE_C = 15.5
