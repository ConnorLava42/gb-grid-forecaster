"""Stand-ins for the Elexon and Open-Meteo APIs, so tests never use the network.

Responses have the same shape as the real APIs and are generated from a
deterministic synthetic signal, so historical and "live" requests agree.
"""

import json
import re
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import responses

from gridforecast.config import CITIES, UK, WEATHER_VARS


def _noise(t: pd.DatetimeIndex, scale: float, salt: float = 0.0) -> np.ndarray:
    x = t.asi8 / 1e9 / 1800 + salt
    return scale * (np.modf(np.sin(x * 12.9898) * 43758.5453)[0])


def synthetic_temperature(t: pd.DatetimeIndex) -> np.ndarray:
    local = t.tz_convert(UK)
    hour = local.hour + local.minute / 60
    doy = local.day_of_year.to_numpy()
    return (
        10
        + 7 * np.sin(2 * np.pi * (doy - 110) / 365)
        + 3 * np.sin(2 * np.pi * (hour - 9) / 24)
        + _noise(t, 1.5, 7)
    )


def synthetic_demand(t: pd.DatetimeIndex) -> np.ndarray:
    local = t.tz_convert(UK)
    hour = local.hour + local.minute / 60
    temp = synthetic_temperature(t)
    shape = 6000 * np.sin(np.pi * np.clip(hour - 6, 0, 16) / 16)
    weekend = np.isin(local.day_of_week, [5, 6]) * -3000
    return 24000 + shape + weekend + 400 * np.clip(15.5 - temp, 0, None) + _noise(t, 400)


def _demand_rows(start: str, end: str) -> list[dict]:
    days = pd.date_range(start, end, freq="D")
    t = pd.date_range(
        days[0].tz_localize(UK),
        (days[-1] + pd.Timedelta(days=1)).tz_localize(UK),
        freq="30min",
        inclusive="left",
    ).tz_convert("UTC")
    t = t[t <= pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=30)]  # no actuals from the future
    local = t.tz_convert(UK)
    period = ((t - local.normalize().tz_convert("UTC")) / pd.Timedelta(minutes=30)).astype(int) + 1
    values = synthetic_demand(t)
    return [
        {
            "publishTime": (s + pd.Timedelta(minutes=45)).isoformat(),
            "startTime": s.isoformat(),
            "settlementDate": d.strftime("%Y-%m-%d"),
            "settlementPeriod": int(p),
            "initialDemandOutturn": int(v),
            "initialTransmissionSystemDemandOutturn": int(v + 1500),
        }
        for s, d, p, v in zip(t, local, period, values, strict=True)
    ]


def _weather_payload(times: pd.DatetimeIndex) -> list[dict]:
    temp = synthetic_temperature(times)
    hour = times.hour.to_numpy()
    out = []
    for i, (lat, lon, _) in enumerate(CITIES.values()):
        hourly = {"time": times.strftime("%Y-%m-%dT%H:%M").tolist()}
        values = {
            "temperature_2m": temp - 0.3 * i,
            "apparent_temperature": temp - 2 - 0.3 * i,
            "shortwave_radiation": np.clip(400 * np.sin(np.pi * (hour - 7) / 10), 0, None),
            "cloud_cover": 50 + 40 * np.sin(times.asi8 / 3.6e12 / 7 + i),
            "wind_speed_10m": 5 + 3 * np.cos(times.asi8 / 3.6e12 / 11 + i),
            "precipitation": np.clip(np.sin(times.asi8 / 3.6e12 / 5 + i), 0, None),
        }
        for var in WEATHER_VARS:
            hourly[var] = np.round(values[var], 2).tolist()
        out.append({"latitude": lat, "longitude": lon, "timezone": "GMT", "hourly": hourly})
    return out


def _query(request) -> dict:
    return {k: v[0] for k, v in parse_qs(urlparse(request.url).query).items()}


def demand_callback(request):
    q = _query(request)
    return (
        200,
        {},
        json.dumps(
            {
                "data": _demand_rows(q["settlementDateFrom"], q["settlementDateTo"]),
                "metadata": {"datasets": ["INDO"]},
            }
        ),
    )


def neso_callback(request):
    q = _query(request)
    t = pd.date_range(
        q["from"],
        pd.Timestamp(q["to"]) + pd.Timedelta(days=1),
        freq="30min",
        tz="UTC",
        inclusive="left",
    )
    rows = [
        {"startTime": s.isoformat(), "nationalDemand": int(v + 350)}
        for s, v in zip(t, synthetic_demand(t), strict=True)
    ]
    return 200, {}, json.dumps({"data": rows})


def weather_range_callback(request):
    q = _query(request)
    times = pd.date_range(
        q["start_date"], pd.Timestamp(q["end_date"]) + pd.Timedelta(hours=23), freq="h", tz="UTC"
    )
    return 200, {}, json.dumps(_weather_payload(times))


def weather_live_callback(request):
    q = _query(request)
    today = pd.Timestamp.now(tz="UTC").normalize()
    times = pd.date_range(
        today - pd.Timedelta(days=int(q["past_days"])),
        today + pd.Timedelta(days=int(q["forecast_days"])),
        freq="h",
        inclusive="left",
    )
    return 200, {}, json.dumps(_weather_payload(times))


def register(rsps: responses.RequestsMock) -> None:
    """Route every API the project calls to the fakes above."""
    rsps.add_callback(responses.GET, re.compile(r".*/demand/outturn.*"), demand_callback)
    rsps.add_callback(
        responses.GET, re.compile(r".*/forecast/demand/day-ahead/earliest.*"), neso_callback
    )
    rsps.add_callback(
        responses.GET,
        re.compile(r"https://(historical-forecast-api|archive-api).*"),
        weather_range_callback,
    )
    rsps.add_callback(
        responses.GET, re.compile(r"https://api\.open-meteo\.com/.*"), weather_live_callback
    )
