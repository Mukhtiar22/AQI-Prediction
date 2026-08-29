"""
utils/data_source.py
---------------------
Fetches raw weather + pollutant data for a city.

Primary source: Open-Meteo (https://open-meteo.com)
    - Free, no API key required, generous rate limits — good fit for a
      GitHub-Actions-driven, "100% serverless" pipeline.
    - Air Quality API   -> pm2_5, pm10, ozone, no2, so2, co, us_aqi
    - Weather API       -> temperature, humidity, wind, pressure, precip

If the network is unavailable (e.g. running in a sandboxed / offline
environment, or Open-Meteo is down), `fetch_raw_data` transparently falls
back to a physically-plausible synthetic generator so the rest of the
pipeline (feature engineering, training, dashboard) can still be built,
tested and demoed end to end. Swap `USE_SYNTHETIC_FALLBACK = False` to
force a hard failure instead once you've verified real network access.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

USE_SYNTHETIC_FALLBACK = True

CITY_COORDS = {
    # Small built-in lookup so the pipeline works without an extra network
    # hop for common cities; geocode_city() is used for anything else.
    "karachi": (24.8607, 67.0011),
    "lahore": (31.5497, 74.3436),
    "islamabad": (33.6844, 73.0479),
    "delhi": (28.6139, 77.2090),
    "london": (51.5072, -0.1276),
    "new york": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437),
    "beijing": (39.9042, 116.4074),
}


def geocode_city(city: str) -> tuple[float, float]:
    key = city.strip().lower()
    if key in CITY_COORDS:
        return CITY_COORDS[key]
    try:
        resp = requests.get(GEOCODE_URL, params={"name": city, "count": 1}, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results")
        if not results:
            raise ValueError(f"Could not geocode city '{city}'")
        return results[0]["latitude"], results[0]["longitude"]
    except Exception as e:
        if USE_SYNTHETIC_FALLBACK:
            print(f"[data_source] geocode failed ({e}); using placeholder coords for '{city}'")
            return 24.8607, 67.0011
        raise


def _fetch_live(city: str, start_date: str, end_date: str) -> pd.DataFrame:
    lat, lon = geocode_city(city)

    aq_resp = requests.get(
        AIR_QUALITY_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "pm2_5,pm10,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,us_aqi",
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "auto",
        },
        timeout=20,
    )
    aq_resp.raise_for_status()
    aq = aq_resp.json()["hourly"]

    wx_resp = requests.get(
        WEATHER_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation",
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "auto",
        },
        timeout=20,
    )
    wx_resp.raise_for_status()
    wx = wx_resp.json()["hourly"]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(aq["time"]),
        "pm2_5": aq.get("pm2_5"),
        "pm10": aq.get("pm10"),
        "ozone": aq.get("ozone"),
        "no2": aq.get("nitrogen_dioxide"),
        "so2": aq.get("sulphur_dioxide"),
        "co": aq.get("carbon_monoxide"),
        "us_aqi": aq.get("us_aqi"),
        "temperature": wx.get("temperature_2m"),
        "humidity": wx.get("relative_humidity_2m"),
        "wind_speed": wx.get("wind_speed_10m"),
        "pressure": wx.get("surface_pressure"),
        "precipitation": wx.get("precipitation"),
    })
    df["city"] = city
    return df


def _fetch_synthetic(city: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Physically-plausible synthetic generator, used only as an offline
    fallback. Produces a diurnal + seasonal AQI pattern with noise, plus
    correlated weather variables, so downstream feature engineering and
    model training have realistic signal to learn from.
    """
    rng = random.Random(hash(city) % (2**32))
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    hours = pd.date_range(start, end + timedelta(days=1) - timedelta(hours=1), freq="h")

    rows = []
    base_aqi = 70 + (hash(city) % 60)  # different baseline pollution per city
    for ts in hours:
        hour_factor = 1.0 + 0.35 * math.sin((ts.hour - 8) / 24 * 2 * math.pi)  # rush-hour bumps
        day_of_year = ts.timetuple().tm_yday
        seasonal_factor = 1.0 + 0.25 * math.sin((day_of_year / 365) * 2 * math.pi)
        noise = rng.gauss(0, 8)
        aqi = max(5, base_aqi * hour_factor * seasonal_factor + noise)

        temperature = 20 + 10 * math.sin((ts.hour - 6) / 24 * 2 * math.pi) + rng.gauss(0, 1.5)
        humidity = min(100, max(10, 55 - 0.6 * (temperature - 20) + rng.gauss(0, 5)))
        wind_speed = max(0, 8 + rng.gauss(0, 3) - 0.05 * aqi)
        pressure = 1013 + rng.gauss(0, 4)
        precipitation = max(0, rng.gauss(0.2, 0.6)) if rng.random() < 0.15 else 0.0

        rows.append({
            "timestamp": ts,
            "pm2_5": round(aqi * 0.6 + rng.gauss(0, 3), 1),
            "pm10": round(aqi * 0.9 + rng.gauss(0, 4), 1),
            "ozone": round(30 + rng.gauss(0, 8), 1),
            "no2": round(20 + rng.gauss(0, 5), 1),
            "so2": round(8 + rng.gauss(0, 2), 1),
            "co": round(300 + rng.gauss(0, 40), 1),
            "us_aqi": round(aqi, 1),
            "temperature": round(temperature, 1),
            "humidity": round(humidity, 1),
            "wind_speed": round(wind_speed, 1),
            "pressure": round(pressure, 1),
            "precipitation": round(precipitation, 2),
            "city": city,
        })
    return pd.DataFrame(rows)


def fetch_raw_data(city: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch raw hourly weather + pollutant data for `city` between
    `start_date` and `end_date` (inclusive, 'YYYY-MM-DD' strings).
    Tries the live Open-Meteo API first, falls back to synthetic data.
    """
    try:
        df = _fetch_live(city, start_date, end_date)
        if df.empty or df["us_aqi"].isna().all():
            raise ValueError("Live API returned no usable AQI data")
        print(f"[data_source] Fetched {len(df)} live rows for {city} ({start_date}..{end_date})")
        return df
    except Exception as e:
        if not USE_SYNTHETIC_FALLBACK:
            raise
        print(f"[data_source] Live fetch failed ({e}); using synthetic fallback for {city}")
        return _fetch_synthetic(city, start_date, end_date)


def fetch_forecast_weather(city: str, days: int = 3) -> pd.DataFrame:
    """Fetch (or synthesize) forward-looking weather, needed as model input
    for a true multi-day-ahead AQI forecast (we don't know future AQI,
    but weather forecasts are available)."""
    today = datetime.now(timezone.utc).date()
    start = today.isoformat()
    end = (today + timedelta(days=days)).isoformat()
    try:
        lat, lon = geocode_city(city)
        resp = requests.get(
            WEATHER_URL,
            params={
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation",
                "start_date": start, "end_date": end, "timezone": "auto",
            },
            timeout=20,
        )
        resp.raise_for_status()
        wx = resp.json()["hourly"]
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(wx["time"]),
            "temperature": wx["temperature_2m"],
            "humidity": wx["relative_humidity_2m"],
            "wind_speed": wx["wind_speed_10m"],
            "pressure": wx["surface_pressure"],
            "precipitation": wx["precipitation"],
        })
        df["city"] = city
        return df
    except Exception as e:
        print(f"[data_source] Forecast weather fetch failed ({e}); using synthetic fallback")
        synth = _fetch_synthetic(city, start, end)
        return synth[["timestamp", "temperature", "humidity", "wind_speed",
                       "pressure", "precipitation", "city"]]
