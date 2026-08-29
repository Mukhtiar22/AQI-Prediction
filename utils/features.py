"""
utils/features.py
------------------
Turns raw hourly (weather, pollutant) rows into a model-ready feature
table with:
  - time-based features (hour, day, month, day-of-week, weekend flag,
    cyclical sin/cos encodings)
  - lag features (AQI 1h, 24h, 72h ago)
  - rolling statistics (6h, 24h mean/std)
  - AQI change rate (derivative)
  - targets: AQI 24h / 48h / 72h ahead (the "next 3 days" forecast targets)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAG_HOURS = [1, 6, 24, 48, 72]
ROLLING_WINDOWS = [6, 24]
TARGET_HORIZONS_HOURS = [24, 48, 72]  # next 1, 2, 3 days


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["day"] = ts.dt.day
    df["month"] = ts.dt.month
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    # cyclical encodings so the model understands hour 23 is close to hour 0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    for lag in LAG_HOURS:
        df[f"aqi_lag_{lag}h"] = df["us_aqi"].shift(lag)
    for window in ROLLING_WINDOWS:
        df[f"aqi_roll_mean_{window}h"] = df["us_aqi"].rolling(window).mean()
        df[f"aqi_roll_std_{window}h"] = df["us_aqi"].rolling(window).std()
    # AQI change rate (first derivative), a directly-requested feature
    df["aqi_change_rate_1h"] = df["us_aqi"].diff(1)
    df["aqi_change_rate_24h"] = df["us_aqi"].diff(24)
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for h in TARGET_HORIZONS_HOURS:
        df[f"target_aqi_{h}h"] = df["us_aqi"].shift(-h)
    return df


def build_feature_table(raw_df: pd.DataFrame, for_inference: bool = False) -> pd.DataFrame:
    """
    Full pipeline: raw hourly rows -> engineered features (+ targets, unless
    for_inference=True, since future AQI obviously isn't known yet).
    Rows are grouped/processed per city so lags don't leak across cities.
    """
    out = []
    for city, city_df in raw_df.groupby("city"):
        df = add_time_features(city_df)
        df = add_lag_and_rolling_features(df)
        if not for_inference:
            df = add_targets(df)
        out.append(df)
    result = pd.concat(out, ignore_index=True)
    result["date"] = result["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return result


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"timestamp", "date", "city"} | {c for c in df.columns if c.startswith("target_")}
    return [c for c in df.columns if c not in exclude]
