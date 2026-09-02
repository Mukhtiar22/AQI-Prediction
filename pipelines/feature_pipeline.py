#!/usr/bin/env python3
"""
pipelines/feature_pipeline.py
------------------------------
Step 1-2-3 of the project spec:
  1. Fetch raw weather + pollutant data from an external API (Open-Meteo).
  2. Compute features (time-based, lag, rolling, AQI change rate) and
     targets (AQI 24h/48h/72h ahead).
  3. Store features in the feature store.

Run hourly in production (see .github/workflows/pipelines.yml).

Usage:
    python pipelines/feature_pipeline.py --cities "Karachi,Lahore" --lookback-hours 96
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.data_source import fetch_raw_data
from utils.features import build_feature_table
from utils.feature_store import FeatureStore

DEFAULT_CITIES = ["Karachi"]


def run(cities: list[str], lookback_hours: int) -> None:
    end = datetime.now(timezone.utc).date()
    start = (end - timedelta(hours=lookback_hours)).__str__() if lookback_hours < 24 else \
        (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).date().isoformat()

    fs = FeatureStore()
    fg = fs.get_or_create_feature_group("aqi_features", primary_key=["city", "date"])

    for city in cities:
        print(f"\n=== Feature pipeline: {city} ===")
        raw_df = fetch_raw_data(city, start_date=start, end_date=end.isoformat())
        if raw_df.empty:
            print(f"[feature_pipeline] No data returned for {city}, skipping")
            continue

        feature_df = build_feature_table(raw_df, for_inference=False)
        # Drop rows where lag features couldn't be computed (start of window)
        # or targets are unknown (end of window, since 72h-ahead AQI isn't
        # available yet for the most recent hours). This is expected and
        # fine — the backfill script covers a wide historical range so
        # plenty of complete rows accumulate over time.
        feature_df = feature_df.dropna(subset=["aqi_lag_72h"])
        if feature_df.empty:
            print(f"[feature_pipeline] Not enough history yet for complete rows in {city}")
            continue

        fg.insert(feature_df)

    print("\n[feature_pipeline] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch raw data, build features, store them.")
    parser.add_argument("--cities", type=str, default=",".join(DEFAULT_CITIES),
                         help="Comma-separated city names")
    parser.add_argument("--lookback-hours", type=int, default=24 * 10,
                         help="How many hours of raw history to (re)fetch each run")
    args = parser.parse_args()

    run(cities=[c.strip() for c in args.cities.split(",") if c.strip()],
        lookback_hours=args.lookback_hours)
