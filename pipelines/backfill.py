#!/usr/bin/env python3
"""
pipelines/backfill.py
----------------------
Backfills historical (features, targets) by running the feature
generation logic over a wide date range in one shot — used once (or
whenever you add a new city) to build enough training data for the
model, rather than waiting for the hourly pipeline to accumulate it.

Usage:
    python pipelines/backfill.py --cities "Karachi,Lahore" --days 120
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


def run(cities: list[str], days: int, chunk_days: int = 30) -> None:
    fs = FeatureStore()
    fg = fs.get_or_create_feature_group("aqi_features", primary_key=["city", "date"])

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days)

    for city in cities:
        print(f"\n=== Backfilling {days} days for {city} ===")
        cursor = start_date
        all_raw = []
        # fetch in chunks to be polite to the API / keep payloads small
        while cursor <= end_date:
            chunk_end = min(cursor + timedelta(days=chunk_days), end_date)
            print(f"  fetching {cursor} -> {chunk_end}")
            raw = fetch_raw_data(city, start_date=cursor.isoformat(), end_date=chunk_end.isoformat())
            all_raw.append(raw)
            cursor = chunk_end + timedelta(days=1)

        if not all_raw:
            continue

        import pandas as pd
        raw_df = pd.concat(all_raw, ignore_index=True).drop_duplicates(subset=["city", "timestamp"])
        feature_df = build_feature_table(raw_df, for_inference=False)
        feature_df = feature_df.dropna(subset=["aqi_lag_72h", "target_aqi_72h"])

        print(f"  -> {len(feature_df)} complete feature rows generated for {city}")
        if not feature_df.empty:
            fg.insert(feature_df)

    print("\n[backfill] Done. Run pipelines/training_pipeline.py next.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical features + targets.")
    parser.add_argument("--cities", type=str, default=",".join(DEFAULT_CITIES))
    parser.add_argument("--days", type=int, default=120, help="How many past days to backfill")
    args = parser.parse_args()

    run(cities=[c.strip() for c in args.cities.split(",") if c.strip()], days=args.days)
