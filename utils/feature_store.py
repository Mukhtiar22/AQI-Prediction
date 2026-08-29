"""
utils/feature_store.py
-----------------------
A tiny, local, file-based "feature store" that mimics the parts of the
Hopsworks / Vertex AI Feature Store API we actually need:

    fs = FeatureStore()
    fg = fs.get_or_create_feature_group("aqi_features", primary_key=["city", "date"])
    fg.insert(dataframe)                 # upsert new rows
    df = fg.read()                       # read everything back
    df = fg.read(city="Karachi")         # filter

Why local instead of a real hosted feature store?
    This repo is meant to run anywhere (including offline / in CI) without
    requiring API keys or account signup. The interface below is a thin
    enough wrapper that swapping it for `hopsworks.login()` +
    `fs.get_or_create_feature_group(...)` is a ~20 line change — see the
    "SWAPPING IN A REAL FEATURE STORE" note at the bottom of this file.

Storage layout:
    data/feature_store/<feature_group_name>.parquet
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_STORE_DIR = Path(os.environ.get("AQI_FEATURE_STORE_DIR", "data/feature_store"))


class FeatureGroup:
    def __init__(self, name: str, primary_key: list[str], store_dir: Path):
        self.name = name
        self.primary_key = primary_key
        self.store_dir = store_dir
        self.path = self.store_dir / f"{name}.parquet"

    def _load(self) -> pd.DataFrame:
        if self.path.exists():
            return pd.read_parquet(self.path)
        return pd.DataFrame()

    def insert(self, df: pd.DataFrame) -> None:
        """Upsert rows into the feature group, deduplicating on primary_key."""
        self.store_dir.mkdir(parents=True, exist_ok=True)
        existing = self._load()
        combined = pd.concat([existing, df], ignore_index=True)
        if self.primary_key:
            combined = combined.drop_duplicates(subset=self.primary_key, keep="last")
        combined = combined.sort_values(
            by=[c for c in ["city", "date"] if c in combined.columns]
        ).reset_index(drop=True)
        combined.to_parquet(self.path, index=False)
        print(f"[feature_store] '{self.name}' now has {len(combined)} rows "
              f"(+{len(df)} inserted) -> {self.path}")

    def read(self, **filters) -> pd.DataFrame:
        df = self._load()
        for col, val in filters.items():
            if col in df.columns:
                df = df[df[col] == val]
        return df.reset_index(drop=True)


class FeatureStore:
    def __init__(self, store_dir: str | Path = DEFAULT_STORE_DIR):
        self.store_dir = Path(store_dir)

    def get_or_create_feature_group(self, name: str, primary_key: list[str]) -> FeatureGroup:
        return FeatureGroup(name=name, primary_key=primary_key, store_dir=self.store_dir)


# ---------------------------------------------------------------------------
# SWAPPING IN A REAL FEATURE STORE (Hopsworks free tier)
# ---------------------------------------------------------------------------
# import hopsworks
# project = hopsworks.login()               # prompts for HOPSWORKS_API_KEY
# fs = project.get_feature_store()
# fg = fs.get_or_create_feature_group(
#     name="aqi_features", version=1, primary_key=["city", "date"],
#     description="Hourly AQI + weather features", online_enabled=True,
# )
# fg.insert(df)
# df = fg.read()
# Everything else in feature_pipeline.py / training_pipeline.py stays the same
# because we call the same `.insert()` / `.read()` methods.
# ---------------------------------------------------------------------------
