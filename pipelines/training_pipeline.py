#!/usr/bin/env python3
"""
pipelines/training_pipeline.py
--------------------------------
Step 1-2-3 of the project spec:
  1. Fetch historical (features, targets) from the Feature Store.
  2. Train and evaluate several ML models per forecast horizon
     (24h / 48h / 72h ahead AQI), picking the best by RMSE.
  3. Store the trained (best) model per horizon in the Model Registry.

Candidate models:
  - Ridge Regression        (fast, interpretable baseline)
  - Random Forest Regressor (handles non-linearity, gives feature importance)
  - Gradient Boosting Regressor (usually strongest of the three here)

Usage:
    python pipelines/training_pipeline.py --city Karachi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.feature_store import FeatureStore
from utils.features import TARGET_HORIZONS_HOURS, get_feature_columns
from utils.model_registry import ModelRegistry

CANDIDATE_MODELS = {
    "ridge": lambda: Ridge(alpha=1.0),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1
    ),
    "gradient_boosting": lambda: GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
    ),
}


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_one_horizon(df: pd.DataFrame, feature_cols: list[str], target_col: str,
                       registry: ModelRegistry, city: str) -> dict:
    data = df.dropna(subset=feature_cols + [target_col]).sort_values("timestamp")
    if len(data) < 50:
        print(f"  [!] Not enough rows ({len(data)}) to train {target_col}, skipping. "
              f"Run backfill.py with more --days.")
        return {}

    X = data[feature_cols].values
    y = data[target_col].values

    # time-based split (no shuffling) — this is a forecasting problem, so we
    # evaluate on the most recent slice to simulate real deployment
    split_idx = int(len(data) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    best_name, best_model, best_metrics, best_uses_scaler = None, None, None, False

    for name, factory in CANDIDATE_MODELS.items():
        model = factory()
        uses_scaler = name == "ridge"
        model.fit(X_train_scaled if uses_scaler else X_train, y_train)
        preds = model.predict(X_test_scaled if uses_scaler else X_test)
        metrics = evaluate(y_test, preds)
        results[name] = metrics
        print(f"  {name:>18s} | RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  R2={metrics['r2']:.3f}")

        if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
            best_name, best_model, best_metrics, best_uses_scaler = name, model, metrics, uses_scaler

    print(f"  -> best model for {target_col}: {best_name}")

    # Wrap scaler + model together when the winner needs scaling, so
    # inference code doesn't need to know which model type won.
    if best_uses_scaler:
        from sklearn.pipeline import make_pipeline
        final_model = make_pipeline(StandardScaler(), CANDIDATE_MODELS[best_name]())
        final_model.fit(X_train, y_train)
    else:
        final_model = best_model

    registry.save_model(
        name=f"aqi_forecast_{city.lower().replace(' ', '_')}_{target_col}",
        model=final_model,
        metrics=best_metrics,
        feature_names=feature_cols,
        target_name=target_col,
        extra={"algorithm": best_name, "all_candidates": results, "n_train": len(X_train), "n_test": len(X_test)},
    )
    return {"model": best_name, **best_metrics}


def run(city: str) -> None:
    fs = FeatureStore()
    fg = fs.get_or_create_feature_group("aqi_features", primary_key=["city", "date"])
    df = fg.read(city=city)

    if df.empty:
        print(f"[training_pipeline] No features found for '{city}'. "
              f"Run pipelines/backfill.py first.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    feature_cols = get_feature_columns(df)
    registry = ModelRegistry()

    print(f"[training_pipeline] {len(df)} feature rows for {city}, "
          f"{len(feature_cols)} candidate features")

    summary = {}
    for h in TARGET_HORIZONS_HOURS:
        target_col = f"target_aqi_{h}h"
        print(f"\n--- Horizon: {h}h ahead ---")
        summary[target_col] = train_one_horizon(df, feature_cols, target_col, registry, city)

    print("\n=== Training summary ===")
    for target, res in summary.items():
        if res:
            print(f"{target}: {res['model']} | RMSE={res['rmse']:.2f} MAE={res['mae']:.2f} R2={res['r2']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train + evaluate AQI forecast models, save best to registry.")
    parser.add_argument("--city", type=str, default="Karachi")
    args = parser.parse_args()
    run(city=args.city)
