#!/usr/bin/env python3
"""
app/app.py
-----------
Interactive dashboard: loads the latest features + production models from
the feature store / model registry, shows current conditions, a 3-day
AQI forecast, historical trend, hazard alerts, and SHAP-based feature
importance for the forecast.

Run:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.aqi_levels import classify_aqi, is_hazardous
from utils.feature_store import FeatureStore
from utils.features import TARGET_HORIZONS_HOURS
from utils.model_registry import ModelRegistry

st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="wide")

HORIZON_LABELS = {24: "Tomorrow (+24h)", 48: "In 2 days (+48h)", 72: "In 3 days (+72h)"}


@st.cache_data(ttl=300)
def load_city_features(city: str) -> pd.DataFrame:
    fs = FeatureStore()
    fg = fs.get_or_create_feature_group("aqi_features", primary_key=["city", "date"])
    df = fg.read(city=city)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
    return df


@st.cache_resource
def load_models(city: str):
    registry = ModelRegistry()
    models = {}
    for h in TARGET_HORIZONS_HOURS:
        model_name = f"aqi_forecast_{city.lower().replace(' ', '_')}_target_aqi_{h}h"
        try:
            model, metadata = registry.load_production_model(model_name)
            models[h] = (model, metadata)
        except FileNotFoundError:
            pass
    return models


def main():
    st.title("🌫️ AQI Predictor")
    st.caption("Serverless, feature-store-backed Air Quality forecasting — next 3 days.")

    with st.sidebar:
        st.header("Settings")
        city = st.text_input("City", value="Karachi")
        st.markdown("---")
        st.markdown(
            "**Pipeline status**\n\n"
            "This dashboard reads directly from the local feature store & "
            "model registry that `pipelines/feature_pipeline.py` and "
            "`pipelines/training_pipeline.py` write to."
        )

    df = load_city_features(city)
    if df.empty:
        st.warning(
            f"No feature data found for **{city}** yet.\n\n"
            f"Run:\n```\npython pipelines/backfill.py --cities \"{city}\" --days 60\n"
            f"python pipelines/training_pipeline.py --city \"{city}\"\n```"
        )
        return

    models = load_models(city)
    latest = df.iloc[-1]

    # ---- Current conditions -------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    current_aqi = latest["us_aqi"]
    cat = classify_aqi(current_aqi)
    col1.metric("Current US AQI", f"{current_aqi:.0f}", help=cat["category"])
    col2.metric("PM2.5", f"{latest.get('pm2_5', float('nan')):.1f} µg/m³")
    col3.metric("Temperature", f"{latest.get('temperature', float('nan')):.1f} °C")
    col4.metric("Humidity", f"{latest.get('humidity', float('nan')):.0f}%")

    st.markdown(
        f"<div style='padding:10px;border-radius:8px;background-color:{cat['color']}22;"
        f"border-left:6px solid {cat['color']}'>"
        f"<b>{cat['category']}</b> — {cat['advice']}</div>",
        unsafe_allow_html=True,
    )

    # ---- Forecast -------------------------------------------------------------
    st.subheader("3-Day Forecast")
    if not models:
        st.info("No trained models found for this city yet. Run `training_pipeline.py`.")
    else:
        feature_row = df.iloc[[-1]]
        forecast_cols = st.columns(len(models))
        alerts = []
        forecast_points = [(latest["timestamp"], current_aqi, "now")]

        for i, h in enumerate(sorted(models.keys())):
            model, metadata = models[h]
            feat_cols = metadata["feature_names"]
            X = feature_row.reindex(columns=feat_cols, fill_value=0)
            pred = float(model.predict(X)[0])
            pred_cat = classify_aqi(pred)

            with forecast_cols[i]:
                st.metric(HORIZON_LABELS.get(h, f"+{h}h"), f"{pred:.0f}", help=pred_cat["category"])
                st.markdown(
                    f"<span style='color:{pred_cat['color']}'>●</span> {pred_cat['category']}",
                    unsafe_allow_html=True,
                )

            forecast_points.append((latest["timestamp"] + pd.Timedelta(hours=h), pred, f"+{h}h"))
            if is_hazardous(pred):
                alerts.append((h, pred, pred_cat["category"]))

        if alerts:
            for h, pred, cat_label in alerts:
                st.error(
                    f"⚠️ **Hazardous air quality alert:** AQI predicted to reach "
                    f"**{pred:.0f} ({cat_label})** in {h}h. Consider limiting outdoor "
                    f"exposure, especially for sensitive groups."
                )

        # forecast chart
        fc_df = pd.DataFrame(forecast_points, columns=["timestamp", "aqi", "label"])
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=fc_df["timestamp"], y=fc_df["aqi"], mode="lines+markers+text",
            text=fc_df["label"], textposition="top center", name="Forecast",
            line=dict(color="#1f77b4", width=3),
        ))
        fig_fc.update_layout(
            height=300, margin=dict(l=10, r=10, t=30, b=10),
            yaxis_title="US AQI", xaxis_title=None,
        )
        st.plotly_chart(fig_fc, use_container_width=True)

    # ---- Historical trend -------------------------------------------------------------
    st.subheader("Historical AQI Trend")
    hist = df.tail(24 * 14)  # last 14 days
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=hist["timestamp"], y=hist["us_aqi"], mode="lines", name="US AQI",
        line=dict(color="#ff7f0e"), fill="tozeroy",
    ))
    for low, high, label, color, _ in [
        (0, 50, "Good", "#00e400", None), (51, 100, "Moderate", "#ffff00", None),
        (101, 150, "USG", "#ff7e00", None), (151, 200, "Unhealthy", "#ff0000", None),
    ]:
        fig_hist.add_hrect(y0=low, y1=high, fillcolor=color, opacity=0.08, line_width=0)
    fig_hist.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="US AQI")
    st.plotly_chart(fig_hist, use_container_width=True)

    # ---- Feature importance (SHAP) -------------------------------------------------------------
    st.subheader("What's driving the 24h forecast? (SHAP)")
    if 24 in models:
        try:
            import shap
            model, metadata = models[24]
            feat_cols = metadata["feature_names"]
            sample = df.dropna(subset=feat_cols).tail(200)
            X_sample = sample[feat_cols]

            # tree explainer for tree models, fallback to KernelExplainer-free
            # permutation-style explainer for pipelines (e.g. scaled Ridge)
            underlying = model.steps[-1][1] if hasattr(model, "steps") else model
            if hasattr(underlying, "feature_importances_"):
                explainer = shap.TreeExplainer(underlying)
                X_for_shap = model[:-1].transform(X_sample) if hasattr(model, "steps") else X_sample
                shap_values = explainer.shap_values(X_for_shap)
            else:
                explainer = shap.Explainer(model.predict, X_sample.iloc[:50])
                shap_values = explainer(X_sample.iloc[:50]).values
                X_sample = X_sample.iloc[:50]

            import numpy as np
            mean_abs_shap = pd.Series(
                abs(shap_values).mean(axis=0), index=feat_cols
            ).sort_values(ascending=False).head(12)

            fig_shap = go.Figure(go.Bar(
                x=mean_abs_shap.values[::-1], y=mean_abs_shap.index[::-1], orientation="h",
            ))
            fig_shap.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10),
                                    xaxis_title="mean |SHAP value|")
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as e:
            st.info(f"SHAP explanation unavailable: {e}")
    else:
        st.info("Train the 24h model first to see feature importance.")

    with st.expander("Model metadata"):
        for h, (model, metadata) in models.items():
            st.json({
                "horizon": f"+{h}h",
                "algorithm": metadata["extra"].get("algorithm"),
                "metrics": metadata["metrics"],
                "trained_at_utc": metadata["trained_at_utc"],
                "version": metadata["version"],
            })


if __name__ == "__main__":
    main()
