#!/usr/bin/env python3
"""Premium AQI intelligence dashboard backed by the existing feature store and model registry."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.aqi_levels import classify_aqi, is_hazardous
from utils.feature_store import FeatureStore
from utils.features import TARGET_HORIZONS_HOURS
from utils.model_registry import ModelRegistry

st.set_page_config(page_title="AI-Based AQI Predictor", page_icon="🌫️", layout="wide")

CITIES = ["Karachi", "Sukkur", "Naushahro Feroze"]
HORIZON_LABELS = {24: "+24H", 48: "+48H", 72: "+72H"}
AQI_SEVERITY_BANDS = [
    (0, 50, "Good", "#17c964"),
    (51, 100, "Moderate", "#ffd166"),
    (101, 150, "Sensitive", "#ff9f1c"),
    (151, 200, "Unhealthy", "#ef476f"),
    (201, 300, "Very Unhealthy", "#9b5de5"),
    (301, 500, "Hazardous", "#7e0023"),
]


@st.cache_data(ttl=300)
def load_city_features(city: str) -> pd.DataFrame:
    fs = FeatureStore()
    fg = fs.get_or_create_feature_group("aqi_features", primary_key=["city", "date"])
    df = fg.read(city=city)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


@st.cache_resource
def load_models(city: str):
    registry = ModelRegistry()
    models = {}
    for h in TARGET_HORIZONS_HOURS:
        model_name = f"aqi_forecast_{city.lower().replace(' ', '_')}_target_aqi_{h}h"
        try:
            model, metadata = registry.load_production_model(model_name)
            models[h] = {"model": model, "metadata": metadata, "name": model_name}
        except FileNotFoundError:
            continue
    return models


def safe_float(value, default=None):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        val = float(value)
        return val if np.isfinite(val) else default
    except (TypeError, ValueError):
        return default


def format_value(value, digits=1, suffix=""):
    val = safe_float(value)
    if val is None:
        return "N/A"
    return f"{val:.{digits}f}{suffix}"


def delta_badge(current, previous, digits=1):
    if current is None or previous is None or previous == 0:
        return "N/A"
    delta = ((current - previous) / abs(previous)) * 100
    if abs(delta) < 0.05:
        return "±0.0%"
    direction = "▲" if delta > 0 else "▼"
    return f"{direction} {abs(delta):.{digits}f}%"


def render_sidebar():
    with st.sidebar:
        st.markdown(
            "<div class='brand-block'><div class='brand'>AI-BASED AQI PREDICTOR</div><div class='subtitle'>AIR QUALITY FORECASTING</div><div class='subtle'>AI-Powered Air Quality Forecasting</div></div>",
            unsafe_allow_html=True,
        )
        nav = st.radio(
            "Navigation",
            ["Overview", "AQI Forecast", "Historical Analytics", "Pollutants", "Model Intelligence", "Explainability", "Pipeline"],
        )
        st.markdown("---")
        st.markdown("### Select City")
        city = st.selectbox("", CITIES, index=0)
        st.markdown("### System Status")
        df_for_status = load_city_features(city)
        fs_status = "ONLINE" if not df_for_status.empty else "OFFLINE"
        model_status = "ONLINE" if load_models(city) else "OFFLINE"
        st.markdown(f"<div class='status-row'><span>Feature Store</span><span class='status online'>● {fs_status}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='status-row'><span>Model Registry</span><span class='status online'>● {model_status}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='status-row'><span>Prediction Engine</span><span class='status online'>● {model_status}</span></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.caption("Reads from the project feature store and production model registry without inventing values.")
        return city, nav


def render_header(city: str, df: pd.DataFrame, models: dict):
    latest = df.iloc[-1] if not df.empty else None
    model_version = "N/A"
    if models:
        first_h = sorted(models.keys())[0]
        model_version = models[first_h].get("metadata", {}).get("version", "N/A")
    last_update = latest["timestamp"] if latest is not None and "timestamp" in latest.index else "N/A"
    st.markdown("<div class='section-header'>Air Quality Intelligence</div>", unsafe_allow_html=True)
    st.caption("Real-time conditions and AI-powered 72-hour AQI forecasting")
    header_cols = st.columns([1.6, 1.4, 1.3, 1.2, 1.1])
    with header_cols[0]:
        st.markdown(f"<div class='meta-card'><div class='meta-label'>Selected City</div><div class='meta-value'>{city}</div></div>", unsafe_allow_html=True)
    with header_cols[1]:
        st.markdown(f"<div class='meta-card'><div class='meta-label'>Last Data Update</div><div class='meta-value'>{last_update}</div></div>", unsafe_allow_html=True)
    with header_cols[2]:
        st.markdown(f"<div class='meta-card'><div class='meta-label'>Model Version</div><div class='meta-value'>{model_version}</div></div>", unsafe_allow_html=True)
    with header_cols[3]:
        st.markdown("<div class='meta-card'><div class='meta-label'>System Status</div><div class='meta-value status-on'>ONLINE</div></div>", unsafe_allow_html=True)
    with header_cols[4]:
        st.markdown("<div class='meta-card'><div class='meta-label'>Forecast</div><div class='meta-value'>72h</div></div>", unsafe_allow_html=True)


def render_current_aqi(df: pd.DataFrame):
    if df.empty:
        st.warning("No feature data is currently available for this city. Please run the feature/backfill pipeline.")
        return None

    latest = df.iloc[-1]
    current_aqi = safe_float(latest.get("us_aqi"))
    cat = classify_aqi(current_aqi) if current_aqi is not None else {"category": "N/A", "color": "#b0bac9", "advice": "AQI data unavailable."}
    last_timestamp = latest.get("timestamp")
    prev_row = df.iloc[-2] if len(df) > 1 else None
    prev_aqi = safe_float(prev_row.get("us_aqi")) if prev_row is not None else None
    delta = delta_badge(current_aqi, prev_aqi)

    st.markdown(
        f"""
        <div class="hero-card" style="border-left: 6px solid {cat['color']}; background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));">
          <div class="eyebrow">CURRENT AQI</div>
          <div class="hero-score" style="color: {cat['color']};">{format_value(current_aqi, digits=0)}</div>
          <div class="hero-category">{cat['category']}</div>
          <div class="hero-meta">{cat['advice']} · Updated {last_timestamp}</div>
          <div class="hero-change">Change vs previous reading: {delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return latest


def render_kpi_cards(df: pd.DataFrame):
    if df.empty:
        return
    latest = df.iloc[-1]
    prev_row = df.iloc[-2] if len(df) > 1 else None
    metrics = [
        ("Current AQI", "us_aqi", 0, "AQI"),
        ("PM2.5", "pm2_5", 1, "µg/m³"),
        ("PM10", "pm10", 1, "µg/m³"),
        ("Temperature", "temperature", 1, "°C"),
        ("Humidity", "humidity", 0, "%"),
    ]
    cards = st.columns(5)
    for idx, (label, field, digits, suffix) in enumerate(metrics):
        current = safe_float(latest.get(field))
        previous = safe_float(prev_row.get(field)) if prev_row is not None else None
        with cards[idx]:
            st.markdown(
                f"<div class='stat-card'><div class='stat-label'>{label}</div><div class='stat-value'>{format_value(current, digits, suffix)}</div><div class='stat-trend'>{delta_badge(current, previous, 1)}</div></div>",
                unsafe_allow_html=True,
            )


def render_aqi_severity(current_aqi: float | None):
    if current_aqi is None:
        st.info("AQI severity cannot be plotted because no current value is available.")
        return
    cat = classify_aqi(current_aqi)
    marker_pct = min(max(current_aqi / 500.0 * 100, 0), 100)
    st.markdown("### AQI Severity")
    band_markup = []
    for low, high, label, color in AQI_SEVERITY_BANDS:
        start = low / 500 * 100
        width = ((high - low) / 500) * 100
        band_markup.append(f"<div class='band' style='background:{color}; width:{width}%; left:{start}%;'></div>")
    st.markdown(
        f"""
        <div class='severity-wrap'>
          <div class='severity-scale'>
            {''.join(band_markup)}
            <div class='marker' style='left:{marker_pct}%'></div>
          </div>
          <div class='severity-labels'>
            {''.join(f"<span>{label}</span>" for _, _, label, _ in AQI_SEVERITY_BANDS)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Current AQI: {current_aqi:.0f} · {cat['category']}")


def render_forecast_section(city: str, df: pd.DataFrame, models: dict):
    if df.empty:
        return
    latest = df.iloc[-1]
    latest_ts = latest.get("timestamp")
    if latest_ts is None:
        st.info("Forecast data is unavailable because the latest timestamp is missing.")
        return
    current_aqi = safe_float(latest.get("us_aqi"))
    forecast_rows = [{"timestamp": latest_ts, "aqi": current_aqi, "label": "NOW"}]
    predictions = []

    for h in sorted(models.keys()):
        model_spec = models[h]
        model = model_spec["model"]
        metadata = model_spec.get("metadata", {})
        feat_cols = metadata.get("feature_names", [])
        if not feat_cols:
            continue
        feature_row = df.iloc[[-1]].copy().reindex(columns=feat_cols, fill_value=0)
        pred_val = safe_float(model.predict(feature_row)[0])
        pred_cat = classify_aqi(pred_val) if pred_val is not None else {"category": "N/A", "color": "#b0bac9"}
        predictions.append({"horizon": h, "aqi": pred_val, "category": pred_cat, "ts": latest_ts + pd.Timedelta(hours=h)})
        forecast_rows.append({"timestamp": latest_ts + pd.Timedelta(hours=h), "aqi": pred_val, "label": HORIZON_LABELS.get(h, f"+{h}h")})

    st.markdown("### AI-Powered 72-Hour Forecast")
    if not predictions:
        st.info("No production model is available for this city yet. Train the model registry to populate forecast outputs.")
        return

    cards = st.columns(4)
    overview_items = [{"label": "NOW", "aqi": current_aqi, "category": classify_aqi(current_aqi) if current_aqi is not None else {"category": "N/A", "color": "#b0bac9"}}]
    overview_items.extend({"label": HORIZON_LABELS.get(p["horizon"], f"+{p['horizon']}h"), "aqi": p["aqi"], "category": p["category"]} for p in predictions)
    for idx, row in enumerate(overview_items):
        with cards[idx]:
            st.markdown(
                f"<div class='forecast-card'><div class='forecast-label'>{row['label']}</div><div class='forecast-aqi' style='color:{row['category']['color']}'>{format_value(row['aqi'], 0)}</div><div class='forecast-status'>{row['category']['category']}</div></div>",
                unsafe_allow_html=True,
            )

    fc_df = pd.DataFrame(forecast_rows)
    fig = go.Figure()
    hist = df.tail(240)
    fig.add_trace(go.Scatter(x=hist["timestamp"], y=hist["us_aqi"], mode="lines", name="Historical AQI", line=dict(color="#8ecae6", width=2), fill="tozeroy"))
    fig.add_trace(go.Scatter(x=fc_df["timestamp"], y=fc_df["aqi"], mode="lines+markers", name="AI Forecast", line=dict(color="#ffb703", width=3, dash="dash"), marker=dict(size=8)))
    for low, high, _, color in AQI_SEVERITY_BANDS:
        fig.add_hrect(y0=low, y1=high, fillcolor=color, opacity=0.06, line_width=0)
    fig.update_layout(
        template="plotly_dark",
        title="AQI Forecast vs Historical Readings",
        xaxis_title="Timestamp",
        yaxis_title="United States AQI",
        height=420,
        margin=dict(l=15, r=15, t=35, b=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    alert_msgs = []
    for pred in predictions:
        if pred["aqi"] is not None and is_hazardous(pred["aqi"]):
            h_label = HORIZON_LABELS.get(pred["horizon"], f"+{pred['horizon']}h")
            alert_msgs.append(f"{h_label} forecast exceeds hazardous threshold at AQI {pred['aqi']:.0f}.")
    if alert_msgs:
        for msg in alert_msgs:
            st.warning(msg)
    else:
        st.success("No hazardous AQI thresholds are forecast for the selected city across the next 72 hours.")


def render_historical_analysis(city: str, df: pd.DataFrame):
    if df.empty:
        st.info("Historical data is unavailable for this city.")
        return
    windows = {"Last 24 hours": 24, "Last 7 days": 7 * 24, "Last 14 days": 14 * 24, "Last 30 days": 30 * 24}
    selection = st.selectbox("Select time window", list(windows.keys()), key="history-window")
    history_rows = df.tail(windows[selection])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_rows["timestamp"], y=history_rows["us_aqi"], mode="lines", name="AQI", line=dict(color="#f72585", width=3)))
    fig.add_trace(go.Scatter(x=history_rows["timestamp"], y=history_rows["us_aqi"].rolling(12, min_periods=1).mean(), mode="lines", name="12h Moving Average", line=dict(color="#80ed99", width=2, dash="dot")))
    for low, high, _, color in AQI_SEVERITY_BANDS:
        fig.add_hrect(y0=low, y1=high, fillcolor=color, opacity=0.05, line_width=0)
    fig.update_layout(template="plotly_dark", xaxis_title="Timestamp", yaxis_title="United States AQI", height=400, margin=dict(l=15, r=15, t=15, b=15))
    st.plotly_chart(fig, use_container_width=True)


def render_pollutant_analysis(df: pd.DataFrame):
    if df.empty:
        st.info("Pollutant data is unavailable.")
        return
    pollutant_cols = [col for col in ["pm2_5", "pm10", "ozone", "no2", "so2", "co"] if col in df.columns]
    if not pollutant_cols:
        st.info("No pollutant columns are available in the current feature store dataset.")
        return
    latest = df.iloc[-1]
    cols = st.columns(min(6, len(pollutant_cols)))
    for idx, col in enumerate(pollutant_cols):
        with cols[idx]:
            suffix = " µg/m³" if col != "co" else " ppm"
            st.markdown(f"<div class='mini-card'><div class='mini-label'>{col.upper()}</div><div class='mini-value'>{format_value(latest.get(col), 1, suffix)}</div></div>", unsafe_allow_html=True)
    pollutant = st.selectbox("Pollutant trend", pollutant_cols, key="pollutant-select")
    plot_df = df.tail(120)
    fig = go.Figure(data=go.Scatter(x=plot_df["timestamp"], y=plot_df[pollutant], mode="lines+markers", line=dict(color="#4cc9f0", width=3)))
    fig.update_layout(template="plotly_dark", title=f"{pollutant.upper()} Trend", xaxis_title="Timestamp", yaxis_title=pollutant.upper(), height=360, margin=dict(l=15, r=15, t=35, b=15))
    st.plotly_chart(fig, use_container_width=True)

    comp_df = pd.DataFrame({"Pollutant": pollutant_cols, "Value": [latest.get(col) for col in pollutant_cols]})
    comp_df = comp_df.dropna(subset=["Value"]).reset_index(drop=True)
    if not comp_df.empty:
        fig2 = go.Figure(go.Bar(x=comp_df["Pollutant"], y=comp_df["Value"], marker_color="#5eead4"))
        fig2.update_layout(template="plotly_dark", title="Current Pollutant Snapshot", height=300, margin=dict(l=15, r=15, t=35, b=15))
        st.plotly_chart(fig2, use_container_width=True)


def render_weather_analysis(df: pd.DataFrame):
    if df.empty:
        st.info("Weather data is unavailable.")
        return
    weather_cols = [col for col in ["temperature", "humidity", "wind_speed", "pressure", "precipitation"] if col in df.columns]
    if not weather_cols:
        st.info("No weather columns are available in the current feature store dataset.")
        return
    for col in weather_cols:
        fig = go.Figure(data=go.Scatter(x=df["timestamp"], y=df[col], mode="lines", line=dict(color="#90be6d", width=2)))
        fig.update_layout(template="plotly_dark", title=f"{col.replace('_', ' ').title()} Trend", xaxis_title="Timestamp", yaxis_title=col.replace('_', ' ').title(), height=280, margin=dict(l=15, r=15, t=35, b=15))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AQI vs Weather Relationships")
    relationship_cols = [col for col in weather_cols if df[col].notna().any()]
    if relationship_cols:
        cols = st.columns(min(3, len(relationship_cols)))
        for idx, col in enumerate(relationship_cols[:3]):
            with cols[idx]:
                fig = go.Figure(data=go.Scatter(x=df[col], y=df["us_aqi"], mode="markers", marker=dict(color="#ffb703", size=8)))
                fig.update_layout(template="plotly_dark", title=f"AQI vs {col.replace('_', ' ').title()}", xaxis_title=col.replace('_', ' ').title(), yaxis_title="AQI", height=260, margin=dict(l=15, r=15, t=35, b=15))
                st.plotly_chart(fig, use_container_width=True)


def render_city_comparison():
    rows = []
    for city in CITIES:
        df = load_city_features(city)
        latest = df.iloc[-1] if not df.empty else None
        if latest is None:
            rows.append({"city": city, "aqi": None, "category": "Data unavailable"})
            continue
        current_aqi = safe_float(latest.get("us_aqi"))
        category = classify_aqi(current_aqi)["category"] if current_aqi is not None else "N/A"
        rows.append({"city": city, "aqi": current_aqi, "category": category})

    comparison = pd.DataFrame(rows)
    colors = []
    for aid in comparison["aqi"]:
        if aid is None:
            colors.append("#94a3b8")
        elif aid <= 100:
            colors.append("#80ed99")
        elif aid <= 200:
            colors.append("#ffb703")
        else:
            colors.append("#ef476f")
    fig = go.Figure(go.Bar(x=comparison["city"], y=comparison["aqi"], text=comparison["aqi"].apply(lambda x: "N/A" if x is None else f"{x:.0f}"), textposition="outside", marker_color=colors))
    fig.update_layout(template="plotly_dark", title="Pakistan Air Quality Overview", height=350, margin=dict(l=15, r=15, t=35, b=15), xaxis_title="City", yaxis_title="Current AQI")
    st.plotly_chart(fig, use_container_width=True)

    ranked = comparison.dropna(subset=["aqi"]).sort_values("aqi", ascending=False).reset_index(drop=True)
    st.markdown("### City Ranking")
    if ranked.empty:
        st.info("No city AQI values are currently available.")
        return
    labels = ["Most polluted", "Second", "Cleanest"]
    for idx, row in ranked.head(3).iterrows():
        st.markdown(f"<div class='rank-card'><div class='rank-position'>{labels[idx]}</div><div class='rank-city'>{row['city']}</div><div class='rank-aqi'>{format_value(row['aqi'], 0)}</div></div>", unsafe_allow_html=True)


def render_alerts(city: str, df: pd.DataFrame, models: dict):
    st.markdown("### Air Quality Alerts")
    if df.empty or not models:
        st.info("No alert evaluation is possible without the required feature and model data.")
        return
    alerts = []
    latest = df.iloc[-1]
    current_aqi = safe_float(latest.get("us_aqi"))
    if current_aqi is not None and is_hazardous(current_aqi):
        alerts.append(("Current conditions", current_aqi, "Current AQI is in the hazardous range."))
    for h in sorted(models.keys()):
        model = models[h]["model"]
        feat_cols = models[h]["metadata"].get("feature_names", [])
        if not feat_cols:
            continue
        feature_row = df.iloc[[-1]].copy().reindex(columns=feat_cols, fill_value=0)
        pred_val = safe_float(model.predict(feature_row)[0])
        if pred_val is not None and is_hazardous(pred_val):
            alerts.append((HORIZON_LABELS.get(h, f"+{h}h"), pred_val, "Hazardous AQI forecast detected."))
    if not alerts:
        st.success("✓ No hazardous AQI forecasts were detected for the current city.")
        return
    for horizon, value, message in alerts:
        cat = classify_aqi(value)
        st.markdown(
            f"<div class='alert-card' style='border-left: 6px solid {cat['color']};'><div class='alert-heading'>{horizon}</div><div class='alert-value'>{value:.0f}</div><div class='alert-msg'>{message}</div></div>",
            unsafe_allow_html=True,
        )


def render_model_performance(city: str, models: dict):
    st.markdown("### Model Intelligence")
    if not models:
        st.info("No model metadata is available for this city. Train the models to populate the registry.")
        return
    rows = []
    for h in sorted(models.keys()):
        metadata = models[h].get("metadata", {})
        metrics = metadata.get("metrics", {})
        rows.append({
            "Horizon": HORIZON_LABELS.get(h, f"+{h}h"),
            "Algorithm": metadata.get("extra", {}).get("algorithm", "N/A"),
            "Version": metadata.get("version", "N/A"),
            "RMSE": metrics.get("rmse", "N/A"),
            "MAE": metrics.get("mae", "N/A"),
            "R²": metrics.get("r2", "N/A"),
            "Trained": metadata.get("trained_at_utc", "N/A"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    with st.expander("View raw metadata"):
        for h in sorted(models.keys()):
            st.json(models[h].get("metadata", {}))


def render_shap_analysis(city: str, df: pd.DataFrame, models: dict):
    st.markdown("### Why is AQI expected to change?")
    if 24 not in models:
        st.info("The 24h model is required for SHAP explainability. Train the 24h horizon model first.")
        return
    try:
        import shap
        model_spec = models[24]
        model = model_spec["model"]
        metadata = model_spec.get("metadata", {})
        feature_cols = metadata.get("feature_names", [])
        if not feature_cols:
            st.info("The model metadata is missing feature names, so SHAP cannot be computed.")
            return
        sample = df.dropna(subset=feature_cols).tail(200)
        if sample.empty:
            st.info("Not enough feature rows are available to compute SHAP values for this city.")
            return
        X_sample = sample[feature_cols]
        try:
            underlying = model.steps[-1][1] if hasattr(model, "steps") else model
            if hasattr(underlying, "feature_importances_"):
                explainer = shap.TreeExplainer(underlying)
                X_for_shap = model[:-1].transform(X_sample) if hasattr(model, "steps") else X_sample
                shap_values = explainer.shap_values(X_for_shap)
                values = np.abs(np.asarray(shap_values)).mean(axis=0)
            else:
                explainer = shap.Explainer(model.predict, X_sample.iloc[:50])
                shap_array = explainer(X_sample.iloc[:50])
                values = np.abs(np.asarray(shap_array.values)).mean(axis=0)
                X_sample = X_sample.iloc[:50]
        except Exception:
            explainer = shap.Explainer(model.predict, X_sample)
            shap_array = explainer(X_sample)
            values = np.abs(np.asarray(shap_array.values)).mean(axis=0)
        importance = pd.Series(values, index=feature_cols).sort_values(ascending=False).head(10)
        fig = go.Figure(go.Bar(x=importance.values[::-1], y=importance.index[::-1], orientation="h", marker_color="#7cc6fe"))
        fig.update_layout(template="plotly_dark", height=380, margin=dict(l=15, r=15, t=15, b=15), xaxis_title="Mean |SHAP value|", yaxis_title="Feature")
        st.plotly_chart(fig, use_container_width=True)
        top_feature = importance.idxmax()
        st.caption("Model explanation — not causal analysis. The strongest contributor to this 24h AQI forecast is the feature with the largest aggregated absolute SHAP value.")
        st.info(f"The highest-impact feature in the current 24h forecast is: {top_feature}.")
    except Exception as exc:
        st.info(f"SHAP unavailable for this model: {exc}")


def render_pipeline_flow():
    st.markdown("### End-to-End AI Data Flow")
    stages = [
        ("External Data APIs", "Weather + pollutant sources"),
        ("Feature Pipeline", "Cleaning + feature engineering"),
        ("Feature Store", "AQI features and historical context"),
        ("Model Registry", "Production models + metadata"),
        ("Prediction Engine", "24h / 48h / 72h inference"),
        ("AQI Intelligence", "Dashboard + alerts + explainability"),
    ]
    cols = st.columns(len(stages))
    for idx, (title, subtitle) in enumerate(stages):
        with cols[idx]:
            st.markdown(f"<div class='flow-card'><div class='flow-title'>{title}</div><div class='flow-subtitle'>{subtitle}</div></div>", unsafe_allow_html=True)
            if idx < len(stages) - 1:
                st.markdown("<div class='flow-arrow'>→</div>", unsafe_allow_html=True)


def render_pipeline_health(city: str, df: pd.DataFrame, models: dict):
    st.markdown("### Pipeline Health")
    ingress = "ONLINE" if not df.empty else "OFFLINE"
    feature_status = "READY" if not df.empty else "MISSING"
    registry_status = "ONLINE" if models else "OFFLINE"
    prediction_status = "READY" if models else "MISSING"
    metrics = [
        ("Feature ingestion", ingress),
        ("Feature computation", feature_status),
        ("Feature Store", ingress),
        ("Model Registry", registry_status),
        ("Prediction", prediction_status),
    ]
    cols = st.columns(len(metrics))
    for idx, (label, status) in enumerate(metrics):
        with cols[idx]:
            st.markdown(f"<div class='mini-card'><div class='mini-label'>{label}</div><div class='mini-value'>{status}</div></div>", unsafe_allow_html=True)


def render_overview(city: str, df: pd.DataFrame, models: dict):
    render_current_aqi(df)
    render_kpi_cards(df)
    if not df.empty:
        render_aqi_severity(safe_float(df.iloc[-1].get("us_aqi")))
    render_forecast_section(city, df, models)
    render_alerts(city, df, models)
    render_city_comparison()


def render_analytics(city: str, df: pd.DataFrame):
    render_historical_analysis(city, df)
    render_pollutant_analysis(df)
    render_weather_analysis(df)


def render_model_tab(city: str, df: pd.DataFrame, models: dict):
    render_model_performance(city, models)


def render_explainability(city: str, df: pd.DataFrame, models: dict):
    render_shap_analysis(city, df, models)


def render_pipeline_tab(city: str, df: pd.DataFrame, models: dict):
    render_pipeline_flow()
    render_pipeline_health(city, df, models)


def inject_css():
    st.html(
        """
        <style>
        :root {
          --bg: #07111d;
          --panel: rgba(17, 24, 39, 0.8);
          --line: rgba(148, 163, 184, 0.18);
          --text: #e5eefc;
          --muted: #9aa9bf;
          --primary: #7cc6fe;
          --green: #4ade80;
        }
        .stApp { background: linear-gradient(180deg, #07111d 0%, #0d1728 100%); color: var(--text); }
        .brand-block { padding: 0.5rem 0.25rem 1rem 0.25rem; }
        .brand { font-size: 2rem; font-weight: 800; letter-spacing: 0.08em; }
        .subtitle { font-size: 0.9rem; font-weight: 700; letter-spacing: 0.12em; color: var(--primary); }
        .subtle { color: var(--muted); font-size: 0.75rem; margin-top: 0.25rem; }
        .status-row { display: flex; justify-content: space-between; align-items: center; padding: 0.35rem 0.2rem; border-bottom: 1px solid var(--line); }
        .status { font-weight: 700; }
        .status.online { color: var(--green); }
        .status-on { color: var(--green); }
        .section-header { font-size: 2.2rem; font-weight: 700; margin-top: 0.5rem; margin-bottom: 0.15rem; }
        .meta-card, .stat-card, .forecast-card, .mini-card, .flow-card, .alert-card, .rank-card {
          background: rgba(15, 23, 42, 0.8);
          border: 1px solid var(--line);
          border-radius: 18px;
          padding: 1rem 1rem;
          box-shadow: 0 10px 30px rgba(15, 23, 42, 0.35);
        }
        .meta-label, .stat-label, .mini-label, .forecast-label, .alert-heading { color: var(--muted); letter-spacing: 0.02em; font-size: 0.72rem; text-transform: uppercase; }
        .meta-value, .stat-value, .mini-value, .forecast-aqi, .alert-value { font-size: 1.5rem; font-weight: 700; margin-top: 0.35rem; }
        .hero-card { background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 20px; padding: 1.3rem 1.4rem; box-shadow: 0 12px 35px rgba(15, 23, 42, 0.45); margin-bottom: 1rem; }
        .eyebrow { font-size: 0.7rem; letter-spacing: 0.15em; color: var(--muted); text-transform: uppercase; }
        .hero-score { font-size: 4rem; font-weight: 800; line-height: 1; margin: 0.35rem 0; }
        .hero-category { font-size: 1.2rem; font-weight: 700; }
        .hero-meta { color: var(--muted); margin-top: 0.3rem; }
        .hero-change { color: #d5e4ff; margin-top: 0.6rem; }
        .stat-trend { color: var(--muted); font-size: 0.7rem; margin-top: 0.35rem; }
        .severity-wrap { margin-top: 0.75rem; }
        .severity-scale { position: relative; height: 22px; border-radius: 999px; overflow: hidden; border: 1px solid rgba(148,163,184,.3); background: #111827; }
        .band { position: absolute; top: 0; bottom: 0; }
        .marker { position: absolute; top: -6px; width: 14px; height: 34px; border-radius: 50%; background: white; box-shadow: 0 0 0 2px rgba(255,255,255,0.3); transform: translateX(-50%); }
        .severity-labels { display: grid; grid-template-columns: repeat(6, minmax(0,1fr)); gap: 0.25rem; font-size: 0.62rem; color: var(--muted); margin-top: 0.5rem; }
        .forecast-card { text-align: center; }
        .forecast-aqi { font-size: 2.1rem; }
        .forecast-status { color: var(--muted); margin-top: 0.3rem; }
        .flow-card { min-height: 120px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; }
        .flow-title { font-weight: 700; margin-bottom: 0.4rem; }
        .flow-subtitle { color: var(--muted); font-size: 0.8rem; }
        .flow-arrow { text-align: center; font-size: 2rem; color: var(--primary); margin-top: -0.25rem; margin-bottom: 0.25rem; }
        .alert-card { margin-top: 0.75rem; }
        .alert-value { font-size: 1.7rem; }
        .alert-msg { color: var(--muted); margin-top: 0.3rem; }
        .rank-card { display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem; }
        .rank-position { color: var(--primary); font-size: 0.7rem; text-transform: uppercase; }
        .rank-city { font-weight: 700; }
        .rank-aqi { font-weight: 700; }
        div[data-testid="stSidebar"] { background: rgba(9, 16, 26, 0.9); }
        </style>
        """
    )


def main():
    inject_css()
    city, nav = render_sidebar()
    df = load_city_features(city)
    models = load_models(city)

    if df.empty:
        st.warning(f"No feature data found for **{city}** yet. Run the backfill pipeline and then train the registry before opening the dashboard.")
        return

    render_header(city, df, models)

    if nav == "Overview":
        render_overview(city, df, models)
    elif nav == "AQI Forecast":
        render_forecast_section(city, df, models)
    elif nav == "Historical Analytics":
        render_analytics(city, df)
    elif nav == "Pollutants":
        render_pollutant_analysis(df)
    elif nav == "Model Intelligence":
        render_model_tab(city, df, models)
    elif nav == "Explainability":
        render_explainability(city, df, models)
    elif nav == "Pipeline":
        render_pipeline_tab(city, df, models)

    with st.expander("Developer diagnostics"):
        st.write({"city": city, "rows": len(df), "models": sorted(models.keys()), "latest_timestamp": df["timestamp"].iloc[-1] if not df.empty else None})


if __name__ == "__main__":
    main()
