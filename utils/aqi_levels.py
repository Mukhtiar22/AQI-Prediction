"""
utils/aqi_levels.py
--------------------
US EPA AQI category breakpoints + alerting helper, shared by the
dashboard and any batch alerting job.
"""

from __future__ import annotations

AQI_CATEGORIES = [
    (0, 50, "Good", "#00e400", "Air quality is satisfactory."),
    (51, 100, "Moderate", "#ffff00", "Acceptable; some pollutants may be a concern for a very small group."),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00",
     "Sensitive groups (children, elderly, respiratory/heart conditions) may experience effects."),
    (151, 200, "Unhealthy", "#ff0000", "Everyone may begin to experience health effects."),
    (201, 300, "Very Unhealthy", "#8f3f97", "Health alert: everyone may experience more serious effects."),
    (301, 500, "Hazardous", "#7e0023", "Health warning of emergency conditions; entire population affected."),
]

HAZARD_THRESHOLD = 151  # "Unhealthy" and above triggers an alert


def classify_aqi(value: float) -> dict:
    for low, high, label, color, advice in AQI_CATEGORIES:
        if low <= value <= high:
            return {"category": label, "color": color, "advice": advice, "range": (low, high)}
    if value > 500:
        return {"category": "Hazardous", "color": "#7e0023",
                "advice": "Health warning of emergency conditions.", "range": (301, 500)}
    return {"category": "Unknown", "color": "#999999", "advice": "", "range": (None, None)}


def is_hazardous(value: float) -> bool:
    return value >= HAZARD_THRESHOLD
