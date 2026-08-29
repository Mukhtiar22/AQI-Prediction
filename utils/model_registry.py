"""
utils/model_registry.py
------------------------
A tiny local model registry: versioned model artifacts + JSON metadata
(metrics, features used, training date). Mirrors the Hopsworks Model
Registry API closely enough that swapping it out later is small.

Storage layout:
    models/registry/<model_name>/v<version>/model.joblib
    models/registry/<model_name>/v<version>/metadata.json
    models/registry/<model_name>/production.json   <- pointer to best version
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

DEFAULT_REGISTRY_DIR = Path(os.environ.get("AQI_MODEL_REGISTRY_DIR", "models/registry"))


class ModelRegistry:
    def __init__(self, registry_dir: str | Path = DEFAULT_REGISTRY_DIR):
        self.registry_dir = Path(registry_dir)

    def _model_dir(self, name: str) -> Path:
        return self.registry_dir / name

    def _next_version(self, name: str) -> int:
        model_dir = self._model_dir(name)
        if not model_dir.exists():
            return 1
        versions = [
            int(p.name.replace("v", ""))
            for p in model_dir.glob("v*")
            if p.name.replace("v", "").isdigit()
        ]
        return max(versions, default=0) + 1

    def save_model(
        self,
        name: str,
        model: Any,
        metrics: dict,
        feature_names: list[str],
        target_name: str,
        extra: dict | None = None,
        set_as_production: bool = True,
    ) -> Path:
        version = self._next_version(name)
        version_dir = self._model_dir(name) / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, version_dir / "model.joblib")

        metadata = {
            "name": name,
            "version": version,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "feature_names": feature_names,
            "target_name": target_name,
            "extra": extra or {},
        }
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[model_registry] Saved '{name}' v{version} "
              f"(RMSE={metrics.get('rmse'):.3f}) -> {version_dir}")

        if set_as_production:
            self._set_production(name, version)

        return version_dir

    def _set_production(self, name: str, version: int) -> None:
        pointer_path = self._model_dir(name) / "production.json"
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pointer_path, "w") as f:
            json.dump({"name": name, "production_version": version}, f, indent=2)
        print(f"[model_registry] '{name}' production pointer -> v{version}")

    def load_production_model(self, name: str) -> tuple[Any, dict]:
        pointer_path = self._model_dir(name) / "production.json"
        if not pointer_path.exists():
            raise FileNotFoundError(
                f"No production model registered for '{name}'. Run training_pipeline.py first."
            )
        with open(pointer_path) as f:
            pointer = json.load(f)
        return self.load_model(name, pointer["production_version"])

    def load_model(self, name: str, version: int) -> tuple[Any, dict]:
        version_dir = self._model_dir(name) / f"v{version}"
        model = joblib.load(version_dir / "model.joblib")
        with open(version_dir / "metadata.json") as f:
            metadata = json.load(f)
        return model, metadata

    def list_versions(self, name: str) -> list[dict]:
        model_dir = self._model_dir(name)
        if not model_dir.exists():
            return []
        out = []
        for v_dir in sorted(model_dir.glob("v*")):
            meta_path = v_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    out.append(json.load(f))
        return out
