"""
Trained model store loaded once at application startup.

Public API
----------
ModelStore
    Holds the scaler, XGBoost model, feature order, and version string.
    .predict(features) validates the feature dict and returns (subtype, probabilities).

load_latest_model(models_dir)
    Scans models_dir for version subdirectories (v*), loads the most recent one.
    Returns None if no model is found.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

log = logging.getLogger(__name__)

# Inverse of the training label map {LUAD: 0, LUSC: 1}
_IDX_TO_LABEL = {0: "LUAD", 1: "LUSC"}


@dataclass
class ModelStore:
    version: str
    feature_names: list[str]
    scaler: object  # sklearn StandardScaler
    xgb: object  # XGBClassifier

    def predict(self, features: dict[str, float]) -> tuple[str, dict[str, float]]:
        """Return (predicted_subtype, {LUAD: prob, LUSC: prob}).

        Caller must guarantee all feature_names are present in features.
        """
        vec = np.array([features[g] for g in self.feature_names], dtype=np.float64).reshape(1, -1)
        vec_scaled = self.scaler.transform(vec)
        proba = self.xgb.predict_proba(vec_scaled)[0]
        predicted = _IDX_TO_LABEL[int(np.argmax(proba))]
        return predicted, {"LUAD": float(proba[0]), "LUSC": float(proba[1])}


def load_latest_model(models_dir: Path) -> ModelStore | None:
    """Load the most recent model version from models_dir.

    Returns None if the directory does not exist or contains no versioned models.
    """
    if not models_dir.is_dir():
        log.warning("models_dir %s does not exist; model store will be unavailable", models_dir)
        return None

    versions = sorted(
        (d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("v")),
        key=lambda d: d.name,
        reverse=True,
    )
    if not versions:
        log.warning("No model versions found in %s", models_dir)
        return None

    model_dir = versions[0]
    try:
        feature_names: list[str] = json.loads((model_dir / "feature_names.json").read_text())
        scaler = joblib.load(model_dir / "scaler.joblib")
        xgb = joblib.load(model_dir / "xgb.joblib")
    except Exception as exc:
        log.error("Failed to load model from %s: %s", model_dir, exc)
        return None

    log.info("Loaded model %s (%d features)", model_dir.name, len(feature_names))
    return ModelStore(
        version=model_dir.name,
        feature_names=feature_names,
        scaler=scaler,
        xgb=xgb,
    )
