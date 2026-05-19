"""General AI-generated image detector for Layer 3.

This module is designed for GenImage-style real/AI image data. It is separate
from the face-forgery detector so CAVE can distinguish two image routes:

- general AIGC images: GenImage classifier
- face forgery/deepfake images: RedFace-style detector suite

If the local model is missing, inference returns None and Layer 3 falls back to
the existing diffusion + face-forgery ensemble.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from layers.fingerprint import FINGERPRINT_FEATURE_ORDER
from layers.fingerprint import _extract_features
from layers.fingerprint import _load_rgb
from layers.fingerprint import _score_features


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "general_aigc_classifier.joblib"
MODEL_META_PATH = Path(__file__).resolve().parents[1] / "models" / "general_aigc_classifier.meta.json"

GENERAL_AIGC_FEATURE_ORDER = FINGERPRINT_FEATURE_ORDER + [
    "fingerprint_heuristic",
]

_MODEL_CACHE: Optional[dict] = None
_MODEL_MTIME: Optional[float] = None


@dataclass
class GeneralAIGCPrediction:
    probability: float
    confidence: float
    source: str
    test_auc: Optional[float]
    features: dict[str, float]


def predict_general_aigc(path: str | Path) -> Optional[GeneralAIGCPrediction]:
    """Return general AI-generated image probability, or None if unavailable."""
    bundle = _load_model()
    if not bundle:
        return None

    model = bundle.get("binary_model")
    if model is None or not hasattr(model, "predict_proba"):
        return None

    image_path = Path(path)
    rgb = _load_rgb(image_path)
    features = extract_general_aigc_features(rgb)
    feature_order = bundle.get("feature_order", GENERAL_AIGC_FEATURE_ORDER)
    vector = np.asarray([[float(features.get(name, 0.0)) for name in feature_order]], dtype=np.float64)

    probabilities = model.predict_proba(vector)[0]
    classes = list(getattr(model, "classes_", [0, 1]))
    ai_idx = classes.index(1) if 1 in classes else int(np.argmax(classes))
    probability = _clamp01(float(probabilities[ai_idx]))
    confidence = _clamp01(abs(probability - 0.5) * 2.0)
    meta = bundle.get("meta", {})
    return GeneralAIGCPrediction(
        probability=probability,
        confidence=confidence,
        source=str(meta.get("data_source", "unknown")),
        test_auc=_as_optional_float(meta.get("test_auc")),
        features=features,
    )


def extract_general_aigc_features(rgb: np.ndarray) -> dict[str, float]:
    features = _extract_features(rgb)
    features["fingerprint_heuristic"] = _score_features(features)
    return features


def feature_vector(features: dict[str, float], feature_order: list[str] | None = None) -> list[float]:
    order = feature_order or GENERAL_AIGC_FEATURE_ORDER
    return [float(features.get(name, 0.0)) for name in order]


def _load_model() -> Optional[dict]:
    global _MODEL_CACHE, _MODEL_MTIME
    if not MODEL_PATH.exists():
        _MODEL_CACHE = None
        _MODEL_MTIME = None
        return None

    mtime = MODEL_PATH.stat().st_mtime
    if _MODEL_CACHE is not None and _MODEL_MTIME == mtime:
        return _MODEL_CACHE

    try:
        import joblib  # type: ignore

        bundle = joblib.load(MODEL_PATH)
        if not isinstance(bundle, dict):
            return None
        _MODEL_CACHE = bundle
        _MODEL_MTIME = mtime
    except Exception:
        _MODEL_CACHE = None
        _MODEL_MTIME = mtime
        return None
    return _MODEL_CACHE


def _as_optional_float(value) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))
