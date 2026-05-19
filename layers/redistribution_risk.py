"""Learned redistribution-risk model for Layer 7.

The model estimates the probability that a case has high re-upload or
redistribution risk. It is intentionally lightweight: a RandomForest over
case metadata and approximate propagation-graph statistics. If the model is
missing, damage_score.py falls back to the rule-based score.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from layers.damage_score import DamageInputs


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "redistribution_risk_classifier.joblib"
MODEL_META_PATH = Path(__file__).resolve().parents[1] / "models" / "redistribution_risk_classifier.meta.json"

FEATURE_ORDER = [
    "has_variants",
    "on_closed_platforms",
    "reappeared_after_deletion",
    "post_log",
    "platform_norm",
    "share_log",
    "view_log",
    "speed_fast",
    "virality_proxy",
    "closed_x_variants",
    "variant_x_reappeared",
    "graph_nodes_norm",
    "graph_edges_norm",
    "graph_density",
]

_model_cache: tuple[float, dict] | None = None


@dataclass
class RedistributionPrediction:
    probability: float
    converted_score: float
    model_type: str
    metadata: dict


def predict_redistribution_risk(inputs: "DamageInputs") -> Optional[RedistributionPrediction]:
    """Return learned redistribution risk probability, or None if unavailable."""
    bundle = _load_model()
    if bundle is None:
        return None

    model = bundle.get("model")
    metadata = bundle.get("metadata") or load_model_metadata() or {}
    feature_order = bundle.get("feature_order", FEATURE_ORDER)
    if model is None:
        return None

    features = extract_redistribution_features(inputs)
    vector = [[float(features.get(key, 0.0)) for key in feature_order]]
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(vector)[0][1])
    else:
        probability = float(model.predict(vector)[0])
    probability = max(0.0, min(1.0, probability))
    return RedistributionPrediction(
        probability=probability,
        converted_score=round(probability * 5.0, 3),
        model_type=str(metadata.get("model_class", type(model).__name__)),
        metadata=metadata,
    )


def extract_redistribution_features(inputs: "DamageInputs") -> dict[str, float]:
    """Feature dictionary shared by training and inference."""
    post_log = _log_norm(inputs.num_posts, 2000)
    share_log = _log_norm(inputs.num_shares, 500_000)
    view_log = _log_norm(inputs.num_views, 5_000_000)
    platform_norm = min(max(float(inputs.num_platforms), 0.0) / 8.0, 1.0)
    speed_fast = (
        max(0.0, 1.0 - float(inputs.spread_speed_hours) / 168.0)
        if inputs.spread_speed_hours > 0 else 0.5
    )
    virality_proxy = min(
        0.20 * post_log
        + 0.25 * share_log
        + 0.20 * view_log
        + 0.15 * platform_norm
        + 0.10 * speed_fast
        + 0.05 * float(inputs.has_variants)
        + 0.05 * float(inputs.on_closed_platforms),
        1.0,
    )

    graph_nodes_norm, graph_edges_norm, graph_density = _graph_stats(inputs)
    return {
        "has_variants": float(inputs.has_variants),
        "on_closed_platforms": float(inputs.on_closed_platforms),
        "reappeared_after_deletion": float(inputs.reappeared_after_deletion),
        "post_log": post_log,
        "platform_norm": platform_norm,
        "share_log": share_log,
        "view_log": view_log,
        "speed_fast": speed_fast,
        "virality_proxy": virality_proxy,
        "closed_x_variants": float(inputs.on_closed_platforms and inputs.has_variants),
        "variant_x_reappeared": float(inputs.has_variants and inputs.reappeared_after_deletion),
        "graph_nodes_norm": graph_nodes_norm,
        "graph_edges_norm": graph_edges_norm,
        "graph_density": graph_density,
    }


def load_model_metadata() -> Optional[dict]:
    if not MODEL_META_PATH.exists():
        return None
    try:
        return json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_model() -> Optional[dict]:
    global _model_cache
    if not MODEL_PATH.exists():
        return None

    mtime = MODEL_PATH.stat().st_mtime
    if _model_cache is not None and _model_cache[0] == mtime:
        return _model_cache[1]

    try:
        import joblib  # type: ignore

        bundle = joblib.load(MODEL_PATH)
    except Exception:
        return None

    if not isinstance(bundle, dict):
        bundle = {
            "model": bundle,
            "feature_order": FEATURE_ORDER,
            "metadata": load_model_metadata() or {},
        }
    _model_cache = (mtime, bundle)
    return bundle


def _graph_stats(inputs: "DamageInputs") -> tuple[float, float, float]:
    try:
        from layers.gnn_spread_model import build_approximate_graph

        graph = build_approximate_graph(inputs)
        nodes = float(graph.num_nodes or 0)
        edges = float(graph.edge_index.size(1)) if graph.edge_index is not None else 0.0
        density = edges / max(nodes * max(nodes - 1.0, 1.0), 1.0)
        return (
            min(nodes / 150.0, 1.0),
            min(edges / 400.0, 1.0),
            min(density * 20.0, 1.0),
        )
    except Exception:
        return 0.0, 0.0, 0.0


def _log_norm(value: float, saturation: float) -> float:
    if saturation <= 0 or value <= 0:
        return 0.0
    return min(math.log1p(float(value)) / math.log1p(saturation), 1.0)
