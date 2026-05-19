"""Train Layer 7 learned redistribution-risk classifier.

The training data is synthetic but aligned to the CAVE DamageInputs surface:
variant presence, closed-platform distribution, deletion/reappearance, and
spread context. The runtime blends this learned score with the transparent
rule score, so it remains explainable for the demo/report.

Usage:
  python scripts/train_redistribution_risk.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layers.damage_score import DamageInputs  # noqa: E402
from layers.redistribution_risk import (  # noqa: E402
    FEATURE_ORDER,
    MODEL_META_PATH,
    MODEL_PATH,
    extract_redistribution_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="재유포 위험 learned classifier 학습")
    parser.add_argument("--samples-per-class", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--estimators", type=int, default=400)
    args = parser.parse_args()

    rows, labels = _build_dataset(args.samples_per_class, args.seed)
    x = np.asarray([[row[key] for key in FEATURE_ORDER] for row in rows], dtype=np.float32)
    y = np.asarray(labels, dtype=int)

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(y))
    train_end = int(len(y) * 0.70)
    val_end = int(len(y) * 0.85)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=args.estimators,
        max_depth=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(x[train_idx], y[train_idx])

    train_metrics = _metrics(y[train_idx], model.predict_proba(x[train_idx])[:, 1])
    val_metrics = _metrics(y[val_idx], model.predict_proba(x[val_idx])[:, 1])
    test_metrics = _metrics(y[test_idx], model.predict_proba(x[test_idx])[:, 1])

    metadata = {
        "task": "redistribution_risk_prediction",
        "data_source": "synthetic_damage_inputs",
        "model_class": "RandomForestClassifier",
        "feature_order": FEATURE_ORDER,
        "samples_per_class": args.samples_per_class,
        "estimators": args.estimators,
        "seed": args.seed,
        "train_auc": train_metrics["auc"],
        "train_accuracy": train_metrics["accuracy"],
        "val_auc": val_metrics["auc"],
        "val_accuracy": val_metrics["accuracy"],
        "test_auc": test_metrics["auc"],
        "test_accuracy": test_metrics["accuracy"],
        "best_threshold": test_metrics["best_threshold"],
        "best_accuracy": test_metrics["best_accuracy"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "RandomForest classifier over redistribution flags, spread context, and approximate graph statistics.",
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    import joblib  # type: ignore

    joblib.dump({"model": model, "feature_order": FEATURE_ORDER, "metadata": metadata}, MODEL_PATH)
    MODEL_META_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"redistribution classifier saved: {MODEL_PATH}")
    print(f"metadata saved: {MODEL_META_PATH}")
    print(
        f"test auc={test_metrics['auc']:.3f}, acc@0.5={test_metrics['accuracy']:.3f}, "
        f"best_threshold={test_metrics['best_threshold']:.3f}, best_acc={test_metrics['best_accuracy']:.3f}"
    )

    for label, inp in (
        ("low", DamageInputs(num_posts=3, num_shares=20, num_views=500)),
        ("variants", DamageInputs(num_posts=20, num_shares=300, num_views=8000, has_variants=True)),
        ("closed+variants", DamageInputs(num_posts=80, num_shares=1200, num_views=45000, has_variants=True, on_closed_platforms=True)),
        ("reappeared", DamageInputs(num_posts=80, num_shares=1200, num_views=45000, has_variants=True, on_closed_platforms=True, reappeared_after_deletion=True)),
    ):
        feats = extract_redistribution_features(inp)
        prob = float(model.predict_proba([[feats[key] for key in FEATURE_ORDER]])[0][1])
        print(f"{label:16s} prob={prob:.3f} score={prob * 5.0:.3f}")


def _build_dataset(samples_per_class: int, seed: int) -> tuple[list[dict[str, float]], list[int]]:
    rng = random.Random(seed)
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    for label in (0, 1):
        for _ in range(samples_per_class):
            inp = _sample_inputs(label, rng)
            rows.append(extract_redistribution_features(inp))
            labels.append(label)
    return rows, labels


def _sample_inputs(label: int, rng: random.Random) -> DamageInputs:
    if label == 1:
        has_variants = rng.random() < 0.78
        on_closed = rng.random() < 0.68
        reappeared = rng.random() < 0.48
        posts = rng.randint(30, 2000)
        platforms = rng.randint(2, 8)
        shares = int(10 ** rng.uniform(2.4, 5.8))
        views = int(10 ** rng.uniform(4.0, 6.8))
        speed = rng.uniform(1, 36)
    else:
        has_variants = rng.random() < 0.12
        on_closed = rng.random() < 0.08
        reappeared = rng.random() < 0.04
        posts = rng.randint(0, 60)
        platforms = rng.randint(0, 3)
        shares = int(10 ** rng.uniform(0.0, 3.0))
        views = int(10 ** rng.uniform(1.0, 4.5))
        speed = rng.uniform(24, 168)

    return DamageInputs(
        num_posts=posts,
        num_platforms=platforms,
        num_shares=shares,
        num_views=views,
        spread_speed_hours=speed,
        has_variants=has_variants,
        on_closed_platforms=on_closed,
        reappeared_after_deletion=reappeared,
    )


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, roc_auc_score

    thresholds = sorted(set(float(score) for score in scores))
    candidates = [0.5]
    candidates.extend(thresholds)
    candidates.extend((a + b) / 2 for a, b in zip(thresholds[:-1], thresholds[1:]))

    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in candidates:
        accuracy = float(accuracy_score(labels, scores >= threshold))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    return {
        "auc": round(float(roc_auc_score(labels, scores)), 6),
        "accuracy": round(float(accuracy_score(labels, scores >= 0.5)), 6),
        "best_threshold": round(best_threshold, 6),
        "best_accuracy": round(best_accuracy, 6),
    }


if __name__ == "__main__":
    main()
