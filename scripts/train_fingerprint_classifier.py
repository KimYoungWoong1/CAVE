"""레이어 5 fingerprint classifier 학습.

RedFace split의 real/fake 이미지에서 FFT/잔차/노이즈 feature를 추출해
학습 기반 fingerprint classifier를 만든다.

실행:
  python scripts/train_fingerprint_classifier.py
  python scripts/train_fingerprint_classifier.py --max-per-label 600 --fake-per-method 150
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layers.fingerprint import CLASSIFIER_PATH  # noqa: E402
from layers.fingerprint import FINGERPRINT_FEATURE_ORDER  # noqa: E402
from layers.fingerprint import _extract_features  # noqa: E402
from layers.fingerprint import _load_rgb  # noqa: E402
from layers.fingerprint import feature_vector  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="CAVE fingerprint classifier 학습")
    parser.add_argument("--train-real-dir", default="test_data/redface/calibration/real")
    parser.add_argument("--train-fake-dir", default="test_data/redface/calibration/fake")
    parser.add_argument("--eval-real-dir", default="test_data/redface/eval/real")
    parser.add_argument("--eval-fake-dir", default="test_data/redface/eval/fake")
    parser.add_argument("--output", default=str(CLASSIFIER_PATH))
    parser.add_argument("--max-per-label", type=int, default=600)
    parser.add_argument("--fake-per-method", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_rows = _collect_rows(
        real_dir=Path(args.train_real_dir),
        fake_dir=Path(args.train_fake_dir),
        max_per_label=args.max_per_label,
        fake_per_method=args.fake_per_method,
        seed=args.seed,
    )
    eval_rows = _collect_rows(
        real_dir=Path(args.eval_real_dir),
        fake_dir=Path(args.eval_fake_dir),
        max_per_label=args.max_per_label,
        fake_per_method=args.fake_per_method,
        seed=args.seed + 1000,
    )

    x_train = np.asarray([row["vector"] for row in train_rows], dtype=np.float64)
    y_train = np.asarray([row["label"] for row in train_rows], dtype=int)
    x_eval = np.asarray([row["vector"] for row in eval_rows], dtype=np.float64)
    y_eval = np.asarray([row["label"] for row in eval_rows], dtype=int)

    binary_model = _train_binary_model(x_train, y_train, args.seed)
    fake_train = [row for row in train_rows if row["label"] == 1]
    method_model = _train_method_model(fake_train, args.seed)

    eval_prob = binary_model.predict_proba(x_eval)[:, _fake_index(binary_model)]
    metrics = _classification_metrics(y_eval, eval_prob)
    method_metrics = _method_metrics(eval_rows, eval_prob)
    method_accuracy = _method_accuracy(method_model, eval_rows)

    meta = {
        "task": "fingerprint_real_fake_and_method_attribution",
        "data_source": "RedFace",
        "feature_order": FINGERPRINT_FEATURE_ORDER,
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "fake_per_method": args.fake_per_method,
        "max_per_label": args.max_per_label,
        "seed": args.seed,
        "test_auc": metrics["auc"],
        "test_accuracy": metrics["accuracy"],
        "test_balanced_accuracy": metrics["balanced_accuracy"],
        "best_threshold": metrics["best_threshold"],
        "best_accuracy": metrics["best_accuracy"],
        "method_accuracy": method_accuracy,
        "method_metrics": method_metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "RandomForest classifier over FFT/residual/noise fingerprint features.",
    }

    bundle = {
        "binary_model": binary_model,
        "method_model": method_model,
        "feature_order": FINGERPRINT_FEATURE_ORDER,
        "meta": meta,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    import joblib  # type: ignore

    joblib.dump(bundle, output)
    output.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"classifier saved: {output}")
    print(f"metadata saved: {output.with_suffix('.meta.json')}")
    print(
        "eval "
        f"auc={metrics['auc']:.3f}, acc@0.5={metrics['accuracy']:.3f}, "
        f"bal_acc@0.5={metrics['balanced_accuracy']:.3f}, "
        f"best_threshold={metrics['best_threshold']:.3f}, best_acc={metrics['best_accuracy']:.3f}, "
        f"method_acc={method_accuracy:.3f}"
    )
    print("fake_method,count,prob_mean,auc_vs_real")
    for method, row in method_metrics.items():
        print(f"{method},{row['count']},{row['prob_mean']:.3f},{row['auc_vs_real']:.3f}")


def _train_binary_model(x_train: np.ndarray, y_train: np.ndarray, seed: int):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=450,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def _train_method_model(fake_rows: list[dict], seed: int):
    from sklearn.ensemble import RandomForestClassifier

    x_train = np.asarray([row["vector"] for row in fake_rows], dtype=np.float64)
    y_train = np.asarray([row["method"] for row in fake_rows], dtype=object)
    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def _collect_rows(
    real_dir: Path,
    fake_dir: Path,
    max_per_label: int,
    fake_per_method: int,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    real_files = _sample_files(real_dir, max_per_label, seed)
    fake_files = _sample_fake_files(fake_dir, max_per_label, fake_per_method, seed + 1)

    for path in real_files:
        rows.append(_analyze_path(path, label=0, method="real"))
    for path in fake_files:
        rows.append(_analyze_path(path, label=1, method=_method_from_file(path)))

    random.Random(seed + 2).shuffle(rows)
    return rows


def _analyze_path(path: Path, label: int, method: str) -> dict:
    rgb = _load_rgb(path)
    features = _extract_features(rgb)
    return {
        "path": str(path),
        "label": label,
        "method": method,
        "features": features,
        "vector": feature_vector(features),
    }


def _sample_files(root: Path, max_count: int, seed: int) -> list[Path]:
    files = _image_files(root)
    rng = random.Random(seed)
    rng.shuffle(files)
    if max_count > 0:
        return files[:max_count]
    return files


def _sample_fake_files(root: Path, max_count: int, fake_per_method: int, seed: int) -> list[Path]:
    if fake_per_method <= 0:
        return _sample_files(root, max_count, seed)

    groups: dict[str, list[Path]] = {}
    for path in _image_files(root):
        groups.setdefault(_method_from_file(path), []).append(path)

    rng = random.Random(seed)
    selected: list[Path] = []
    for method in sorted(groups):
        files = groups[method]
        rng.shuffle(files)
        selected.extend(files[:fake_per_method])
    rng.shuffle(selected)
    return selected


def _image_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _method_from_file(path: Path) -> str:
    name = path.name
    if "_" not in name:
        return "unknown"
    return name.split("_", 1)[0]


def _classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    thresholds = sorted(set(float(score) for score in scores))
    candidates = [0.5]
    if thresholds:
        candidates.extend(thresholds)
        candidates.extend((a + b) / 2 for a, b in zip(thresholds[:-1], thresholds[1:]))

    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in candidates:
        preds = (scores >= threshold).astype(int)
        accuracy = float(np.mean(preds == labels))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    return {
        "auc": _auc(labels, scores),
        "accuracy": float(np.mean((scores >= 0.5).astype(int) == labels)),
        "balanced_accuracy": _balanced_accuracy(labels, (scores >= 0.5).astype(int)),
        "best_threshold": best_threshold,
        "best_accuracy": best_accuracy,
    }


def _method_metrics(rows: list[dict], scores: np.ndarray) -> dict[str, dict]:
    real_scores = [float(score) for row, score in zip(rows, scores) if row["label"] == 0]
    result: dict[str, dict] = {}
    methods = sorted({row["method"] for row in rows if row["label"] == 1})
    for method in methods:
        method_scores = [float(score) for row, score in zip(rows, scores) if row["method"] == method]
        labels = np.asarray([0] * len(real_scores) + [1] * len(method_scores), dtype=int)
        method_arr = np.asarray(real_scores + method_scores, dtype=float)
        result[method] = {
            "count": len(method_scores),
            "prob_mean": float(np.mean(method_scores)) if method_scores else 0.0,
            "auc_vs_real": _auc(labels, method_arr) if method_scores and real_scores else 0.5,
        }
    return result


def _method_accuracy(method_model, rows: list[dict]) -> float:
    fake_rows = [row for row in rows if row["label"] == 1]
    if not fake_rows:
        return 0.0
    x_eval = np.asarray([row["vector"] for row in fake_rows], dtype=np.float64)
    y_eval = np.asarray([row["method"] for row in fake_rows], dtype=object)
    preds = method_model.predict(x_eval)
    return float(np.mean(preds == y_eval))


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    total = 0.0
    for p in pos:
        total += float(np.sum(p > neg))
        total += 0.5 * float(np.sum(p == neg))
    return float(total / (len(pos) * len(neg)))


def _balanced_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    values = []
    for label in (0, 1):
        mask = labels == label
        if np.any(mask):
            values.append(float(np.mean(preds[mask] == labels[mask])))
    return float(np.mean(values)) if values else 0.0


def _fake_index(model) -> int:
    classes = list(getattr(model, "classes_", [0, 1]))
    return classes.index(1) if 1 in classes else int(np.argmax(classes))


if __name__ == "__main__":
    main()
