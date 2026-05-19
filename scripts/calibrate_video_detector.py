"""로컬 real/fake 쌍으로 영상 detector calibration 생성.

실행:
  python scripts/calibrate_video_detector.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layers.deepfake_video_detectors import CALIBRATION_PATH  # noqa: E402
from layers.deepfake_video_detectors import predict_video_detector_suite  # noqa: E402
from layers.video_utils import sample_video_face_crops  # noqa: E402


FEATURE_ORDER = [
    "ensemble_raw",
    "faceforge_xception",
    "deepfakebench_resnet18",
    "deepfakebench_r3d18",
    "deepfakebench_efficientnet_b4",
    "legacy_efficientnet_b0",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="CAVE 영상 detector 로컬 calibration")
    parser.add_argument("--real-dir", default="test_data/real")
    parser.add_argument("--fake-dir", default="test_data/deepfake")
    parser.add_argument("--output", default=str(CALIBRATION_PATH))
    parser.add_argument("--unpaired", action="store_true", help="파일명 pair가 아닌 real/fake 폴더 전체를 사용")
    parser.add_argument("--max-per-label", type=int, default=80)
    parser.add_argument("--fake-per-method", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = _collect_rows(
        Path(args.real_dir),
        Path(args.fake_dir),
        args.unpaired,
        args.max_per_label,
        args.fake_per_method,
        args.seed,
    )
    if len(rows) < 4:
        raise SystemExit("calibration에 사용할 real/fake 샘플이 부족합니다.")

    features = np.asarray([[row["features"].get(name, 0.5) for name in FEATURE_ORDER] for row in rows], dtype=float)
    labels = np.asarray([row["label"] for row in rows], dtype=int)

    calibration = _fit_logistic(features, labels)
    calibration.update({
        "feature_order": FEATURE_ORDER,
        "samples": len(rows),
        "real_dir": args.real_dir,
        "fake_dir": args.fake_dir,
        "fake_per_method": args.fake_per_method,
        "note": "Small local calibration for demo data; do not treat as forensic validation.",
    })

    predictions = _predict_with_calibration(features, calibration)
    metrics = _classification_metrics(labels, predictions)
    calibration["training_metrics"] = metrics

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"calibration saved: {output}")
    print(
        "metrics "
        f"auc={metrics['auc']:.3f}, acc@0.5={metrics['accuracy']:.3f}, "
        f"best_threshold={metrics['best_threshold']:.3f}, best_acc={metrics['best_accuracy']:.3f}"
    )
    for row, score in zip(rows, predictions):
        print(f"{row['label_name']},{row['path']},{score:.3f},{row['features']}")


def _collect_rows(
    real_dir: Path,
    fake_dir: Path,
    unpaired: bool,
    max_per_label: int,
    fake_per_method: int,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    if unpaired:
        rng = random.Random(seed)
        real_files = _sample_files(real_dir, max_per_label, seed)
        fake_files = _sample_fake_files(fake_dir, max_per_label, fake_per_method, seed + 1)
        for label, label_name, files in ((0, "real", real_files), (1, "fake", fake_files)):
            for path in files:
                rows.append(_analyze_path(path, label=label, label_name=label_name))
        rng.shuffle(rows)
        return rows

    for real_path in sorted(real_dir.glob("*.mp4")):
        fake_path = fake_dir / f"{real_path.stem}_fake{real_path.suffix}"
        if not fake_path.exists():
            continue

        rows.append(_analyze_path(real_path, label=0, label_name="real"))
        rows.append(_analyze_path(fake_path, label=1, label_name="fake"))
    return rows


def _sample_files(root: Path, max_count: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    files = sorted(root.rglob("*.mp4"))
    rng.shuffle(files)
    if max_count > 0:
        return files[:max_count]
    return files


def _sample_fake_files(root: Path, max_count: int, fake_per_method: int, seed: int) -> list[Path]:
    if fake_per_method <= 0:
        return _sample_files(root, max_count, seed)

    rng = random.Random(seed)
    groups: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*.mp4")):
        groups.setdefault(_method_from_file(path), []).append(path)

    selected: list[Path] = []
    for method in sorted(groups):
        files = groups[method]
        rng.shuffle(files)
        selected.extend(files[:fake_per_method])
    rng.shuffle(selected)
    return selected


def _method_from_file(path: Path) -> str:
    name = path.name
    if "_" not in name:
        return "unknown"
    return name.split("_", 1)[0]


def _analyze_path(path: Path, label: int, label_name: str) -> dict:
    samples = sample_video_face_crops(path, num_frames=16)
    face_ratio = sum(1 for sample in samples if sample.detected) / max(len(samples), 1)
    suite = predict_video_detector_suite([sample.image for sample in samples], face_ratio=face_ratio)
    return {
        "path": str(path),
        "label": label,
        "label_name": label_name,
        "features": suite.feature_map,
    }


def _fit_logistic(features: np.ndarray, labels: np.ndarray) -> dict:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return _fit_threshold_shift(features[:, 0], labels)

    model = LogisticRegression(
        C=0.8,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    model.fit(features, labels)
    return {
        "method": "logistic",
        "intercept": float(model.intercept_[0]),
        "coefficients": [float(value) for value in model.coef_[0]],
    }


def _fit_threshold_shift(scores: np.ndarray, labels: np.ndarray) -> dict:
    metrics = _classification_metrics(labels, scores)
    return {
        "method": "threshold_shift",
        "decision_threshold": float(metrics["best_threshold"]),
        "temperature": 0.65,
    }


def _predict_with_calibration(features: np.ndarray, calibration: dict) -> np.ndarray:
    if calibration["method"] == "logistic":
        coefficients = np.asarray(calibration["coefficients"], dtype=float)
        z = float(calibration["intercept"]) + features.dot(coefficients)
        return 1.0 / (1.0 + np.exp(-z))
    threshold = float(calibration["decision_threshold"])
    return np.asarray([1.0 if score >= threshold else 0.0 for score in features[:, 0]], dtype=float)


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
        "best_threshold": best_threshold,
        "best_accuracy": best_accuracy,
    }


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


if __name__ == "__main__":
    main()
