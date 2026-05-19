"""GenImage 기반 Layer 3 general AIGC classifier 학습."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layers.general_aigc_detector import GENERAL_AIGC_FEATURE_ORDER  # noqa: E402
from layers.general_aigc_detector import MODEL_PATH  # noqa: E402
from layers.general_aigc_detector import extract_general_aigc_features  # noqa: E402
from layers.general_aigc_detector import feature_vector  # noqa: E402
from layers.fingerprint import _load_rgb  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="GenImage general AIGC classifier 학습")
    parser.add_argument("--train-real-dir", default="test_data/genimage/calibration/real")
    parser.add_argument("--train-ai-dir", default="test_data/genimage/calibration/ai")
    parser.add_argument("--eval-real-dir", default="test_data/genimage/eval/real")
    parser.add_argument("--eval-ai-dir", default="test_data/genimage/eval/ai")
    parser.add_argument("--output", default=str(MODEL_PATH))
    parser.add_argument("--max-per-label", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_rows = _collect_rows(Path(args.train_real_dir), Path(args.train_ai_dir), args.max_per_label, args.seed)
    eval_rows = _collect_rows(Path(args.eval_real_dir), Path(args.eval_ai_dir), args.max_per_label, args.seed + 1000)
    if len(train_rows) < 10 or len(eval_rows) < 4:
        raise SystemExit("GenImage 학습/평가 샘플이 부족합니다. prepare_genimage_dataset.py 결과를 확인하세요.")

    x_train = np.asarray([row["vector"] for row in train_rows], dtype=np.float64)
    y_train = np.asarray([row["label"] for row in train_rows], dtype=int)
    x_eval = np.asarray([row["vector"] for row in eval_rows], dtype=np.float64)
    y_eval = np.asarray([row["label"] for row in eval_rows], dtype=int)

    model = _train_model(x_train, y_train, args.seed)
    eval_prob = model.predict_proba(x_eval)[:, _ai_index(model)]
    metrics = _classification_metrics(y_eval, eval_prob)
    generator_metrics = _generator_metrics(eval_rows, eval_prob)

    meta = {
        "task": "general_aigc_real_ai_detection",
        "data_source": "GenImage",
        "feature_order": GENERAL_AIGC_FEATURE_ORDER,
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "max_per_label": args.max_per_label,
        "seed": args.seed,
        "test_auc": metrics["auc"],
        "test_accuracy": metrics["accuracy"],
        "test_balanced_accuracy": metrics["balanced_accuracy"],
        "best_threshold": metrics["best_threshold"],
        "best_accuracy": metrics["best_accuracy"],
        "generator_metrics": generator_metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "RandomForest classifier over FFT/residual/noise features from GenImage.",
    }
    bundle = {
        "binary_model": model,
        "feature_order": GENERAL_AIGC_FEATURE_ORDER,
        "meta": meta,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    import joblib  # type: ignore

    joblib.dump(bundle, output)
    output.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"classifier saved: {output}")
    print(f"metadata saved: {output.with_suffix('.meta.json')}")
    print(
        "eval "
        f"auc={metrics['auc']:.3f}, acc@0.5={metrics['accuracy']:.3f}, "
        f"bal_acc@0.5={metrics['balanced_accuracy']:.3f}, "
        f"best_threshold={metrics['best_threshold']:.3f}, best_acc={metrics['best_accuracy']:.3f}"
    )
    print("generator,count,prob_mean,auc_vs_real")
    for generator, row in generator_metrics.items():
        print(f"{generator},{row['count']},{row['prob_mean']:.3f},{row['auc_vs_real']:.3f}")


def _train_model(x_train: np.ndarray, y_train: np.ndarray, seed: int):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=16,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def _collect_rows(real_dir: Path, ai_dir: Path, max_per_label: int, seed: int) -> list[dict]:
    rows: list[dict] = []
    manifest = _read_manifest(real_dir.parents[1] / "manifest.csv") if len(real_dir.parents) >= 2 else {}
    real_files = _sample_files(real_dir, max_per_label, seed)
    ai_files = _sample_files(ai_dir, max_per_label, seed + 1)
    for path in real_files:
        rows.append(_analyze_path(path, label=0, generator=_generator_from_path(path, manifest)))
    for path in ai_files:
        rows.append(_analyze_path(path, label=1, generator=_generator_from_path(path, manifest)))
    random.Random(seed + 2).shuffle(rows)
    return rows


def _analyze_path(path: Path, label: int, generator: str) -> dict:
    rgb = _load_rgb(path)
    features = extract_general_aigc_features(rgb)
    return {
        "path": str(path),
        "label": label,
        "generator": generator,
        "features": features,
        "vector": feature_vector(features),
    }


def _read_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            target = row.get("target_path", "")
            generator = row.get("generator", "")
            if target and generator:
                result[str(Path(target).resolve())] = generator
    return result


def _generator_from_path(path: Path, manifest: dict[str, str]) -> str:
    resolved = str(path.resolve())
    if resolved in manifest:
        return manifest[resolved]
    name = path.name
    if "__" in name:
        return name.split("__", 1)[0]
    return "unknown"


def _sample_files(root: Path, max_count: int, seed: int) -> list[Path]:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    rng = random.Random(seed)
    rng.shuffle(files)
    if max_count > 0:
        return files[:max_count]
    return files


def _ai_index(model) -> int:
    classes = list(getattr(model, "classes_", [0, 1]))
    return classes.index(1) if 1 in classes else int(np.argmax(classes))


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

    preds = (scores >= 0.5).astype(int)
    return {
        "auc": _auc(labels, scores),
        "accuracy": float(np.mean(preds == labels)),
        "balanced_accuracy": _balanced_accuracy(labels, preds),
        "best_threshold": best_threshold,
        "best_accuracy": best_accuracy,
    }


def _generator_metrics(rows: list[dict], scores: np.ndarray) -> dict[str, dict]:
    real_scores = [float(score) for row, score in zip(rows, scores) if row["label"] == 0]
    result: dict[str, dict] = {}
    for generator in sorted({row["generator"] for row in rows if row["label"] == 1}):
        generator_scores = [float(score) for row, score in zip(rows, scores) if row["label"] == 1 and row["generator"] == generator]
        labels = np.asarray([0] * len(real_scores) + [1] * len(generator_scores), dtype=int)
        combined = np.asarray(real_scores + generator_scores, dtype=float)
        result[generator] = {
            "count": len(generator_scores),
            "prob_mean": float(np.mean(generator_scores)) if generator_scores else 0.0,
            "auc_vs_real": _auc(labels, combined) if len(generator_scores) and len(real_scores) else 0.5,
        }
    return result


def _balanced_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    values = []
    for label in (0, 1):
        mask = labels == label
        if np.any(mask):
            values.append(float(np.mean(preds[mask] == labels[mask])))
    return float(np.mean(values)) if values else 0.0


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
