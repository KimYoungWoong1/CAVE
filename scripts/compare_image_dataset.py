"""real/fake 이미지 폴더의 CAVE 이미지 detector 성능 확인."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layers import ai_detection, fingerprint  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="CAVE 이미지 real/fake 비교")
    parser.add_argument("--real-dir", default="test_data/redface/eval/real")
    parser.add_argument("--fake-dir", default="test_data/redface/eval/fake")
    parser.add_argument("--max-per-label", type=int, default=80)
    parser.add_argument("--fake-per-method", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    real_files = _sample_files(Path(args.real_dir), args.max_per_label, args.seed)
    fake_files = _sample_fake_files(Path(args.fake_dir), args.max_per_label, args.fake_per_method, args.seed + 1)
    for label, label_name, files in ((0, "real", real_files), (1, "fake", fake_files)):
        for path in files:
            ai = ai_detection.run(str(path))
            fp = fingerprint.run(str(path))
            rows.append({
                "label": label,
                "label_name": label_name,
                "file": str(path),
                "ai_detection": ai.ai_score,
                "fingerprint": fp.ai_score,
                "verdict": ai.verdict,
            })

    _print_table(rows)
    _print_metrics(rows)


def _sample_files(root: Path, max_count: int, seed: int) -> list[Path]:
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    rng = random.Random(seed)
    rng.shuffle(files)
    if max_count > 0:
        return files[:max_count]
    return files


def _sample_fake_files(root: Path, max_count: int, fake_per_method: int, seed: int) -> list[Path]:
    if fake_per_method <= 0:
        return _sample_files(root, max_count, seed)

    rng = random.Random(seed)
    groups: dict[str, list[Path]] = {}
    for path in _sample_files(root, 0, seed):
        groups.setdefault(_method_from_file(str(path)), []).append(path)

    selected: list[Path] = []
    for method in sorted(groups):
        files = groups[method]
        rng.shuffle(files)
        selected.extend(files[:fake_per_method])
    rng.shuffle(selected)
    return selected


def _print_table(rows: list[dict]) -> None:
    print("label,file,ai_detection,fingerprint,verdict")
    for row in rows[:40]:
        print(
            f"{row['label_name']},{Path(row['file']).name},"
            f"{_fmt(row['ai_detection'])},{_fmt(row['fingerprint'])},{row['verdict']}"
        )
    if len(rows) > 40:
        print(f"... {len(rows) - 40} more rows")


def _print_metrics(rows: list[dict]) -> None:
    for key in ("ai_detection", "fingerprint"):
        labels = np.asarray([row["label"] for row in rows if row[key] is not None], dtype=int)
        scores = np.asarray([float(row[key]) for row in rows if row[key] is not None], dtype=float)
        if len(scores) == 0:
            continue
        metrics = _classification_metrics(labels, scores)
        print(
            f"{key}: auc={metrics['auc']:.3f}, "
            f"acc@0.5={metrics['accuracy']:.3f}, "
            f"best_threshold={metrics['best_threshold']:.3f}, "
            f"best_acc={metrics['best_accuracy']:.3f}"
        )
    _print_fake_method_metrics(rows)


def _print_fake_method_metrics(rows: list[dict]) -> None:
    real_rows = [row for row in rows if row["label"] == 0]
    fake_rows = [row for row in rows if row["label"] == 1]
    if not real_rows or not fake_rows:
        return

    methods = sorted({_method_from_file(row["file"]) for row in fake_rows})
    print()
    print("fake_method,count,ai_mean,ai_auc_vs_real,fp_mean,fp_auc_vs_real")
    for method in methods:
        method_rows = [row for row in fake_rows if _method_from_file(row["file"]) == method]
        values = [
            method,
            str(len(method_rows)),
            *_method_metric_columns(real_rows, method_rows, "ai_detection"),
            *_method_metric_columns(real_rows, method_rows, "fingerprint"),
        ]
        print(",".join(values))


def _method_metric_columns(real_rows: list[dict], fake_rows: list[dict], key: str) -> list[str]:
    fake_scores = [float(row[key]) for row in fake_rows if row.get(key) is not None]
    real_scores = [float(row[key]) for row in real_rows if row.get(key) is not None]
    if not fake_scores or not real_scores:
        return ["N/A", "N/A"]

    labels = np.asarray([0] * len(real_scores) + [1] * len(fake_scores), dtype=int)
    scores = np.asarray(real_scores + fake_scores, dtype=float)
    return [f"{np.mean(fake_scores):.3f}", f"{_auc(labels, scores):.3f}"]


def _method_from_file(file_path: str) -> str:
    name = Path(file_path).name
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


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
