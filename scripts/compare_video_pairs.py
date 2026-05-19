"""real/fake 영상 쌍 비교 리포트.

실행:
  python scripts/compare_video_pairs.py
  python scripts/compare_video_pairs.py --real-dir test_data/real --fake-dir test_data/deepfake
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layers import ai_detection, fingerprint, rppg_check  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="CAVE 영상 real/fake 쌍 비교")
    parser.add_argument("--real-dir", default="test_data/real")
    parser.add_argument("--fake-dir", default="test_data/deepfake")
    parser.add_argument("--output", default="")
    parser.add_argument("--unpaired", action="store_true")
    parser.add_argument("--max-per-label", type=int, default=80)
    parser.add_argument("--fake-per-method", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    real_dir = Path(args.real_dir)
    fake_dir = Path(args.fake_dir)
    rows = []

    if args.unpaired:
        rows = _collect_unpaired(real_dir, fake_dir, args.max_per_label, args.fake_per_method, args.seed)
        _print_unpaired_table(rows)
        _print_unpaired_metrics(rows)
        if args.output:
            _write_csv(Path(args.output), rows)
        return

    for real_path in sorted(real_dir.glob("*.mp4")):
        fake_path = fake_dir / f"{real_path.stem}_fake{real_path.suffix}"
        if not fake_path.exists():
            continue

        real_scores = _analyze_video(real_path)
        fake_scores = _analyze_video(fake_path)
        rows.append({
            "pair": real_path.stem,
            "real_file": str(real_path),
            "fake_file": str(fake_path),
            **{f"real_{k}": v for k, v in real_scores.items()},
            **{f"fake_{k}": v for k, v in fake_scores.items()},
            "delta_ai_detection": _delta(fake_scores["ai_detection"], real_scores["ai_detection"]),
            "delta_fingerprint": _delta(fake_scores["fingerprint"], real_scores["fingerprint"]),
            "delta_rppg": _delta(fake_scores["rppg"], real_scores["rppg"]),
        })

    _print_table(rows)
    _print_metrics(rows)
    if args.output:
        _write_csv(Path(args.output), rows)


def _collect_unpaired(
    real_dir: Path,
    fake_dir: Path,
    max_per_label: int,
    fake_per_method: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    real_files = _sample_files(real_dir, max_per_label, seed)
    fake_files = _sample_fake_files(fake_dir, max_per_label, fake_per_method, seed + 1)
    for label, label_name, files in ((0, "real", real_files), (1, "fake", fake_files)):
        for path in files:
            scores = _analyze_video(path)
            rows.append({
                "label": label,
                "label_name": label_name,
                "file": str(path),
                **scores,
            })
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
        groups.setdefault(_method_from_file(str(path)), []).append(path)

    selected: list[Path] = []
    for method in sorted(groups):
        files = groups[method]
        rng.shuffle(files)
        selected.extend(files[:fake_per_method])
    rng.shuffle(selected)
    return selected


def _analyze_video(path: Path) -> dict[str, float | str | None]:
    ai = ai_detection.run(str(path))
    fp = fingerprint.run(str(path))
    rppg = rppg_check.run(str(path))
    return {
        "ai_detection": ai.ai_score,
        "fingerprint": fp.ai_score,
        "rppg": rppg.ai_score,
        "ai_notes": ai.notes,
        "fingerprint_notes": fp.notes,
        "rppg_notes": rppg.notes,
    }


def _print_unpaired_table(rows: list[dict]) -> None:
    print("label,file,ai_detection,fingerprint,rppg")
    for row in rows[:40]:
        print(
            f"{row['label_name']},{Path(row['file']).name},"
            f"{_fmt(row['ai_detection'])},{_fmt(row['fingerprint'])},{_fmt(row['rppg'])}"
        )
    if len(rows) > 40:
        print(f"... {len(rows) - 40} more rows")


def _print_unpaired_metrics(rows: list[dict]) -> None:
    if not rows:
        return
    print()
    for metric_name in ("ai_detection", "fingerprint", "rppg"):
        labels = np.asarray([row["label"] for row in rows if row.get(metric_name) is not None], dtype=int)
        scores = np.asarray([float(row[metric_name]) for row in rows if row.get(metric_name) is not None], dtype=float)
        if len(scores) == 0:
            continue
        metrics = _classification_metrics(labels, scores)
        print(
            f"{metric_name}: auc={metrics['auc']:.3f}, "
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
    print("fake_method,count,ai_mean,ai_auc_vs_real,fp_mean,fp_auc_vs_real,rppg_mean,rppg_auc_vs_real")
    for method in methods:
        method_rows = [row for row in fake_rows if _method_from_file(row["file"]) == method]
        values = [
            method,
            str(len(method_rows)),
            *_method_metric_columns(real_rows, method_rows, "ai_detection"),
            *_method_metric_columns(real_rows, method_rows, "fingerprint"),
            *_method_metric_columns(real_rows, method_rows, "rppg"),
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


def _delta(fake_value, real_value):
    if fake_value is None or real_value is None:
        return None
    return round(float(fake_value) - float(real_value), 3)


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("매칭되는 real/fake 영상 쌍이 없습니다.")
        return
    print("pair,real_ai,fake_ai,delta_ai,real_fp,fake_fp,delta_fp,real_rppg,fake_rppg,delta_rppg")
    for row in rows:
        print(
            f"{row['pair']},"
            f"{_fmt(row['real_ai_detection'])},{_fmt(row['fake_ai_detection'])},{_fmt(row['delta_ai_detection'])},"
            f"{_fmt(row['real_fingerprint'])},{_fmt(row['fake_fingerprint'])},{_fmt(row['delta_fingerprint'])},"
            f"{_fmt(row['real_rppg'])},{_fmt(row['fake_rppg'])},{_fmt(row['delta_rppg'])}"
        )


def _print_metrics(rows: list[dict]) -> None:
    if not rows:
        return
    print()
    for metric_name, real_key, fake_key in (
        ("ai_detection", "real_ai_detection", "fake_ai_detection"),
        ("fingerprint", "real_fingerprint", "fake_fingerprint"),
        ("rppg", "real_rppg", "fake_rppg"),
    ):
        labels: list[int] = []
        scores: list[float] = []
        for row in rows:
            for label, key in ((0, real_key), (1, fake_key)):
                value = row.get(key)
                if value is None:
                    continue
                labels.append(label)
                scores.append(float(value))
        if not scores:
            continue
        metrics = _classification_metrics(np.asarray(labels), np.asarray(scores))
        print(
            f"{metric_name}: auc={metrics['auc']:.3f}, "
            f"acc@0.5={metrics['accuracy']:.3f}, "
            f"best_threshold={metrics['best_threshold']:.3f}, "
            f"best_acc={metrics['best_accuracy']:.3f}"
        )


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


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV 저장: {path}")


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    main()
