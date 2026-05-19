"""FFPP 영상 샘플의 레이어 3/4/5 및 Cross-layer Audit 판정 검증.

실행:
  python scripts/evaluate_video_audit.py
  python scripts/evaluate_video_audit.py --max-per-label 12 --fake-per-method 2
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layers import ai_detection, c2pa_check, cross_layer_audit, fingerprint, rppg_check, watermark_check  # noqa: E402


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AI_VERDICTS = {
    "ai_generated_likely",
    "ai_generated_with_disagreement",
    "ai_suspected_unverified",
    "watermark_compromised",
    "integrity_clash",
}
REAL_VERDICTS = {"authentic_likely"}


def main() -> None:
    parser = argparse.ArgumentParser(description="CAVE 영상 Cross-layer Audit 검증")
    parser.add_argument("--real-dir", default="test_data/ffpp_c23/eval/real")
    parser.add_argument("--fake-dir", default="test_data/ffpp_c23/eval/deepfake")
    parser.add_argument("--max-per-label", type=int, default=6)
    parser.add_argument("--fake-per-method", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="")
    parser.add_argument("--skip-rppg", action="store_true", help="레이어 4 rPPG를 제외하고 빠르게 평가")
    args = parser.parse_args()

    real_files = _sample_files(ROOT / args.real_dir, args.max_per_label, args.seed)
    fake_files = _sample_fake_files(ROOT / args.fake_dir, args.max_per_label, args.fake_per_method, args.seed + 1)
    if not real_files and not fake_files:
        raise SystemExit("평가할 영상을 찾지 못했습니다.")

    rows: list[dict] = []
    for label, label_name, files in ((0, "real", real_files), (1, "deepfake", fake_files)):
        for path in files:
            rows.append(_evaluate(path, label=label, label_name=label_name, skip_rppg=args.skip_rppg))

    _print_rows(rows)
    _print_summary(rows)
    if args.output:
        _write_csv(Path(args.output), rows)


def _evaluate(path: Path, label: int, label_name: str, skip_rppg: bool) -> dict:
    l1 = c2pa_check.run(str(path))
    l2 = watermark_check.run(str(path))
    l3 = ai_detection.run(str(path))
    l4 = None if skip_rppg else rppg_check.run(str(path))
    l5 = fingerprint.run(str(path))
    l6 = cross_layer_audit.run({
        "c2pa": l1,
        "watermark": l2,
        "ai_detection": l3,
        "rppg": l4,
        "fingerprint": l5,
    })
    predicted = _predicted_label(l6.verdict)
    return {
        "label": label,
        "label_name": label_name,
        "method": _method_from_file(path) if label == 1 else "real",
        "file": str(path.relative_to(ROOT)),
        "predicted": predicted,
        "audit_verdict": l6.verdict,
        "audit_verdict_kr": l6.verdict_kr,
        "expert_review": l6.expert_review_needed,
        "consistency": l6.consistency_score,
        "c2pa": l6.layer_scores.get("c2pa"),
        "watermark": l6.layer_scores.get("watermark"),
        "ai_detection": l6.layer_scores.get("ai_detection"),
        "rppg": l6.layer_scores.get("rppg"),
        "fingerprint": l6.layer_scores.get("fingerprint"),
        "ai_verdict": l3.verdict,
        "fp_likelihood": l5.ai_likelihood,
        "fp_method": l5.generation_method,
        "fp_family": l5.model_family,
        "fp_learned_prob": getattr(l5, "evidence", {}).get("learned_prob", ""),
        "fp_calibrated_prob": getattr(l5, "evidence", {}).get("calibrated_prob", ""),
        "fp_threshold": getattr(l5, "evidence", {}).get("threshold", ""),
        "fp_face_detected": getattr(l5, "evidence", {}).get("face_detected", ""),
        "fp_temporal_delta": getattr(l5, "evidence", {}).get("temporal_delta", ""),
        "rppg_learned_prob": getattr(l4, "evidence", {}).get("learned_prob", "") if l4 is not None else "",
        "rppg_calibrated_prob": getattr(l4, "evidence", {}).get("calibrated_prob", "") if l4 is not None else "",
        "rppg_threshold": getattr(l4, "evidence", {}).get("threshold", "") if l4 is not None else "",
        "rppg_face_detected": getattr(l4, "evidence", {}).get("face_detected", "") if l4 is not None else "",
        "rppg_peak_bpm": getattr(l4, "evidence", {}).get("peak_bpm", "") if l4 is not None else "",
    }


def _sample_files(root: Path, max_count: int, seed: int) -> list[Path]:
    files = _video_files(root)
    rng = random.Random(seed)
    rng.shuffle(files)
    if max_count > 0:
        return files[:max_count]
    return files


def _sample_fake_files(root: Path, max_count: int, fake_per_method: int, seed: int) -> list[Path]:
    if fake_per_method <= 0:
        return _sample_files(root, max_count, seed)

    groups: dict[str, list[Path]] = {}
    for path in _video_files(root):
        groups.setdefault(_method_from_file(path), []).append(path)

    rng = random.Random(seed)
    selected: list[Path] = []
    for method in sorted(groups):
        files = groups[method]
        rng.shuffle(files)
        selected.extend(files[:fake_per_method])
    rng.shuffle(selected)
    return selected


def _video_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def _method_from_file(path: Path) -> str:
    name = path.name
    if "_" not in name:
        return "unknown"
    return name.split("_", 1)[0]


def _predicted_label(verdict: str) -> str:
    if verdict in AI_VERDICTS:
        return "deepfake"
    if verdict in REAL_VERDICTS:
        return "real"
    return "review"


def _print_rows(rows: list[dict]) -> None:
    print(
        "label,method,file,predicted,audit_verdict,expert_review,"
        "ai_detection,rppg,fingerprint,fp_likelihood,fp_method,"
        "fp_learned_prob,fp_calibrated_prob,fp_threshold,fp_face_detected,fp_temporal_delta,"
        "rppg_learned_prob,rppg_calibrated_prob,rppg_threshold,rppg_face_detected,rppg_peak_bpm"
    )
    for row in rows:
        print(
            f"{row['label_name']},{row['method']},{Path(row['file']).name},"
            f"{row['predicted']},{row['audit_verdict']},{row['expert_review']},"
            f"{_fmt(row['ai_detection'])},{_fmt(row['rppg'])},{_fmt(row['fingerprint'])},"
            f"{row['fp_likelihood']},{row['fp_method']},"
            f"{row['fp_learned_prob']},{row['fp_calibrated_prob']},{row['fp_threshold']},"
            f"{row['fp_face_detected']},{row['fp_temporal_delta']},"
            f"{row['rppg_learned_prob']},{row['rppg_calibrated_prob']},{row['rppg_threshold']},"
            f"{row['rppg_face_detected']},{row['rppg_peak_bpm']}"
        )


def _print_summary(rows: list[dict]) -> None:
    if not rows:
        return
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    predicted = np.asarray([1 if row["predicted"] == "deepfake" else 0 for row in rows], dtype=int)
    comparable = [row for row in rows if row["predicted"] in {"real", "deepfake"}]
    exact = [row for row in comparable if _label_name(row["label"]) == row["predicted"]]
    real_rows = [row for row in rows if row["label"] == 0]
    fake_rows = [row for row in rows if row["label"] == 1]
    false_positive = [row for row in real_rows if row["predicted"] == "deepfake"]
    false_negative = [row for row in fake_rows if row["predicted"] == "real"]
    real_review = [row for row in real_rows if row["predicted"] == "review"]
    fake_review = [row for row in fake_rows if row["predicted"] == "review"]

    print()
    print("summary")
    print(f"  files={len(rows)}, real={len(real_rows)}, deepfake={len(fake_rows)}")
    if comparable:
        print(f"  exact_match={len(exact)}/{len(comparable)} ({len(exact) / len(comparable):.3f})")
    print(f"  false_positive_real_as_deepfake={len(false_positive)}")
    print(f"  false_negative_deepfake_as_real={len(false_negative)}")
    print(f"  real_review={len(real_review)}")
    print(f"  deepfake_review={len(fake_review)}")
    print(f"  audit_deepfake_rate={float(np.mean(predicted)):.3f}")

    for key in ("ai_detection", "rppg", "fingerprint"):
        metric_rows = [row for row in rows if row.get(key) is not None]
        if not metric_rows:
            continue
        metric_labels = np.asarray([row["label"] for row in metric_rows], dtype=int)
        scores = np.asarray([float(row[key]) for row in metric_rows], dtype=float)
        metrics = _classification_metrics(metric_labels, scores)
        print(
            f"  {key}: auc={metrics['auc']:.3f}, acc@0.5={metrics['accuracy']:.3f}, "
            f"best_threshold={metrics['best_threshold']:.3f}, best_acc={metrics['best_accuracy']:.3f}"
        )


def _label_name(label: int) -> str:
    return "deepfake" if label == 1 else "real"


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
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
