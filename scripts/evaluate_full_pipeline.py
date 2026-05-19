"""CAVE 전체 레이어 통합 평가.

이미지 데모, 영상 데모, FFPP 영상 샘플, Layer 7 피해/GNN 시나리오를
한 번에 실행하고 발표/보고서에 바로 넣을 수 있는 CSV/JSON/Markdown
요약을 생성한다.

Adobe Firefly 등 생성형 AI 서비스 출력물은 AI/ML 모델 학습·테스트·개선
목적으로 사용하지 않기 위해 기본 통합 평가 대상에서 제외한다.

실행:
  python scripts/evaluate_full_pipeline.py
  python scripts/evaluate_full_pipeline.py --skip-ffpp
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layers import (  # noqa: E402
    ai_detection,
    c2pa_check,
    cross_layer_audit,
    damage_score,
    fingerprint,
    rppg_check,
    watermark_check,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

IMAGE_AI_VERDICTS = {
    "ai_generated_likely",
    "ai_generated_with_disagreement",
    "ai_suspected_unverified",
    "watermark_compromised",
    "integrity_clash",
}
VIDEO_AI_VERDICTS = set(IMAGE_AI_VERDICTS)
REAL_VERDICTS = {"authentic_likely"}

MEDIA_FIELDNAMES = [
    "dataset",
    "media_type",
    "expected",
    "predicted",
    "method",
    "file",
    "audit_verdict",
    "audit_verdict_kr",
    "expert_review",
    "consistency",
    "c2pa",
    "watermark",
    "ai_detection",
    "rppg",
    "fingerprint",
    "ai_verdict",
    "fp_likelihood",
    "fp_method",
    "fp_family",
    "fp_learned_prob",
    "fp_calibrated_prob",
    "fp_threshold",
    "fp_face_detected",
    "fp_temporal_delta",
    "rppg_learned_prob",
    "rppg_calibrated_prob",
    "rppg_threshold",
    "rppg_face_detected",
    "rppg_peak_bpm",
]

DAMAGE_FIELDNAMES = [
    "scenario",
    "crime_type",
    "diffusion_score",
    "redistribution_score",
    "id_risk_score",
    "severity_score",
    "economic_score",
    "social_score",
    "total_score",
    "grade_kr",
    "gnn_risk",
    "gnn_converted_score",
    "heuristic_diffusion",
    "blend_ratio",
    "gnn_test_acc",
    "gnn_fallback_reason",
    "redistribution_learned_prob",
    "redistribution_converted_score",
    "redistribution_heuristic_score",
    "redistribution_blend_ratio",
    "redistribution_test_auc",
    "redistribution_fallback_reason",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="CAVE 전체 레이어 통합 평가")
    parser.add_argument("--image-dir", default="test_data/demo_images")
    parser.add_argument("--video-real-dir", default="test_data/demo_videos/real")
    parser.add_argument("--video-fake-dir", default="test_data/demo_videos/deepfake")
    parser.add_argument("--ffpp-real-dir", default="test_data/ffpp_c23/eval/real")
    parser.add_argument("--ffpp-fake-dir", default="test_data/ffpp_c23/eval/deepfake")
    parser.add_argument("--ffpp-max-per-label", type=int, default=6)
    parser.add_argument("--ffpp-fake-per-method", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="output/full_eval")
    parser.add_argument("--skip-ffpp", action="store_true", help="느린 FFPP 샘플 평가 생략")
    parser.add_argument("--skip-rppg", action="store_true", help="영상 rPPG 평가 생략")
    parser.add_argument(
        "--include-manual-upload-samples",
        action="store_true",
        help="Firefly 등 수동 업로드 시연용 샘플도 CSV에 포함합니다. 요약 정확도 계산에서는 제외됩니다.",
    )
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    media_rows: list[dict[str, Any]] = []
    damage_rows: list[dict[str, Any]] = []

    image_files = _image_files(ROOT / args.image_dir, include_manual=args.include_manual_upload_samples)
    if image_files:
        manifest = _read_manifest(ROOT / args.image_dir)
        media_rows.extend(
            _evaluate_media_file(
                dataset="demo_images",
                media_type="image",
                path=path,
                expected=_image_expected(path, manifest),
                method=_manifest_method(path, manifest),
                skip_rppg=True,
            )
            for path in image_files
        )

    demo_video_manifest = _read_manifest((ROOT / args.video_real_dir).parents[0])
    demo_real = _video_files(ROOT / args.video_real_dir)
    demo_fake = _video_files(ROOT / args.video_fake_dir)
    for path in demo_real:
        media_rows.append(
            _evaluate_media_file(
                dataset="demo_videos",
                media_type="video",
                path=path,
                expected="real",
                method=_manifest_method(path, demo_video_manifest, default="Original"),
                skip_rppg=args.skip_rppg,
            )
        )
    for path in demo_fake:
        media_rows.append(
            _evaluate_media_file(
                dataset="demo_videos",
                media_type="video",
                path=path,
                expected="deepfake",
                method=_manifest_method(path, demo_video_manifest, default=_method_from_file(path)),
                skip_rppg=args.skip_rppg,
            )
        )

    if not args.skip_ffpp:
        ffpp_real = _sample_files(ROOT / args.ffpp_real_dir, args.ffpp_max_per_label, args.seed)
        ffpp_fake = _sample_fake_files(
            ROOT / args.ffpp_fake_dir,
            args.ffpp_max_per_label,
            args.ffpp_fake_per_method,
            args.seed + 1,
        )
        for path in ffpp_real:
            media_rows.append(
                _evaluate_media_file(
                    dataset="ffpp_eval_sample",
                    media_type="video",
                    path=path,
                    expected="real",
                    method="Original",
                    skip_rppg=args.skip_rppg,
                )
            )
        for path in ffpp_fake:
            media_rows.append(
                _evaluate_media_file(
                    dataset="ffpp_eval_sample",
                    media_type="video",
                    path=path,
                    expected="deepfake",
                    method=_method_from_file(path),
                    skip_rppg=args.skip_rppg,
                )
            )

    damage_rows = _evaluate_damage_scenarios()
    media_summaries = _summarize_media(media_rows)
    damage_summary = _summarize_damage(damage_rows)
    summary = {
        "media": media_summaries,
        "damage": damage_summary,
        "outputs": {
            "media_rows_csv": str(output_dir / "full_pipeline_media_rows.csv"),
            "damage_rows_csv": str(output_dir / "full_pipeline_damage_rows.csv"),
            "summary_json": str(output_dir / "full_pipeline_summary.json"),
            "summary_md": str(output_dir / "full_pipeline_summary.md"),
        },
    }

    _write_csv(output_dir / "full_pipeline_media_rows.csv", media_rows, MEDIA_FIELDNAMES)
    _write_csv(output_dir / "full_pipeline_damage_rows.csv", damage_rows, DAMAGE_FIELDNAMES)
    (output_dir / "full_pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown = _render_markdown(summary)
    (output_dir / "full_pipeline_summary.md").write_text(markdown, encoding="utf-8")

    print(markdown)
    print()
    print(f"saved: {output_dir}")


def _evaluate_media_file(
    *,
    dataset: str,
    media_type: str,
    path: Path,
    expected: str,
    method: str,
    skip_rppg: bool,
) -> dict[str, Any]:
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
    fp_evidence = getattr(l5, "evidence", {}) or {}
    rppg_evidence = getattr(l4, "evidence", {}) if l4 is not None else {}
    predicted = _predicted_label(l6.verdict, media_type)
    return {
        "dataset": dataset,
        "media_type": media_type,
        "expected": expected,
        "predicted": predicted,
        "method": method,
        "file": _relative(path),
        "audit_verdict": l6.verdict,
        "audit_verdict_kr": l6.verdict_kr,
        "expert_review": l6.expert_review_needed,
        "consistency": _round(l6.consistency_score),
        "c2pa": _round(l6.layer_scores.get("c2pa")),
        "watermark": _round(l6.layer_scores.get("watermark")),
        "ai_detection": _round(l6.layer_scores.get("ai_detection")),
        "rppg": _round(l6.layer_scores.get("rppg")),
        "fingerprint": _round(l6.layer_scores.get("fingerprint")),
        "ai_verdict": getattr(l3, "verdict", ""),
        "fp_likelihood": getattr(l5, "ai_likelihood", ""),
        "fp_method": getattr(l5, "generation_method", ""),
        "fp_family": getattr(l5, "model_family", ""),
        "fp_learned_prob": fp_evidence.get("learned_prob", ""),
        "fp_calibrated_prob": fp_evidence.get("calibrated_prob", ""),
        "fp_threshold": fp_evidence.get("threshold", ""),
        "fp_face_detected": fp_evidence.get("face_detected", ""),
        "fp_temporal_delta": fp_evidence.get("temporal_delta", ""),
        "rppg_learned_prob": rppg_evidence.get("learned_prob", ""),
        "rppg_calibrated_prob": rppg_evidence.get("calibrated_prob", ""),
        "rppg_threshold": rppg_evidence.get("threshold", ""),
        "rppg_face_detected": rppg_evidence.get("face_detected", ""),
        "rppg_peak_bpm": rppg_evidence.get("peak_bpm", ""),
    }


def _evaluate_damage_scenarios() -> list[dict[str, Any]]:
    scenarios = [
        ("empty", "default", damage_score.DamageInputs()),
        ("low_spread", "default", damage_score.DamageInputs(
            num_posts=3,
            num_shares=20,
            num_views=500,
            spread_speed_hours=72,
        )),
        ("deepfake_sexual_demo", "deepfake_sexual", damage_score.DamageInputs.example_deepfake_sexual()),
        ("financial_fraud_demo", "financial_fraud", damage_score.DamageInputs.example_financial_fraud()),
        ("viral_reupload", "deepfake_sexual", damage_score.DamageInputs(
            num_posts=1500,
            num_platforms=6,
            num_shares=200_000,
            num_views=2_000_000,
            spread_speed_hours=2,
            has_variants=True,
            on_closed_platforms=True,
            reappeared_after_deletion=True,
            face_match_score=1.0,
            real_name_mentioned=True,
            affiliation_revealed=True,
            is_sexual_manipulation=True,
            is_defamatory=True,
            reputation_damaged=True,
        )),
    ]
    rows: list[dict[str, Any]] = []
    for name, crime_type, inputs in scenarios:
        result = damage_score.run(inputs, crime_type=crime_type)
        rows.append({
            "scenario": name,
            "crime_type": crime_type,
            "diffusion_score": _round(result.diffusion_score),
            "redistribution_score": _round(result.redistribution_score),
            "id_risk_score": _round(result.id_risk_score),
            "severity_score": _round(result.severity_score),
            "economic_score": _round(result.economic_score),
            "social_score": _round(result.social_score),
            "total_score": _round(result.total_score),
            "grade_kr": result.grade_kr,
            "gnn_risk": _round(result.gnn_risk),
            "gnn_converted_score": _round(result.gnn_converted_score),
            "heuristic_diffusion": _round(result.heuristic_diffusion),
            "blend_ratio": result.blend_ratio,
            "gnn_test_acc": _round(result.gnn_test_acc),
            "gnn_fallback_reason": result.gnn_fallback_reason or "",
            "redistribution_learned_prob": _round(result.redistribution_learned_prob),
            "redistribution_converted_score": _round(result.redistribution_converted_score),
            "redistribution_heuristic_score": _round(result.redistribution_heuristic_score),
            "redistribution_blend_ratio": result.redistribution_blend_ratio,
            "redistribution_test_auc": _round(result.redistribution_test_auc),
            "redistribution_fallback_reason": result.redistribution_fallback_reason or "",
        })
    return rows


def _summarize_media(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        summaries[dataset] = _dataset_summary(dataset_rows)
    summaries["overall"] = _dataset_summary(rows)
    return summaries


def _dataset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = [row for row in rows if row["expected"] in {"real", "ai", "deepfake"}]
    decided = [row for row in comparable if row["predicted"] != "review"]
    exact_all = [row for row in comparable if row["predicted"] == row["expected"]]
    exact_decided = [row for row in decided if row["predicted"] == row["expected"]]
    real_rows = [row for row in comparable if row["expected"] == "real"]
    positive_rows = [row for row in comparable if row["expected"] in {"ai", "deepfake"}]
    false_positive = [
        row for row in real_rows
        if row["predicted"] in {"ai", "deepfake"}
    ]
    false_negative = [
        row for row in positive_rows
        if row["predicted"] == "real"
    ]
    review_rows = [row for row in comparable if row["predicted"] == "review"]

    summary: dict[str, Any] = {
        "files": len(rows),
        "comparable": len(comparable),
        "decided": len(decided),
        "exact_all": len(exact_all),
        "exact_all_rate": _safe_rate(len(exact_all), len(comparable)),
        "exact_decided": len(exact_decided),
        "exact_decided_rate": _safe_rate(len(exact_decided), len(decided)),
        "review": len(review_rows),
        "review_rate": _safe_rate(len(review_rows), len(comparable)),
        "false_positive": len(false_positive),
        "false_negative": len(false_negative),
        "expert_review": sum(1 for row in rows if row["expert_review"]),
    }

    labels = np.asarray([0 if row["expected"] == "real" else 1 for row in comparable], dtype=int)
    for key in ("ai_detection", "rppg", "fingerprint"):
        metric_rows = [
            row for row in comparable
            if row.get(key) not in (None, "", "N/A")
        ]
        if not metric_rows:
            continue
        metric_labels = np.asarray([0 if row["expected"] == "real" else 1 for row in metric_rows], dtype=int)
        scores = np.asarray([float(row[key]) for row in metric_rows], dtype=float)
        summary[key] = _classification_metrics(metric_labels, scores)

    if len(labels) == 0:
        return summary
    return summary


def _summarize_damage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scenarios": len(rows),
        "max_total_score": max((row["total_score"] for row in rows), default=0.0),
        "high_risk_scenarios": [
            row["scenario"] for row in rows
            if float(row["total_score"]) >= 16.0
        ],
        "rows": rows,
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# CAVE Full Pipeline Evaluation",
        "",
        "## Media Summary",
        "",
        "| Dataset | Files | Decided Exact | All Exact | Review | FP | FN | L3 AUC | L4 AUC | L5 AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, item in summary["media"].items():
        lines.append(
            "| {dataset} | {files} | {decided_exact} | {all_exact} | {review} | {fp} | {fn} | {l3} | {l4} | {l5} |".format(
                dataset=dataset,
                files=item["files"],
                decided_exact=_ratio_text(item["exact_decided"], item["decided"], item["exact_decided_rate"]),
                all_exact=_ratio_text(item["exact_all"], item["comparable"], item["exact_all_rate"]),
                review=f"{item['review']} ({item['review_rate']:.3f})",
                fp=item["false_positive"],
                fn=item["false_negative"],
                l3=_metric_text(item.get("ai_detection")),
                l4=_metric_text(item.get("rppg")),
                l5=_metric_text(item.get("fingerprint")),
            )
        )

    lines.extend([
        "",
        "## Damage / GNN Summary",
        "",
        "| Scenario | Crime Type | Total | Grade | GNN Risk | GNN Evidence | Redistribution Evidence |",
        "|---|---|---:|---|---:|---|---|",
    ])
    for row in summary["damage"]["rows"]:
        gnn_evidence = (
            f"converted {row['gnn_converted_score']} / heuristic {row['heuristic_diffusion']} / "
            f"blend {row['blend_ratio']}"
            if row["gnn_risk"] not in (None, "", "N/A")
            else f"fallback {row['gnn_fallback_reason']}"
        )
        redis_evidence = (
            f"prob {row['redistribution_learned_prob']} / converted {row['redistribution_converted_score']} / "
            f"heuristic {row['redistribution_heuristic_score']} / blend {row['redistribution_blend_ratio']}"
            if row["redistribution_learned_prob"] not in (None, "", "N/A")
            else f"fallback {row['redistribution_fallback_reason']}"
        )
        lines.append(
            f"| {row['scenario']} | {row['crime_type']} | {row['total_score']} | "
            f"{row['grade_kr']} | {_fmt_md(row['gnn_risk'])} | {gnn_evidence} | {redis_evidence} |"
        )

    lines.extend([
        "",
        "## Output Files",
        "",
    ])
    for label, path in summary["outputs"].items():
        lines.append(f"- `{label}`: `{path}`")
    return "\n".join(lines)


def _image_files(root: Path, include_manual: bool = False) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and (include_manual or not _is_manual_upload_sample(path))
    )


def _video_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def _sample_files(root: Path, max_count: int, seed: int) -> list[Path]:
    files = _video_files(root)
    rng = random.Random(seed)
    rng.shuffle(files)
    return files[:max_count] if max_count > 0 else files


def _sample_fake_files(root: Path, max_count: int, per_method: int, seed: int) -> list[Path]:
    if per_method <= 0:
        return _sample_files(root, max_count, seed)
    groups: dict[str, list[Path]] = {}
    for path in _video_files(root):
        groups.setdefault(_method_from_file(path), []).append(path)
    rng = random.Random(seed)
    selected: list[Path] = []
    for method in sorted(groups):
        files = groups[method]
        rng.shuffle(files)
        selected.extend(files[:per_method])
    rng.shuffle(selected)
    return selected[:max_count] if max_count > 0 and len(selected) > max_count and per_method <= 0 else selected


def _read_manifest(root: Path) -> dict[str, dict[str, str]]:
    manifest_path = root / "manifest.csv"
    if not manifest_path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            target = row.get("target_path", "")
            if not target:
                continue
            rows[str((ROOT / target).resolve())] = row
    return rows


def _image_expected(path: Path, manifest: dict[str, dict[str, str]]) -> str:
    if _is_manual_upload_sample(path):
        return "manual_upload"
    row = manifest.get(str(path.resolve()), {})
    label = row.get("label", "").lower()
    if label == "real":
        return "real"
    if label == "fake":
        return "ai"
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    if "real" in parts or "real" in name:
        return "real"
    if "fake" in parts or "fake" in name:
        return "ai"
    return "unknown"


def _is_manual_upload_sample(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    return bool(parts & {"manual_upload", "ai_generated"}) or "firefly" in name or "adobe" in name


def _manifest_method(path: Path, manifest: dict[str, dict[str, str]], default: str = "unknown") -> str:
    row = manifest.get(str(path.resolve()), {})
    return row.get("method") or default


def _method_from_file(path: Path) -> str:
    name = path.name
    if "_" not in name:
        return "unknown"
    return name.split("_", 1)[0]


def _predicted_label(verdict: str, media_type: str) -> str:
    if media_type == "image":
        if verdict in IMAGE_AI_VERDICTS:
            return "ai"
        if verdict in REAL_VERDICTS:
            return "real"
        return "review"
    if verdict in VIDEO_AI_VERDICTS:
        return "deepfake"
    if verdict in REAL_VERDICTS:
        return "real"
    return "review"


def _classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    thresholds = sorted(set(float(score) for score in scores))
    candidates = [0.5]
    candidates.extend(thresholds)
    candidates.extend((a + b) / 2 for a, b in zip(thresholds[:-1], thresholds[1:]))
    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in candidates:
        accuracy = float(np.mean((scores >= threshold).astype(int) == labels))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return {
        "auc": _round(_auc(labels, scores)),
        "accuracy": _round(float(np.mean((scores >= 0.5).astype(int) == labels))),
        "best_threshold": _round(best_threshold),
        "best_accuracy": _round(best_accuracy),
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


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _round(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 3)
    return value


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 3) if denominator else 0.0


def _ratio_text(numerator: int, denominator: int, rate: float) -> str:
    if denominator == 0:
        return "N/A"
    return f"{numerator}/{denominator} ({rate:.3f})"


def _metric_text(metric: Optional[dict[str, float]]) -> str:
    if not metric:
        return "N/A"
    return f"{metric['auc']:.3f}"


def _fmt_md(value: Any) -> str:
    if value in (None, "", "N/A"):
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return str(value)


if __name__ == "__main__":
    main()
