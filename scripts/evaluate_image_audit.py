"""이미지 샘플의 레이어 3/5 및 Cross-layer Audit 판정 검증.

실행:
  python scripts/evaluate_image_audit.py --input-dir test_data/demo_images
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layers import ai_detection, c2pa_check, cross_layer_audit, fingerprint, rppg_check, watermark_check  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
AI_VERDICTS = {
    "ai_generated_likely",
    "ai_generated_with_disagreement",
    "ai_suspected_unverified",
    "watermark_compromised",
    "integrity_clash",
}
REAL_VERDICTS = {"authentic_likely"}


def main() -> None:
    parser = argparse.ArgumentParser(description="CAVE 이미지 Audit 검증")
    parser.add_argument("--input-dir", default="test_data/demo_images")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    files = _image_files(ROOT / args.input_dir)
    if not files:
        raise SystemExit(f"이미지 파일을 찾지 못했습니다: {args.input_dir}")

    if args.max_files > 0:
        rng = random.Random(args.seed)
        rng.shuffle(files)
        files = files[: args.max_files]

    rows = [_evaluate(path) for path in files]
    _print_rows(rows)
    _print_summary(rows)


def _image_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _evaluate(path: Path) -> dict:
    l1 = c2pa_check.run(str(path))
    l2 = watermark_check.run(str(path))
    l3 = ai_detection.run(str(path))
    l4 = rppg_check.run(str(path))
    l5 = fingerprint.run(str(path))
    l6 = cross_layer_audit.run({
        "c2pa": l1,
        "watermark": l2,
        "ai_detection": l3,
        "rppg": l4,
        "fingerprint": l5,
    })

    expected = _expected_label(path)
    predicted = _predicted_label(l6.verdict)
    return {
        "file": path.relative_to(ROOT),
        "expected": expected,
        "predicted": predicted,
        "c2pa": l6.layer_scores.get("c2pa"),
        "watermark": l6.layer_scores.get("watermark"),
        "ai_detection": l6.layer_scores.get("ai_detection"),
        "fingerprint": l6.layer_scores.get("fingerprint"),
        "verdict": l6.verdict,
        "verdict_kr": l6.verdict_kr,
        "expert_review": l6.expert_review_needed,
        "consistency": l6.consistency_score,
        "fp_method": l5.generation_method,
        "fp_family": l5.model_family,
    }


def _expected_label(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    if "real" in parts or "real" in name:
        return "real"
    if "fake" in parts or "ai_generated" in parts or "fake" in name or "firefly" in name:
        return "ai"
    return "unknown"


def _predicted_label(verdict: str) -> str:
    if verdict in AI_VERDICTS:
        return "ai"
    if verdict in REAL_VERDICTS:
        return "real"
    return "review"


def _print_rows(rows: list[dict]) -> None:
    print(
        "file,expected,predicted,c2pa,watermark,ai_detection,fingerprint,"
        "fp_method,fp_family,audit_verdict,expert_review,consistency"
    )
    for row in rows:
        print(
            f"{row['file']},{row['expected']},{row['predicted']},"
            f"{_fmt(row['c2pa'])},{_fmt(row['watermark'])},"
            f"{_fmt(row['ai_detection'])},{_fmt(row['fingerprint'])},"
            f"{row['fp_method']},{row['fp_family']},"
            f"{row['verdict']},{row['expert_review']},{row['consistency']:.3f}"
        )


def _print_summary(rows: list[dict]) -> None:
    comparable = [row for row in rows if row["expected"] in {"real", "ai"}]
    exact = [
        row for row in comparable
        if row["predicted"] == row["expected"]
    ]
    review = [row for row in comparable if row["predicted"] == "review"]
    ai_scores = [row["ai_detection"] for row in rows if row["ai_detection"] is not None]
    fp_scores = [row["fingerprint"] for row in rows if row["fingerprint"] is not None]

    print()
    print("summary")
    print(f"  files={len(rows)}, comparable={len(comparable)}")
    if comparable:
        print(f"  exact_match={len(exact)}/{len(comparable)} ({len(exact) / len(comparable):.3f})")
        print(f"  review_or_uncertain={len(review)}")
    if ai_scores:
        print(f"  layer3_ai_score_mean={sum(ai_scores) / len(ai_scores):.3f}")
    if fp_scores:
        print(f"  layer5_fp_score_mean={sum(fp_scores) / len(fp_scores):.3f}")


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
