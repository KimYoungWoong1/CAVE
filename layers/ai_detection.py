"""레이어 3: AI 탐지 모델 분석.

현재 구현: Hugging Face `Organika/sdxl-detector` 이미지 분류 모델.
  - SDXL/diffusion 계열 AI 생성 이미지 탐지에 특화된 모델
  - 출력 라벨 artificial/real을 CAVE의 ai_probability로 변환

한계:
  - SDXL 계열에 강한 탐지기이므로 모든 생성 모델에 일반화된 판정은 아님.
  - 얼굴 합성/프레임/조명 불일치 세부 분석은 후속 모델에서 보강 예정.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np
from PIL import Image

from layers.deepfake_image_detectors import predict_image_detector_suite
from layers.deepfake_video_detectors import predict_video_detector_suite
from layers.video_utils import is_video_file, sample_video_face_crops


MODEL_ID = "Organika/sdxl-detector"
VIDEO_MODEL_ID = "DeepfakeBench-style ensemble"
AI_HIGH_THRESHOLD = 0.75
AI_LOW_THRESHOLD = 0.45
VIDEO_AI_HIGH_THRESHOLD = 0.60
VIDEO_AI_LOW_THRESHOLD = 0.45
VIDEO_SAMPLE_FRAMES = 16

_PIPELINE: Any = None


@dataclass
class AIDetectionResult:
    ai_probability: Optional[float]     # 0.0~1.0
    face_synthesis_level: Optional[str] # '높음' | '중간' | '낮음' | None
    frame_inconsistency: Optional[bool]
    lighting_inconsistency: Optional[bool]
    verdict: Optional[str]              # '진본 의심' | '위조 의심' | '불확실' | None
    ai_score: Optional[float]
    notes: str


def run(file_path: str) -> AIDetectionResult:
    """AI 생성 이미지/영상 탐지 실행."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    if is_video_file(path):
        return _run_video(path)
    return _run_image(path)


def _run_image(path: Path) -> AIDetectionResult:
    try:
        classifier = _get_pipeline()
        raw_result = classifier(str(path))
        diffusion_probability = _extract_ai_probability(raw_result)
        suite = predict_image_detector_suite(path, diffusion_probability=diffusion_probability)
        ai_probability = suite.probability
    except ImportError:
        return _unavailable_result(
            "transformers가 설치되지 않아 AI 탐지 모델을 실행할 수 없습니다. "
            "`pip install transformers` 후 재실행하세요."
        )
    except Exception as exc:
        return _unavailable_result(
            f"AI 탐지 모델 실행 실패: {type(exc).__name__}: {exc}"
        )

    face_level, verdict = _interpret_probability(ai_probability)

    return AIDetectionResult(
        ai_probability=ai_probability,
        face_synthesis_level=face_level,
        frame_inconsistency=None,
        lighting_inconsistency=None,
        verdict=verdict,
        ai_score=ai_probability,
        notes=(
            f"{MODEL_ID} diffusion detector + 얼굴 조작 detector 앙상블 기반 이미지 분석. "
            f"diffusion={diffusion_probability:.3f}, final={ai_probability:.3f}. "
            f"{suite.notes}. "
            "정지 이미지는 SDXL/diffusion 생성과 얼굴 조작 신호를 함께 해석."
        ),
    )


def _run_video(path: Path) -> AIDetectionResult:
    try:
        samples = sample_video_face_crops(path, num_frames=VIDEO_SAMPLE_FRAMES)
        face_crops = [sample.image for sample in samples]
        luminance = [sample.luminance for sample in samples]
        face_count = sum(1 for sample in samples if sample.detected)
        detector_counts = _detector_counts(sample.detector for sample in samples)
        face_ratio = face_count / max(len(face_crops), 1)
        suite = predict_video_detector_suite(face_crops, face_ratio=face_ratio)
    except ImportError:
        return _unavailable_result(
            "torchvision, torch, timm, huggingface_hub 또는 opencv-python이 설치되지 않아 영상 AI 탐지 모델을 실행할 수 없습니다."
        )
    except Exception as exc:
        return _unavailable_result(
            f"영상 딥페이크 탐지 앙상블 실행 실패: {type(exc).__name__}: {exc}"
        )

    ai_probability = suite.calibrated_probability
    face_level, verdict = _interpret_video_probability(ai_probability)

    frame_stds = [item.frame_std for item in suite.predictions if item.frame_std is not None]
    frame_std = float(np.mean(frame_stds)) if frame_stds else 0.0
    model_spread = float(np.std([item.probability for item in suite.predictions]))
    frame_inconsistency = frame_std >= 0.25 or model_spread >= 0.25
    lighting_inconsistency = _lighting_inconsistency(luminance)
    model_summary = "; ".join(
        f"{item.name}={item.probability:.3f}"
        for item in suite.predictions
    )
    errors = ""
    if suite.errors:
        errors = " 일부 detector 실패: " + " | ".join(suite.errors[:3])

    return AIDetectionResult(
        ai_probability=ai_probability,
        face_synthesis_level=face_level,
        frame_inconsistency=frame_inconsistency,
        lighting_inconsistency=lighting_inconsistency,
        verdict=verdict,
        ai_score=ai_probability,
        notes=(
            f"{VIDEO_MODEL_ID}를 영상 대표 얼굴 crop {len(face_crops)}장에 적용. "
            f"raw_ensemble={suite.raw_probability:.3f}, calibrated={ai_probability:.3f}, "
            f"model_spread={model_spread:.3f}, frame_std={frame_std:.3f}. "
            f"models=[{model_summary}]. "
            f"calibration={suite.calibration_note}. "
            f"face_detected={face_count}/{len(face_crops)}, detectors={detector_counts}. "
            "영상 판정은 Xception/EfficientNet-B4/R3D-18 계열 얼굴 crop 앙상블이며, "
            f"C2PA/rPPG/핑거프린트와 함께 해석.{errors}"
        ),
    )


def _get_pipeline():
    """Hugging Face pipeline을 최초 1회만 로드한다."""
    global _PIPELINE
    if _PIPELINE is None:
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:
            raise ImportError from exc

        _PIPELINE = pipeline(
            "image-classification",
            model=MODEL_ID,
            top_k=None,
        )
    return _PIPELINE


def _extract_ai_probability(raw_result: Any) -> float:
    """pipeline 출력에서 artificial 라벨 확률을 추출한다."""
    results = raw_result
    if isinstance(results, list) and results and isinstance(results[0], list):
        results = results[0]
    if not isinstance(results, list):
        raise ValueError(f"예상하지 못한 모델 출력 형식: {type(raw_result).__name__}")

    scores: dict[str, float] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0.0))
        scores[label] = score

    artificial = _find_score(scores, ("artificial", "ai", "fake", "generated", "synthetic"))
    if artificial is not None:
        return _clamp01(artificial)

    real = _find_score(scores, ("real", "human", "natural", "authentic"))
    if real is not None:
        return _clamp01(1.0 - real)

    raise ValueError(f"artificial/real 라벨을 찾을 수 없습니다: {scores}")


def _extract_ai_probabilities(raw_results: Any) -> list[float]:
    if isinstance(raw_results, list) and raw_results and isinstance(raw_results[0], dict):
        return [_extract_ai_probability(raw_results)]
    if not isinstance(raw_results, list):
        return [_extract_ai_probability(raw_results)]

    probabilities: list[float] = []
    for item in raw_results:
        probabilities.append(_extract_ai_probability(item))
    return probabilities


def _detector_counts(detectors) -> str:
    counts: dict[str, int] = {}
    for detector in detectors:
        counts[str(detector)] = counts.get(str(detector), 0) + 1
    return ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def _lighting_inconsistency(luminance: list[float]) -> Optional[bool]:
    if len(luminance) < 3:
        return None
    arr = np.asarray(luminance, dtype=np.float64)
    cv = float(np.std(arr) / (np.mean(arr) + 1e-8))
    return cv >= 0.22


def _find_score(scores: dict[str, float], needles: tuple[str, ...]) -> Optional[float]:
    for label, score in scores.items():
        if any(needle in label for needle in needles):
            return score
    return None


def _interpret_probability(ai_probability: float) -> tuple[str, str]:
    if ai_probability >= AI_HIGH_THRESHOLD:
        return "높음", "위조 의심"
    if ai_probability >= AI_LOW_THRESHOLD:
        return "중간", "불확실"
    return "낮음", "진본 의심"


def _interpret_video_probability(ai_probability: float) -> tuple[str, str]:
    if ai_probability >= VIDEO_AI_HIGH_THRESHOLD:
        return "높음", "위조 의심"
    if ai_probability >= VIDEO_AI_LOW_THRESHOLD:
        return "중간", "불확실"
    return "낮음", "진본 의심"


def _unavailable_result(notes: str) -> AIDetectionResult:
    return AIDetectionResult(
        ai_probability=None,
        face_synthesis_level=None,
        frame_inconsistency=None,
        lighting_inconsistency=None,
        verdict=None,
        ai_score=None,
        notes=notes + " ai_score=None으로 레이어 6에서 제외 처리.",
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))
