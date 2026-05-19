"""정지 이미지용 얼굴 조작/AI 생성 detector 앙상블.

Organika 같은 diffusion 이미지 detector는 전체 생성 이미지에는 강하지만,
RedFace의 face swap / attribute manipulation / reenactment류에는 약할 수 있다.
이 모듈은 얼굴 crop 기반 detector feature와 로컬 calibration을 결합한다.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from layers.deepfake_video_detectors import predict_video_detector_suite
from layers.fingerprint import run as fingerprint_run
from layers.video_utils import extract_face_crop_bgr


CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "models" / "image_calibration.json"
DEFAULT_FEATURE_ORDER = [
    "diffusion_detector",
    "general_aigc",
    "fingerprint",
    "face_ensemble_raw",
    "faceforge_xception",
    "deepfakebench_efficientnet_b4",
    "deepfakebench_resnet18",
    "deepfakebench_r3d18",
    "legacy_efficientnet_b0",
]

CALIBRATION_CACHE: Optional[dict] = None
CALIBRATION_MTIME: Optional[float] = None
CALIBRATION_CACHE_PATH: Optional[Path] = None


@dataclass
class ImageDetectorSuiteResult:
    probability: float
    raw_probability: float
    features: dict[str, float]
    calibration_note: str
    face_detected: bool
    face_detector: str
    notes: str


def predict_image_detector_suite(
    path: str | Path,
    diffusion_probability: Optional[float] = None,
    general_aigc_probability: Optional[float] = None,
) -> ImageDetectorSuiteResult:
    image_path = Path(path)
    face_image, detected, detector = _load_face_image(image_path)
    repeated_faces = [face_image] * 16
    face_suite = predict_video_detector_suite(repeated_faces, face_ratio=1.0 if detected else 0.0)

    fingerprint_score = _safe_fingerprint_score(image_path)
    features = {
        "diffusion_detector": _coalesce(diffusion_probability, 0.5),
        "general_aigc": _coalesce(general_aigc_probability, 0.5),
        "fingerprint": _coalesce(fingerprint_score, 0.5),
        "face_ensemble_raw": face_suite.raw_probability,
    }
    features.update(face_suite.feature_map)

    raw_probability = _default_raw_probability(features)
    probability, calibration_note = _calibrate_probability(raw_probability, features)

    model_summary = ", ".join(
        f"{name}={features.get(name, 0.0):.3f}"
        for name in DEFAULT_FEATURE_ORDER
        if name in features
    )

    return ImageDetectorSuiteResult(
        probability=probability,
        raw_probability=raw_probability,
        features=features,
        calibration_note=calibration_note,
        face_detected=detected,
        face_detector=detector,
        notes=(
            f"image face-forgery ensemble. raw={raw_probability:.3f}, calibrated={probability:.3f}, "
            f"face_detected={detected}, detector={detector}, features=[{model_summary}], "
            f"calibration={calibration_note}"
        ),
    )


def _load_face_image(path: Path) -> tuple[Image.Image, bool, str]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("opencv-python이 필요합니다.") from exc

    with Image.open(path) as img:
        rgb = img.convert("RGB")
    frame_rgb = np.asarray(rgb)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    crop_bgr, detected, detector = extract_face_crop_bgr(frame_bgr)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb), detected, detector


def _safe_fingerprint_score(path: Path) -> Optional[float]:
    try:
        return fingerprint_run(str(path)).ai_score
    except Exception:
        return None


def _default_raw_probability(features: dict[str, float]) -> float:
    return _clamp01(
        0.30 * features.get("diffusion_detector", 0.5)
        + 0.24 * features.get("general_aigc", 0.5)
        + 0.16 * features.get("fingerprint", 0.5)
        + 0.14 * features.get("face_ensemble_raw", 0.5)
        + 0.10 * features.get("deepfakebench_efficientnet_b4", 0.5)
        + 0.06 * features.get("faceforge_xception", 0.5)
    )


def _calibrate_probability(raw_probability: float, features: dict[str, float]) -> tuple[float, str]:
    calibration_path = _active_calibration_path()
    calibration = _load_calibration()
    if calibration:
        method = calibration.get("method")
        if method == "logistic":
            feature_order = calibration.get("feature_order", [])
            coefficients = calibration.get("coefficients", [])
            if len(feature_order) == len(coefficients):
                z = float(calibration.get("intercept", 0.0))
                for name, coefficient in zip(feature_order, coefficients):
                    z += float(coefficient) * float(features.get(name, 0.5))
                probability = _sigmoid(z)
                probability, suffix = _blend_uncalibrated_general_aigc(probability, features, feature_order)
                return probability, f"local logistic calibration: {calibration_path.name}{suffix}"
        elif method == "threshold_shift":
            threshold = float(calibration.get("decision_threshold", 0.5))
            temperature = float(calibration.get("temperature", 0.75))
            return _threshold_shift(raw_probability, threshold, temperature), (
                f"local threshold calibration threshold={threshold:.3f}"
            )

    return raw_probability, "default uncalibrated weighted ensemble"


def _blend_uncalibrated_general_aigc(
    probability: float,
    features: dict[str, float],
    feature_order: list[str],
) -> tuple[float, str]:
    """기존 calibration 파일이 GenImage feature를 모를 때 약하게 post-blend한다."""
    if "general_aigc" in feature_order:
        return probability, ""
    general = features.get("general_aigc", 0.5)
    if abs(general - 0.5) < 0.02:
        return probability, ""
    blended = _clamp01(0.82 * probability + 0.18 * general)
    return blended, " + general_aigc post-blend"


def _load_calibration() -> Optional[dict]:
    global CALIBRATION_CACHE, CALIBRATION_MTIME, CALIBRATION_CACHE_PATH
    calibration_path = _active_calibration_path()
    if not calibration_path.exists():
        CALIBRATION_CACHE = {}
        CALIBRATION_MTIME = None
        CALIBRATION_CACHE_PATH = calibration_path
        return None

    mtime = calibration_path.stat().st_mtime
    if (
        CALIBRATION_CACHE is not None
        and CALIBRATION_MTIME == mtime
        and CALIBRATION_CACHE_PATH == calibration_path
    ):
        return CALIBRATION_CACHE

    try:
        CALIBRATION_CACHE = json.loads(calibration_path.read_text(encoding="utf-8"))
        CALIBRATION_MTIME = mtime
        CALIBRATION_CACHE_PATH = calibration_path
    except Exception:
        CALIBRATION_CACHE = {}
        CALIBRATION_MTIME = mtime
        CALIBRATION_CACHE_PATH = calibration_path
        return None
    return CALIBRATION_CACHE


def _active_calibration_path() -> Path:
    override = os.environ.get("CAVE_IMAGE_CALIBRATION", "").strip()
    if override:
        return Path(override).expanduser()
    return CALIBRATION_PATH


def _threshold_shift(raw_probability: float, threshold: float, temperature: float) -> float:
    raw_logit = _logit(_clamp01(raw_probability))
    threshold_logit = _logit(_clamp01(threshold))
    return _sigmoid((raw_logit - threshold_logit) / max(temperature, 1e-6))


def _logit(value: float) -> float:
    value = min(max(value, 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _coalesce(value: Optional[float], default: float) -> float:
    if value is None:
        return default
    return _clamp01(value)


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))
