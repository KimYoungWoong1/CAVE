"""생체·생리적 신호 일관성 검사.

영상 (8 s+):
  CHROM rPPG 방법(de Haan & Jeanne, IEEE TBME 2013) 심박 대역 에너지 분석.
  조명 공통 성분을 크로미넌스 투영으로 제거하여 순수 심박 신호 추출.
  딥페이크는 심박 리듬 신호가 약하거나 비자연적 패턴을 보임.

정지 이미지는 시간축이 없으므로 이 모듈에서 제외한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from layers.video_utils import extract_face_crop_bgr


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MIN_DURATION_SEC = 8.0
TARGET_FPS = 12.5
MIN_SIGNAL_SAMPLES = 80
HEART_LOW_HZ = 0.7
HEART_HIGH_HZ = 3.0
RPPG_CLASSIFIER_PATH = Path(__file__).resolve().parents[1] / "models" / "rppg_classifier.joblib"
RPPG_THRESHOLD_TARGET = 0.60
RPPG_FEATURE_ORDER = [
    "heuristic_ai_score",
    "naturalness",
    "band_power_ratio",
    "peak_prominence",
    "temporal_stability",
    "peak_bpm_norm",
    "bpm_plausibility",
    "sample_rate_norm",
    "duration_norm",
    "frame_count_norm",
    "face_detected_ratio",
]

_CLASSIFIER_CACHE: Optional[dict] = None
_CLASSIFIER_MTIME: Optional[float] = None

@dataclass
class RPPGResult:
    heartbeat_naturalness: Optional[float]  # 0~1 (1=자연, 0=비자연)
    biometric_match: Optional[bool]
    ai_score: Optional[float]               # None=분석 불가 (레이어 6 제외)
    notes: str
    analysis_mode: str = "none"             # "image" | "video" | "none"
    evidence: dict[str, str] = field(default_factory=dict)


@dataclass
class LearnedRPPGPrediction:
    probability: float
    calibrated_probability: float
    confidence: float
    threshold: Optional[float]
    test_auc: Optional[float]
    best_accuracy: Optional[float]


def run(file_path: str) -> RPPGResult:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        return RPPGResult(
            heartbeat_naturalness=None,
            biometric_match=None,
            ai_score=None,
            notes="정지 이미지 입력은 시간축이 없어 rPPG 생체 신호 분석 불가. 다층 감사에서 제외.",
            analysis_mode="none",
        )
    return _analyze_video(path)


# ── 영상: CHROM rPPG 분석 ────────────────────────────────────────────────────

def _analyze_video(path: Path) -> RPPGResult:
    """CHROM 방법(de Haan & Jeanne, IEEE TBME 2013) 기반 rPPG 심박 신호 분석."""
    try:
        features, meta = extract_rppg_features(path)
        naturalness = float(features["naturalness"])
        metrics = meta["metrics"]
        learned = _predict_learned(features)
    except ImportError:
        return RPPGResult(
            heartbeat_naturalness=None,
            biometric_match=None,
            ai_score=None,
            notes="opencv-python 미설치 — 영상 rPPG 분석 불가.",
            analysis_mode="none",
        )
    except Exception as exc:
        return RPPGResult(
            heartbeat_naturalness=None,
            biometric_match=None,
            ai_score=None,
            notes=f"rPPG 분석 실패: {type(exc).__name__}: {exc}",
            analysis_mode="none",
        )

    heuristic_ai_score = round(float(features["heuristic_ai_score"]), 3)
    if learned is not None:
        ai_score = round(_clamp01(0.85 * learned.calibrated_probability + 0.15 * heuristic_ai_score), 3)
        notes_prefix = (
            "학습 기반 rPPG feature classifier 분석. "
            f"learned_prob={learned.probability:.3f}, "
            f"calibrated_prob={learned.calibrated_probability:.3f}, "
            f"threshold={_fmt_optional(learned.threshold)}, "
            f"heuristic={heuristic_ai_score:.3f}, final={ai_score:.3f}, "
            f"test_auc={_fmt_optional(learned.test_auc)}, best_acc={_fmt_optional(learned.best_accuracy)}. "
        )
        evidence = {
            "learned_prob": f"{learned.probability:.3f}",
            "calibrated_prob": f"{learned.calibrated_probability:.3f}",
            "threshold": _fmt_optional(learned.threshold),
            "heuristic": f"{heuristic_ai_score:.3f}",
            "test_auc": _fmt_optional(learned.test_auc),
            "best_acc": _fmt_optional(learned.best_accuracy),
            "face_detected": f"{int(meta['face_count'])}/{int(meta['n_frames'])}",
            "peak_bpm": f"{metrics['peak_bpm']:.1f}",
        }
    else:
        ai_score = heuristic_ai_score
        notes_prefix = "CHROM rPPG 휴리스틱 분석 (학습 classifier 없음 또는 사용 불가). "
        evidence = {
            "heuristic": f"{heuristic_ai_score:.3f}",
            "face_detected": f"{int(meta['face_count'])}/{int(meta['n_frames'])}",
            "peak_bpm": f"{metrics['peak_bpm']:.1f}",
        }

    return RPPGResult(
        heartbeat_naturalness=naturalness,
        biometric_match=naturalness >= 0.45,
        ai_score=ai_score,
        notes=(
            notes_prefix +
            f"CHROM rPPG 분석 (영상). "
            f"duration={float(meta['duration']):.2f}s, frames={int(meta['n_frames'])}, "
            f"fps={float(meta['sample_rate']):.2f}, "
            f"peak_bpm={metrics['peak_bpm']:.1f}, "
            f"band_power_ratio={metrics['band_power_ratio']:.3f}, "
            f"peak_prominence={metrics['peak_prominence']:.3f}, "
            f"temporal_stability={metrics['temporal_stability']:.3f}. "
            f"face_detected={int(meta['face_count'])}/{int(meta['n_frames'])}, "
            f"detectors={meta['detector_counts']}. "
            "MediaPipe/MTCNN 기반 얼굴 ROI의 피부 후보 영역에서 추출한 보조 생체 신호."
        ),
        analysis_mode="video",
        evidence=evidence,
    )


def extract_rppg_features(path: str | Path) -> tuple[dict[str, float], dict[str, object]]:
    """영상에서 CHROM rPPG feature와 학습용 메타데이터를 추출한다."""
    signal, sample_rate, duration, n_frames, face_count, detector_counts = _extract_chrom_signal(Path(path))
    naturalness, metrics = _analyze_signal(signal, sample_rate)
    heuristic_ai_score = round(1.0 - naturalness, 3)
    face_detected_ratio = face_count / max(n_frames, 1)
    peak_bpm = float(metrics["peak_bpm"])
    bpm_plausibility = 1.0 if 45.0 <= peak_bpm <= 180.0 else 0.0

    features = {
        "heuristic_ai_score": _clamp01(heuristic_ai_score),
        "naturalness": _clamp01(naturalness),
        "band_power_ratio": _clamp01(float(metrics["band_power_ratio"])),
        "peak_prominence": _clamp01(float(metrics["peak_prominence"])),
        "temporal_stability": _clamp01(float(metrics["temporal_stability"])),
        "peak_bpm_norm": _clamp01(peak_bpm / 220.0),
        "bpm_plausibility": bpm_plausibility,
        "sample_rate_norm": _clamp01(sample_rate / 30.0),
        "duration_norm": _clamp01(duration / 30.0),
        "frame_count_norm": _clamp01(n_frames / 400.0),
        "face_detected_ratio": _clamp01(face_detected_ratio),
    }
    meta: dict[str, object] = {
        "signal": signal,
        "sample_rate": sample_rate,
        "duration": duration,
        "n_frames": n_frames,
        "face_count": face_count,
        "detector_counts": detector_counts,
        "metrics": metrics,
    }
    return features, meta


def _extract_chrom_signal(path: Path) -> tuple[np.ndarray, float, float, int, int, str]:
    """RGB 채널 크로미넌스 투영(CHROM)으로 rPPG 원시 신호 추출.

    조명 공통 성분(DC)을 제거하고 심박 대역 AC 신호만 추출.
    """
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("영상을 열 수 없습니다.")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        cap.release()
        raise ValueError("FPS 정보를 읽을 수 없습니다.")

    duration = frame_count / fps if frame_count else 0.0
    if duration < MIN_DURATION_SEC:
        cap.release()
        raise ValueError(
            f"rPPG 분석에 최소 {MIN_DURATION_SEC:.0f}초 영상 필요 (현재 {duration:.2f}초)."
        )

    step = max(1, round(fps / TARGET_FPS))
    sample_rate = fps / step

    R_list: list[float] = []
    G_list: list[float] = []
    B_list: list[float] = []
    face_count = 0
    detector_names: list[str] = []

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            roi, detected, detector = extract_face_crop_bgr(frame)
            if detected:
                face_count += 1
            detector_names.append(detector)
            skin = _skin_mask_bgr(roi)
            if skin.sum() < 50:
                skin = np.ones(roi.shape[:2], dtype=bool)
            R_list.append(float(np.mean(roi[:, :, 2][skin])))
            G_list.append(float(np.mean(roi[:, :, 1][skin])))
            B_list.append(float(np.mean(roi[:, :, 0][skin])))
        idx += 1

    cap.release()

    R = np.asarray(R_list, dtype=np.float64)
    G = np.asarray(G_list, dtype=np.float64)
    B = np.asarray(B_list, dtype=np.float64)

    if len(R) < MIN_SIGNAL_SAMPLES:
        raise ValueError(f"rPPG 샘플 부족: {len(R)}개")

    # CHROM 투영: 조명 공통 성분 제거 후 크로미넌스 심박 신호 추출
    Rn = R / (np.mean(R) + 1e-8) - 1.0
    Gn = G / (np.mean(G) + 1e-8) - 1.0
    Bn = B / (np.mean(B) + 1e-8) - 1.0

    X = 3.0 * Rn - 2.0 * Gn
    Y = 1.5 * Rn + Gn - 1.5 * Bn

    alpha = (np.std(X) + 1e-8) / (np.std(Y) + 1e-8)
    signal = X - alpha * Y

    return signal, sample_rate, duration, len(signal), face_count, _detector_counts(detector_names)


def _detector_counts(detectors: list[str]) -> str:
    counts: dict[str, int] = {}
    for detector in detectors:
        counts[detector] = counts.get(detector, 0) + 1
    return ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def _skin_mask_bgr(roi_bgr: np.ndarray) -> np.ndarray:
    """BGR 프레임에서 피부 후보 마스크."""
    b = roi_bgr[:, :, 0].astype(np.float32)
    g = roi_bgr[:, :, 1].astype(np.float32)
    r = roi_bgr[:, :, 2].astype(np.float32)
    return (
        (r > 45) & (g > 35) & (b > 20)
        & (r >= g * 0.75) & (g >= b * 0.75)
        & (r > b)
    )


# ── 신호 분석 공통 (영상) ────────────────────────────────────────────────────

def _analyze_signal(signal: np.ndarray, sample_rate: float) -> tuple[float, dict]:
    cleaned = _preprocess_signal(signal)
    freqs = np.fft.rfftfreq(len(cleaned), d=1.0 / sample_rate)
    spectrum = np.abs(np.fft.rfft(cleaned * np.hanning(len(cleaned)))) ** 2

    valid = (freqs >= 0.2) & (freqs <= 4.0)
    heart = (freqs >= HEART_LOW_HZ) & (freqs <= HEART_HIGH_HZ)
    if not np.any(heart):
        raise ValueError("심박 대역 FFT 해상도 부족")

    total_power = float(np.sum(spectrum[valid])) + 1e-8
    heart_power = float(np.sum(spectrum[heart]))
    band_power_ratio = _clamp01(heart_power / total_power)

    heart_spectrum = spectrum[heart]
    heart_freqs = freqs[heart]
    peak_idx = int(np.argmax(heart_spectrum))
    peak_bpm = float(heart_freqs[peak_idx] * 60.0)
    peak_prominence = _clamp01(
        (float(heart_spectrum[peak_idx]) / (float(np.mean(heart_spectrum)) + 1e-8) - 1.0) / 8.0
    )

    temporal_stability = _temporal_stability(cleaned, sample_rate, peak_bpm / 60.0)

    naturalness = round(
        _clamp01(
            0.42 * _smoothstep(band_power_ratio, 0.18, 0.55)
            + 0.38 * peak_prominence
            + 0.20 * temporal_stability
        ),
        3,
    )
    return naturalness, {
        "band_power_ratio": band_power_ratio,
        "peak_prominence": peak_prominence,
        "peak_bpm": peak_bpm,
        "temporal_stability": temporal_stability,
    }


def _predict_learned(features: dict[str, float]) -> Optional[LearnedRPPGPrediction]:
    bundle = _load_classifier()
    if not bundle:
        return None

    feature_order = bundle.get("feature_order", RPPG_FEATURE_ORDER)
    vector = np.asarray([[float(features.get(name, 0.0)) for name in feature_order]], dtype=np.float64)
    model = bundle.get("model")
    if model is None or not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(vector)[0]
    classes = list(getattr(model, "classes_", [0, 1]))
    fake_idx = classes.index(1) if 1 in classes else int(np.argmax(classes))
    probability = _clamp01(float(probabilities[fake_idx]))
    meta = bundle.get("meta", {})
    if bool(meta.get("invert_probability", False)):
        probability = _clamp01(1.0 - probability)
    threshold = _as_optional_float(meta.get("best_threshold"))
    calibrated_probability = _threshold_calibrated_probability(
        probability,
        threshold,
        target=RPPG_THRESHOLD_TARGET,
    )
    return LearnedRPPGPrediction(
        probability=probability,
        calibrated_probability=calibrated_probability,
        confidence=_clamp01(abs(probability - 0.5) * 2.0),
        threshold=threshold,
        test_auc=_as_optional_float(meta.get("test_auc")),
        best_accuracy=_as_optional_float(meta.get("best_accuracy")),
    )


def _load_classifier() -> Optional[dict]:
    global _CLASSIFIER_CACHE, _CLASSIFIER_MTIME
    if not RPPG_CLASSIFIER_PATH.exists():
        _CLASSIFIER_CACHE = None
        _CLASSIFIER_MTIME = None
        return None

    mtime = RPPG_CLASSIFIER_PATH.stat().st_mtime
    if _CLASSIFIER_CACHE is not None and _CLASSIFIER_MTIME == mtime:
        return _CLASSIFIER_CACHE

    try:
        import joblib  # type: ignore

        bundle = joblib.load(RPPG_CLASSIFIER_PATH)
        if not isinstance(bundle, dict):
            return None
        _CLASSIFIER_CACHE = bundle
        _CLASSIFIER_MTIME = mtime
    except Exception:
        _CLASSIFIER_CACHE = None
        _CLASSIFIER_MTIME = mtime
        return None
    return _CLASSIFIER_CACHE


def rppg_feature_vector(features: dict[str, float], feature_order: list[str] | None = None) -> list[float]:
    """rPPG 학습/추론 스크립트용 고정 순서 feature vector."""
    order = feature_order or RPPG_FEATURE_ORDER
    return [float(features.get(name, 0.0)) for name in order]


def _preprocess_signal(signal: np.ndarray) -> np.ndarray:
    centered = signal - np.mean(signal)
    trend = _moving_average(centered, max(5, len(centered) // 12))
    detrended = centered - trend
    std = float(np.std(detrended))
    if std < 1e-8:
        raise ValueError("rPPG 신호 변화가 너무 작습니다.")
    return detrended / std


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(3, min(window, len(values) // 2))
    return np.convolve(values, np.ones(window) / window, mode="same")


def _temporal_stability(signal: np.ndarray, sample_rate: float, peak_hz: float) -> float:
    if peak_hz <= 0:
        return 0.0
    window = int(sample_rate * 4)
    if window < 16 or len(signal) < window * 2:
        return 0.5
    powers = []
    for start in range(0, len(signal) - window + 1, max(1, window // 2)):
        seg = signal[start:start + window]
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / sample_rate)
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        band = (
            (freqs >= max(HEART_LOW_HZ, peak_hz - 0.25))
            & (freqs <= min(HEART_HIGH_HZ, peak_hz + 0.25))
        )
        powers.append(float(np.sum(spec[band])))
    arr = np.asarray(powers, dtype=np.float64)
    if len(arr) < 2 or np.mean(arr) <= 1e-8:
        return 0.5
    return _clamp01(1.0 - float(np.std(arr) / (np.mean(arr) + 1e-8)))


# ── 공통 유틸 ────────────────────────────────────────────────────────────────

def _smoothstep(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    x = _clamp01((value - low) / (high - low))
    return x * x * (3.0 - 2.0 * x)


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _as_optional_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _fmt_optional(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _threshold_calibrated_probability(
    probability: float,
    threshold: Optional[float],
    target: float,
) -> float:
    probability = _clamp01(probability)
    if threshold is None or threshold <= 0.0 or threshold >= 1.0:
        return probability
    target = _clamp01(target)
    if probability < threshold:
        return _clamp01((probability / threshold) * target)
    return _clamp01(target + ((probability - threshold) / (1.0 - threshold)) * (1.0 - target))
