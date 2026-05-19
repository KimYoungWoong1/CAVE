"""레이어 5: 생성 모델 핑거프린트 추정.

학습된 fingerprint classifier가 있으면 FFT/잔차/노이즈 feature를 입력으로
AI 생성·조작 확률과 조작 방식을 추정한다. 이미지와 영상은 별도 모델을 사용하며,
모델 파일이 없거나 실패하면 기존 신호처리 휴리스틱으로 fallback한다.

주의: 특정 생성 모델명을 단정하지 않는 보조 지표이며, 법적 판단의
단독 근거가 아니라 레이어 1/3/6과 함께 해석해야 한다.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

from layers.video_utils import is_video_file, sample_video_face_crops


AI_HIGH = 0.65
AI_MID = 0.40
VIDEO_AI_HIGH = 0.60
VIDEO_AI_MID = 0.45
VIDEO_THRESHOLD_TARGET = 0.60
MAX_ANALYSIS_SIZE = 512
VIDEO_SAMPLE_FRAMES = 12
CLASSIFIER_PATH = Path(__file__).resolve().parents[1] / "models" / "fingerprint_classifier.joblib"
VIDEO_CLASSIFIER_PATH = Path(__file__).resolve().parents[1] / "models" / "video_fingerprint_classifier.joblib"
FINGERPRINT_FEATURE_ORDER = [
    "high_freq_ratio",
    "mid_freq_ratio",
    "low_freq_ratio",
    "spectral_flatness",
    "spectral_centroid",
    "spectral_rolloff",
    "channel_noise_corr",
    "block_periodicity",
    "residual_strength",
    "residual_mean_abs",
    "residual_skewness",
    "residual_kurtosis",
    "gray_entropy",
    "gradient_energy",
    "color_std_mean",
    "saturation_mean",
    "saturation_std",
]
VIDEO_FEATURE_STATS = ("mean", "std", "p10", "p90")
VIDEO_FINGERPRINT_FEATURE_ORDER = [
    f"{name}_{stat}"
    for name in FINGERPRINT_FEATURE_ORDER
    for stat in VIDEO_FEATURE_STATS
] + [
    "frame_score_mean",
    "frame_score_std",
    "frame_score_p90",
    "frame_score_max",
    "temporal_feature_delta",
    "temporal_score_delta",
    "face_detected_ratio",
]

_METHOD_LABELS = {
    "EFS": "entire-face-synthesis",
    "FAM": "face-attribute-manipulation",
    "FR": "face-reenactment",
    "FS": "face-swap",
}
_VIDEO_METHOD_LABELS = {
    "DeepFakeDetection": "dataset-specific-deepfake",
    "Deepfakes": "autoencoder-face-swap",
    "Face2Face": "face-reenactment",
    "FaceShifter": "face-swap",
    "FaceSwap": "face-swap",
    "NeuralTextures": "neural-texture-rendering",
}

_CLASSIFIER_CACHE: Optional[dict] = None
_CLASSIFIER_MTIME: Optional[float] = None
_VIDEO_CLASSIFIER_CACHE: Optional[dict] = None
_VIDEO_CLASSIFIER_MTIME: Optional[float] = None


@dataclass
class FingerprintResult:
    ai_likelihood: Optional[str]       # '높음' | '중간' | '낮음' | None
    generation_method: Optional[str]   # 'face-swap' | 'diffusion' | 'GAN' | None
    model_family: Optional[str]
    confidence: float
    ai_score: Optional[float]
    notes: str
    evidence: dict[str, str] = field(default_factory=dict)


@dataclass
class LearnedFingerprintPrediction:
    probability: float
    method: str
    confidence: float
    source: str
    test_auc: Optional[float]
    threshold: Optional[float] = None
    calibrated_probability: Optional[float] = None
    best_accuracy: Optional[float] = None


def run(file_path: str) -> FingerprintResult:
    """이미지 또는 영상 프레임 기반 생성 흔적 분석."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    if is_video_file(path):
        return _run_video(path)
    return _run_image(path)


def _run_image(path: Path) -> FingerprintResult:
    try:
        rgb = _load_rgb(path)
        features = _extract_features(rgb)
        heuristic_score = _score_features(features)
        learned = _predict_learned(features)
    except Exception as exc:
        return FingerprintResult(
            ai_likelihood=None,
            generation_method=None,
            model_family=None,
            confidence=0.0,
            ai_score=None,
            notes=(
                f"핑거프린트 분석 실패: {type(exc).__name__}: {exc}. "
                "ai_score=None으로 레이어 6에서 제외 처리."
            ),
        )

    if learned is not None:
        ai_score = round(_clamp01(0.85 * learned.probability + 0.15 * heuristic_score), 3)
        likelihood = _likelihood_label(ai_score)
        generation_method = learned.method if ai_score >= AI_MID else "unknown"
        model_family = "learned fingerprint classifier" if ai_score >= AI_MID else "unknown"
        confidence = round(_clamp01(0.70 * learned.confidence + 0.30 * _confidence(ai_score, features)), 3)
        return FingerprintResult(
            ai_likelihood=likelihood,
            generation_method=generation_method,
            model_family=model_family,
            confidence=confidence,
            ai_score=ai_score,
            notes=(
                "학습 기반 fingerprint classifier 분석. "
                f"learned_prob={learned.probability:.3f}, heuristic={heuristic_score:.3f}, "
                f"final={ai_score:.3f}, method={learned.method}, "
                f"source={learned.source}, test_auc={_fmt_optional(learned.test_auc)}. "
                f"high_freq={features['high_freq_ratio']:.3f}, "
                f"spectral_flatness={features['spectral_flatness']:.3f}, "
                f"channel_noise_corr={features['channel_noise_corr']:.3f}, "
                f"residual_strength={features['residual_strength']:.3f}. "
                "특정 생성 모델명이 아닌 RedFace-style 조작 계열 attribution."
            ),
            evidence={
                "learned_prob": f"{learned.probability:.3f}",
                "heuristic": f"{heuristic_score:.3f}",
                "final": f"{ai_score:.3f}",
                "test_auc": _fmt_optional(learned.test_auc),
            },
        )

    ai_score = heuristic_score
    likelihood = _likelihood_label(ai_score)
    generation_method = "diffusion" if ai_score >= AI_MID else "unknown"
    model_family = "diffusion-like" if ai_score >= AI_MID else "unknown"
    confidence = _confidence(ai_score, features)

    return FingerprintResult(
        ai_likelihood=likelihood,
        generation_method=generation_method,
        model_family=model_family,
        confidence=confidence,
        ai_score=ai_score,
        notes=(
            "FFT/노이즈 기반 보조 분석 (학습 classifier 없음 또는 사용 불가). "
            f"high_freq={features['high_freq_ratio']:.3f}, "
            f"spectral_flatness={features['spectral_flatness']:.3f}, "
            f"channel_noise_corr={features['channel_noise_corr']:.3f}, "
            f"block_periodicity={features['block_periodicity']:.3f}. "
            "특정 모델명 단정이 아닌 diffusion-like 생성 흔적 추정값."
        ),
    )


def _run_video(path: Path) -> FingerprintResult:
    try:
        video_features, meta = extract_video_fingerprint_features(path, num_frames=VIDEO_SAMPLE_FRAMES)
        face_count = int(meta["face_count"])
        sample_count = int(meta["sample_count"])
        detector_counts = str(meta["detector_counts"])
        frame_scores = list(meta["frame_scores"])
        frame_features = list(meta["frame_features"])
    except Exception as exc:
        return FingerprintResult(
            ai_likelihood=None,
            generation_method=None,
            model_family=None,
            confidence=0.0,
            ai_score=None,
            notes=(
                f"영상 핑거프린트 분석 실패: {type(exc).__name__}: {exc}. "
                "ai_score=None으로 다층 감사에서 제외 처리."
            ),
        )

    scores = np.asarray(frame_scores, dtype=np.float64)
    mean_score = float(np.mean(scores))
    p90_score = float(np.percentile(scores, 90))
    max_score = float(np.max(scores))
    heuristic_ai_score = round(_clamp01(0.65 * mean_score + 0.25 * p90_score + 0.10 * max_score), 3)
    avg_features = _average_features(frame_features)
    learned = _predict_video_learned(video_features)

    if learned is not None:
        learned_score = learned.calibrated_probability or learned.probability
        ai_score = round(_clamp01(0.95 * learned_score + 0.05 * heuristic_ai_score), 3)
        likelihood = _video_likelihood_label(ai_score)
        generation_method = learned.method if ai_score >= VIDEO_AI_MID else "unknown"
        model_family = "learned video fingerprint classifier" if ai_score >= VIDEO_AI_MID else "unknown"
        confidence = round(_clamp01(0.70 * learned.confidence + 0.30 * _confidence(ai_score, avg_features)), 3)
        return FingerprintResult(
            ai_likelihood=likelihood,
            generation_method=generation_method,
            model_family=model_family,
            confidence=confidence,
            ai_score=ai_score,
            notes=(
                f"학습 기반 video fingerprint classifier 분석. "
                f"learned_prob={learned.probability:.3f}, "
                f"calibrated_prob={learned_score:.3f}, "
                f"threshold={_fmt_optional(learned.threshold)}, "
                f"heuristic={heuristic_ai_score:.3f}, "
                f"final={ai_score:.3f}, method={learned.method}, "
                f"source={learned.source}, test_auc={_fmt_optional(learned.test_auc)}, "
                f"best_acc={_fmt_optional(learned.best_accuracy)}. "
                f"face_detected={face_count}/{sample_count}, detectors={detector_counts}. "
                f"frame_mean={mean_score:.3f}, p90={p90_score:.3f}, max={max_score:.3f}, "
                f"std={float(np.std(scores)):.3f}, temporal_delta={video_features['temporal_feature_delta']:.3f}. "
                f"avg_high_freq={avg_features['high_freq_ratio']:.3f}, "
                f"avg_spectral_flatness={avg_features['spectral_flatness']:.3f}, "
                f"avg_channel_noise_corr={avg_features['channel_noise_corr']:.3f}. "
                "FFPP C23 얼굴 crop 시계열 기반 video-level attribution."
            ),
            evidence={
                "learned_prob": f"{learned.probability:.3f}",
                "calibrated_prob": f"{learned_score:.3f}",
                "threshold": _fmt_optional(learned.threshold),
                "heuristic": f"{heuristic_ai_score:.3f}",
                "final": f"{ai_score:.3f}",
                "test_auc": _fmt_optional(learned.test_auc),
                "best_acc": _fmt_optional(learned.best_accuracy),
                "face_detected": f"{face_count}/{sample_count}",
                "temporal_delta": f"{video_features['temporal_feature_delta']:.3f}",
                "frame_p90": f"{p90_score:.3f}",
            },
        )

    generation_method = "diffusion/face-swap 후보" if heuristic_ai_score >= AI_MID else "unknown"
    model_family = "frame-level synthetic-like" if heuristic_ai_score >= AI_MID else "unknown"
    confidence = _confidence(heuristic_ai_score, avg_features)
    ai_score = heuristic_ai_score
    likelihood = _likelihood_label(ai_score)

    return FingerprintResult(
        ai_likelihood=likelihood,
        generation_method=generation_method,
        model_family=model_family,
        confidence=confidence,
        ai_score=ai_score,
        notes=(
            f"영상 대표 얼굴 crop {len(scores)}장에 fingerprint 분석 적용. "
            "영상용 learned fingerprint classifier 없음 또는 사용 불가로 heuristic 적용. "
            f"mean={mean_score:.3f}, p90={p90_score:.3f}, max={max_score:.3f}, std={float(np.std(scores)):.3f}. "
            f"face_detected={face_count}/{len(scores)}, detectors={detector_counts}. "
            f"avg_high_freq={avg_features['high_freq_ratio']:.3f}, "
            f"avg_spectral_flatness={avg_features['spectral_flatness']:.3f}, "
            f"avg_channel_noise_corr={avg_features['channel_noise_corr']:.3f}. "
            "영상 압축으로 인한 채널 상관/스펙트럼 고점수를 보정한 heuristic 점수."
        ),
    )


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return _pil_to_rgb_array(img)


def _pil_to_rgb_array(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img.thumbnail((MAX_ANALYSIS_SIZE, MAX_ANALYSIS_SIZE))
    return np.asarray(img, dtype=np.float32) / 255.0


def _average_features(features: list[dict[str, float]]) -> dict[str, float]:
    if not features:
        return {
            "high_freq_ratio": 0.0,
            "spectral_flatness": 0.0,
            "channel_noise_corr": 0.0,
            "block_periodicity": 0.0,
            "residual_strength": 0.0,
        }
    keys = features[0].keys()
    return {key: float(np.mean([item[key] for item in features])) for key in keys}


def _detector_counts(detectors) -> str:
    counts: dict[str, int] = {}
    for detector in detectors:
        counts[str(detector)] = counts.get(str(detector), 0) + 1
    return ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def extract_video_fingerprint_features(
    path: str | Path,
    num_frames: int = VIDEO_SAMPLE_FRAMES,
) -> tuple[dict[str, float], dict[str, object]]:
    """영상 얼굴 crop 시계열에서 video-level fingerprint feature를 추출한다."""
    samples = sample_video_face_crops(path, num_frames=num_frames)
    frame_scores: list[float] = []
    frame_features: list[dict[str, float]] = []

    for sample in samples:
        rgb = _pil_to_rgb_array(sample.image)
        features = _extract_features(rgb)
        frame_features.append(features)
        frame_scores.append(_score_video_features(features))

    face_count = sum(1 for sample in samples if sample.detected)
    video_features = _aggregate_video_features(
        frame_features=frame_features,
        frame_scores=frame_scores,
        face_detected_ratio=face_count / max(len(samples), 1),
    )
    meta: dict[str, object] = {
        "sample_count": len(samples),
        "face_count": face_count,
        "detector_counts": _detector_counts(sample.detector for sample in samples),
        "frame_scores": frame_scores,
        "frame_features": frame_features,
    }
    return video_features, meta


def _aggregate_video_features(
    frame_features: list[dict[str, float]],
    frame_scores: list[float],
    face_detected_ratio: float,
) -> dict[str, float]:
    if not frame_features:
        raise ValueError("영상 fingerprint feature를 만들 프레임이 없습니다.")

    matrix = np.asarray(
        [[float(features.get(name, 0.0)) for name in FINGERPRINT_FEATURE_ORDER] for features in frame_features],
        dtype=np.float64,
    )
    scores = np.asarray(frame_scores, dtype=np.float64)

    result: dict[str, float] = {}
    stat_values = {
        "mean": np.mean(matrix, axis=0),
        "std": np.std(matrix, axis=0),
        "p10": np.percentile(matrix, 10, axis=0),
        "p90": np.percentile(matrix, 90, axis=0),
    }
    for idx, name in enumerate(FINGERPRINT_FEATURE_ORDER):
        for stat in VIDEO_FEATURE_STATS:
            result[f"{name}_{stat}"] = _clamp01(float(stat_values[stat][idx]))

    result["frame_score_mean"] = _clamp01(float(np.mean(scores)))
    result["frame_score_std"] = _clamp01(float(np.std(scores)))
    result["frame_score_p90"] = _clamp01(float(np.percentile(scores, 90)))
    result["frame_score_max"] = _clamp01(float(np.max(scores)))
    if len(frame_features) > 1:
        result["temporal_feature_delta"] = _clamp01(float(np.mean(np.abs(np.diff(matrix, axis=0)))))
        result["temporal_score_delta"] = _clamp01(float(np.mean(np.abs(np.diff(scores)))))
    else:
        result["temporal_feature_delta"] = 0.0
        result["temporal_score_delta"] = 0.0
    result["face_detected_ratio"] = _clamp01(face_detected_ratio)
    return result


def _extract_features(rgb: np.ndarray) -> dict[str, float]:
    gray = _rgb_to_gray(rgb)
    spectrum = _fft_magnitude(gray)

    low_freq_ratio = _frequency_band_ratio(spectrum, 0.03, 0.18)
    mid_freq_ratio = _frequency_band_ratio(spectrum, 0.18, 0.38)
    high_freq_ratio = _frequency_band_ratio(spectrum, 0.38, 0.92)
    spectral_flatness = _spectral_flatness(spectrum)
    spectral_centroid = _spectral_centroid(spectrum)
    spectral_rolloff = _spectral_rolloff(spectrum)
    channel_noise_corr = _channel_noise_correlation(rgb)
    block_periodicity = _block_periodicity(gray)
    residual_strength = _residual_strength(gray)
    residual_mean_abs, residual_skewness, residual_kurtosis = _residual_stats(gray)
    gray_entropy = _gray_entropy(gray)
    gradient_energy = _gradient_energy(gray)
    color_std_mean, saturation_mean, saturation_std = _color_stats(rgb)

    return {
        "high_freq_ratio": high_freq_ratio,
        "mid_freq_ratio": mid_freq_ratio,
        "low_freq_ratio": low_freq_ratio,
        "spectral_flatness": spectral_flatness,
        "spectral_centroid": spectral_centroid,
        "spectral_rolloff": spectral_rolloff,
        "channel_noise_corr": channel_noise_corr,
        "block_periodicity": block_periodicity,
        "residual_strength": residual_strength,
        "residual_mean_abs": residual_mean_abs,
        "residual_skewness": residual_skewness,
        "residual_kurtosis": residual_kurtosis,
        "gray_entropy": gray_entropy,
        "gradient_energy": gradient_energy,
        "color_std_mean": color_std_mean,
        "saturation_mean": saturation_mean,
        "saturation_std": saturation_std,
    }


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def _fft_magnitude(gray: np.ndarray) -> np.ndarray:
    centered = gray - float(np.mean(gray))
    window_y = np.hanning(centered.shape[0])
    window_x = np.hanning(centered.shape[1])
    window = np.outer(window_y, window_x)
    fft = np.fft.fftshift(np.fft.fft2(centered * window))
    return np.abs(fft)


def _radial_mask(shape: tuple[int, int], low: float, high: float) -> np.ndarray:
    h, w = shape
    y, x = np.indices((h, w))
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    radius /= max(np.max(radius), 1e-8)
    return (radius >= low) & (radius < high)


def _high_frequency_ratio(spectrum: np.ndarray) -> float:
    return _frequency_band_ratio(spectrum, 0.38, 0.92)


def _frequency_band_ratio(spectrum: np.ndarray, low: float, high: float) -> float:
    total = float(np.sum(spectrum)) + 1e-8
    band = float(np.sum(spectrum[_radial_mask(spectrum.shape, low, high)]))
    return _clamp01(band / total)


def _spectral_flatness(spectrum: np.ndarray) -> float:
    band = spectrum[_radial_mask(spectrum.shape, 0.25, 0.85)] + 1e-8
    geometric = math.exp(float(np.mean(np.log(band))))
    arithmetic = float(np.mean(band)) + 1e-8
    return _clamp01(geometric / arithmetic)


def _spectral_centroid(spectrum: np.ndarray) -> float:
    h, w = spectrum.shape
    y, x = np.indices((h, w))
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    radius = radius / max(float(np.max(radius)), 1e-8)
    weights = spectrum + 1e-8
    return _clamp01(float(np.sum(radius * weights) / np.sum(weights)))


def _spectral_rolloff(spectrum: np.ndarray, threshold: float = 0.85) -> float:
    h, w = spectrum.shape
    y, x = np.indices((h, w))
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    radius = radius / max(float(np.max(radius)), 1e-8)
    flat_radius = radius.ravel()
    flat_power = spectrum.ravel() + 1e-8
    order = np.argsort(flat_radius)
    cumulative = np.cumsum(flat_power[order])
    cutoff = threshold * float(cumulative[-1])
    idx = int(np.searchsorted(cumulative, cutoff))
    idx = min(idx, len(order) - 1)
    return _clamp01(float(flat_radius[order[idx]]))


def _channel_noise_correlation(rgb: np.ndarray) -> float:
    residuals = []
    for idx in range(3):
        channel = Image.fromarray(np.uint8(np.clip(rgb[:, :, idx] * 255, 0, 255)))
        blurred = np.asarray(channel.filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float32) / 255.0
        residuals.append((rgb[:, :, idx] - blurred).ravel())

    corr_values = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        if np.std(residuals[a]) < 1e-8 or np.std(residuals[b]) < 1e-8:
            continue
        corr_values.append(abs(float(np.corrcoef(residuals[a], residuals[b])[0, 1])))

    if not corr_values:
        return 0.0
    return _clamp01(float(np.mean(corr_values)))


def _block_periodicity(gray: np.ndarray) -> float:
    """8px 경계의 차분이 주변 차분보다 강한 정도를 측정한다."""
    if gray.shape[0] < 16 or gray.shape[1] < 16:
        return 0.0

    vdiff = np.abs(np.diff(gray, axis=1))
    hdiff = np.abs(np.diff(gray, axis=0))

    v_boundary = vdiff[:, 7::8].mean() if vdiff[:, 7::8].size else 0.0
    h_boundary = hdiff[7::8, :].mean() if hdiff[7::8, :].size else 0.0
    v_all = float(vdiff.mean()) + 1e-8
    h_all = float(hdiff.mean()) + 1e-8

    ratio = ((float(v_boundary) / v_all) + (float(h_boundary) / h_all)) / 2.0
    return _clamp01((ratio - 1.0) / 0.8)


def _residual_strength(gray: np.ndarray) -> float:
    source = Image.fromarray(np.uint8(np.clip(gray * 255, 0, 255)))
    blurred = np.asarray(source.filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float32) / 255.0
    residual = gray - blurred
    return _clamp01(float(np.std(residual)) / 0.12)


def _residual_stats(gray: np.ndarray) -> tuple[float, float, float]:
    source = Image.fromarray(np.uint8(np.clip(gray * 255, 0, 255)))
    blurred = np.asarray(source.filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float32) / 255.0
    residual = (gray - blurred).astype(np.float64)
    centered = residual - float(np.mean(residual))
    std = float(np.std(centered)) + 1e-8
    mean_abs = _clamp01(float(np.mean(np.abs(residual))) / 0.12)
    skewness = _clamp01(abs(float(np.mean((centered / std) ** 3))) / 2.5)
    kurtosis = _clamp01(abs(float(np.mean((centered / std) ** 4)) - 3.0) / 8.0)
    return mean_abs, skewness, kurtosis


def _gray_entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=64, range=(0.0, 1.0), density=False)
    prob = hist.astype(np.float64) / max(float(np.sum(hist)), 1.0)
    prob = prob[prob > 0]
    entropy = -float(np.sum(prob * np.log2(prob)))
    return _clamp01(entropy / 6.0)


def _gradient_energy(gray: np.ndarray) -> float:
    gy, gx = np.gradient(gray.astype(np.float64))
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return _clamp01(float(np.mean(mag)) / 0.18)


def _color_stats(rgb: np.ndarray) -> tuple[float, float, float]:
    color_std_mean = _clamp01(float(np.mean(np.std(rgb, axis=(0, 1)))) / 0.30)
    max_c = np.max(rgb, axis=2)
    min_c = np.min(rgb, axis=2)
    saturation = (max_c - min_c) / (max_c + 1e-8)
    return (
        color_std_mean,
        _clamp01(float(np.mean(saturation))),
        _clamp01(float(np.std(saturation)) / 0.35),
    )


def _predict_learned(features: dict[str, float]) -> Optional[LearnedFingerprintPrediction]:
    bundle = _load_classifier()
    if not bundle:
        return None

    feature_order = bundle.get("feature_order", FINGERPRINT_FEATURE_ORDER)
    vector = np.asarray([[float(features.get(name, 0.0)) for name in feature_order]], dtype=np.float64)
    binary_model = bundle.get("binary_model")
    if binary_model is None or not hasattr(binary_model, "predict_proba"):
        return None

    probabilities = binary_model.predict_proba(vector)[0]
    classes = list(getattr(binary_model, "classes_", [0, 1]))
    fake_idx = classes.index(1) if 1 in classes else int(np.argmax(classes))
    probability = _clamp01(float(probabilities[fake_idx]))
    confidence = _clamp01(abs(probability - 0.5) * 2.0)

    method = "unknown"
    method_model = bundle.get("method_model")
    if method_model is not None and hasattr(method_model, "predict_proba"):
        method_probs = method_model.predict_proba(vector)[0]
        method_classes = list(getattr(method_model, "classes_", []))
        if method_classes:
            method_raw = str(method_classes[int(np.argmax(method_probs))])
            method = _METHOD_LABELS.get(method_raw, method_raw)
            confidence = _clamp01(0.65 * confidence + 0.35 * float(np.max(method_probs)))

    meta = bundle.get("meta", {})
    return LearnedFingerprintPrediction(
        probability=probability,
        method=method,
        confidence=confidence,
        source=str(meta.get("data_source", "unknown")),
        test_auc=_as_optional_float(meta.get("test_auc")),
    )


def _predict_video_learned(features: dict[str, float]) -> Optional[LearnedFingerprintPrediction]:
    bundle = _load_video_classifier()
    if not bundle:
        return None

    feature_order = bundle.get("feature_order", VIDEO_FINGERPRINT_FEATURE_ORDER)
    vector = np.asarray([[float(features.get(name, 0.0)) for name in feature_order]], dtype=np.float64)
    binary_model = bundle.get("binary_model")
    if binary_model is None or not hasattr(binary_model, "predict_proba"):
        return None

    probabilities = binary_model.predict_proba(vector)[0]
    classes = list(getattr(binary_model, "classes_", [0, 1]))
    fake_idx = classes.index(1) if 1 in classes else int(np.argmax(classes))
    probability = _clamp01(float(probabilities[fake_idx]))
    confidence = _clamp01(abs(probability - 0.5) * 2.0)

    method = "unknown"
    method_raw = "unknown"
    method_model = bundle.get("method_model")
    if method_model is not None and hasattr(method_model, "predict_proba"):
        method_probs = method_model.predict_proba(vector)[0]
        method_classes = list(getattr(method_model, "classes_", []))
        if method_classes:
            method_raw = str(method_classes[int(np.argmax(method_probs))])
            method = _VIDEO_METHOD_LABELS.get(method_raw, method_raw)
            confidence = _clamp01(0.65 * confidence + 0.35 * float(np.max(method_probs)))

    meta = bundle.get("meta", {})
    method_metrics = meta.get("method_metrics", {})
    method_threshold = None
    if isinstance(method_metrics, dict):
        method_row = method_metrics.get(method_raw, {})
        if isinstance(method_row, dict):
            method_auc = _as_optional_float(method_row.get("auc_vs_real"))
            if method_auc is not None and method_auc >= 0.80:
                method_threshold = _as_optional_float(method_row.get("best_threshold"))
    threshold = method_threshold or _as_optional_float(meta.get("best_threshold"))
    calibrated_probability = _threshold_calibrated_probability(
        probability,
        threshold,
        target=VIDEO_THRESHOLD_TARGET,
    )
    return LearnedFingerprintPrediction(
        probability=probability,
        method=method,
        confidence=confidence,
        source=str(meta.get("data_source", "unknown")),
        test_auc=_as_optional_float(meta.get("test_auc")),
        threshold=threshold,
        calibrated_probability=calibrated_probability,
        best_accuracy=_as_optional_float(meta.get("best_accuracy")),
    )


def _load_classifier() -> Optional[dict]:
    global _CLASSIFIER_CACHE, _CLASSIFIER_MTIME
    if not CLASSIFIER_PATH.exists():
        _CLASSIFIER_CACHE = None
        _CLASSIFIER_MTIME = None
        return None

    mtime = CLASSIFIER_PATH.stat().st_mtime
    if _CLASSIFIER_CACHE is not None and _CLASSIFIER_MTIME == mtime:
        return _CLASSIFIER_CACHE

    try:
        import joblib  # type: ignore

        bundle = joblib.load(CLASSIFIER_PATH)
        if not isinstance(bundle, dict):
            return None
        _CLASSIFIER_CACHE = bundle
        _CLASSIFIER_MTIME = mtime
    except Exception:
        _CLASSIFIER_CACHE = None
        _CLASSIFIER_MTIME = mtime
        return None
    return _CLASSIFIER_CACHE


def _load_video_classifier() -> Optional[dict]:
    global _VIDEO_CLASSIFIER_CACHE, _VIDEO_CLASSIFIER_MTIME
    if not VIDEO_CLASSIFIER_PATH.exists():
        _VIDEO_CLASSIFIER_CACHE = None
        _VIDEO_CLASSIFIER_MTIME = None
        return None

    mtime = VIDEO_CLASSIFIER_PATH.stat().st_mtime
    if _VIDEO_CLASSIFIER_CACHE is not None and _VIDEO_CLASSIFIER_MTIME == mtime:
        return _VIDEO_CLASSIFIER_CACHE

    try:
        import joblib  # type: ignore

        bundle = joblib.load(VIDEO_CLASSIFIER_PATH)
        if not isinstance(bundle, dict):
            return None
        _VIDEO_CLASSIFIER_CACHE = bundle
        _VIDEO_CLASSIFIER_MTIME = mtime
    except Exception:
        _VIDEO_CLASSIFIER_CACHE = None
        _VIDEO_CLASSIFIER_MTIME = mtime
        return None
    return _VIDEO_CLASSIFIER_CACHE


def feature_vector(features: dict[str, float], feature_order: list[str] | None = None) -> list[float]:
    """학습/추론 스크립트용 고정 순서 feature vector."""
    order = feature_order or FINGERPRINT_FEATURE_ORDER
    return [float(features.get(name, 0.0)) for name in order]


def video_feature_vector(features: dict[str, float], feature_order: list[str] | None = None) -> list[float]:
    """영상 fingerprint 학습/추론 스크립트용 고정 순서 feature vector."""
    order = feature_order or VIDEO_FINGERPRINT_FEATURE_ORDER
    return [float(features.get(name, 0.0)) for name in order]


def _score_features(features: dict[str, float]) -> float:
    high_freq_score = _smoothstep(features["high_freq_ratio"], 0.22, 0.44)
    flatness_score = _smoothstep(features["spectral_flatness"], 0.16, 0.36)
    corr_score = _smoothstep(features["channel_noise_corr"], 0.38, 0.78)
    residual_score = 1.0 - _smoothstep(features["residual_strength"], 0.55, 0.95)
    block_score = 1.0 - (0.45 * features["block_periodicity"])

    score = (
        0.30 * high_freq_score
        + 0.25 * flatness_score
        + 0.25 * corr_score
        + 0.15 * residual_score
        + 0.05 * block_score
    )
    return round(_clamp01(score), 3)


def _score_video_features(features: dict[str, float]) -> float:
    """영상 얼굴 crop용 압축 보정 점수.

    동영상 프레임은 인코딩/디코딩 과정만으로도 채널 노이즈 상관성과
    spectral flatness가 높아질 수 있어 이미지용 score를 그대로 쓰면
    real/fake 모두 강한 AI 신호로 포화된다.
    """
    raw_score = _score_features(features)
    compression_bias = (
        0.12 * _smoothstep(features["channel_noise_corr"], 0.92, 0.99)
        + 0.06 * _smoothstep(features["spectral_flatness"], 0.42, 0.56)
    )
    block_support = 0.08 * _smoothstep(features["block_periodicity"], 0.04, 0.12)
    return round(_clamp01(raw_score - compression_bias + block_support), 3)


def _likelihood_label(ai_score: float) -> str:
    if ai_score >= AI_HIGH:
        return "높음"
    if ai_score >= AI_MID:
        return "중간"
    return "낮음"


def _video_likelihood_label(ai_score: float) -> str:
    if ai_score >= VIDEO_AI_HIGH:
        return "높음"
    if ai_score >= VIDEO_AI_MID:
        return "중간"
    return "낮음"


def _threshold_calibrated_probability(
    probability: float,
    threshold: Optional[float],
    target: float,
) -> float:
    """분류기 best_threshold가 target 점수에 오도록 영상 확률을 재스케일한다."""
    probability = _clamp01(probability)
    if threshold is None or threshold <= 0.0 or threshold >= 1.0:
        return probability
    target = _clamp01(target)
    if probability < threshold:
        return _clamp01((probability / threshold) * target)
    return _clamp01(target + ((probability - threshold) / (1.0 - threshold)) * (1.0 - target))


def _confidence(ai_score: float, features: dict[str, float]) -> float:
    distance = abs(ai_score - 0.5) * 2.0
    feature_support = np.mean([
        _smoothstep(features["high_freq_ratio"], 0.22, 0.44),
        _smoothstep(features["spectral_flatness"], 0.16, 0.36),
        _smoothstep(features["channel_noise_corr"], 0.38, 0.78),
    ])
    return round(_clamp01(0.45 * distance + 0.55 * float(feature_support)), 3)


def _majority_method(methods: list[str]) -> str:
    if not methods:
        return "unknown"
    return Counter(methods).most_common(1)[0][0]


def _as_optional_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _fmt_optional(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _smoothstep(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    x = _clamp01((value - low) / (high - low))
    return x * x * (3.0 - 2.0 * x)


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))
