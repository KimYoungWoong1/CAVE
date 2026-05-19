"""영상 프레임 샘플링과 얼굴 crop 공통 유틸."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import numpy as np
from PIL import Image


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MEDIAPIPE_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
MEDIAPIPE_FACE_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "blaze_face_short_range.tflite"
)

_MP_FACE_DETECTOR = None
_MTCNN_DETECTOR = None
_HAAR_CASCADE = None


@dataclass
class FaceCropSample:
    image: Image.Image
    luminance: float
    detected: bool
    detector: str
    frame_index: int


def is_video_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def sample_video_face_crops(
    path: str | Path,
    num_frames: int = 12,
    start_ratio: float = 0.08,
    end_ratio: float = 0.92,
) -> list[FaceCropSample]:
    """영상에서 대표 프레임을 뽑아 얼굴 crop PIL 이미지 목록을 반환한다."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("opencv-python이 필요합니다.") from exc

    video_path = Path(path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("영상을 열 수 없습니다.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        raise ValueError("영상 프레임 수를 읽을 수 없습니다.")

    start = int(frame_count * start_ratio)
    end = max(start + 1, int(frame_count * end_ratio))
    indices = np.linspace(start, end - 1, num=min(num_frames, frame_count), dtype=int)

    samples: list[FaceCropSample] = []
    for idx in sorted(set(int(i) for i in indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        crop_bgr, detected, detector = extract_face_crop_bgr(frame_bgr)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        samples.append(
            FaceCropSample(
                image=Image.fromarray(crop_rgb),
                luminance=float(np.mean(crop_rgb)),
                detected=detected,
                detector=detector,
                frame_index=idx,
            )
        )

    cap.release()
    if not samples:
        raise ValueError("영상에서 대표 얼굴 crop을 추출하지 못했습니다.")
    return samples


def extract_face_crop_bgr(frame_bgr: np.ndarray) -> tuple[np.ndarray, bool, str]:
    """MediaPipe -> MTCNN -> Haar -> 중앙 crop 순서로 얼굴 crop을 반환한다."""
    mp_result = _extract_mediapipe_face(frame_bgr)
    if mp_result is not None:
        return mp_result, True, "mediapipe"

    mtcnn_result = _extract_mtcnn_face(frame_bgr)
    if mtcnn_result is not None:
        return mtcnn_result, True, "mtcnn"

    haar_result = _extract_haar_face(frame_bgr)
    if haar_result is not None:
        return haar_result, True, "haar"

    return _central_face_fallback(frame_bgr), False, "center_fallback"


def _extract_mediapipe_face(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
    except ImportError:
        return None

    detector_info = _get_mediapipe_detector()
    if detector_info is None:
        return None

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detector_kind, detector = detector_info

    if detector_kind == "solutions":
        result = detector.process(frame_rgb)
        detections = getattr(result, "detections", None)
        if not detections:
            return None

        h, w = frame_bgr.shape[:2]
        best = max(detections, key=_legacy_mediapipe_score)
        bbox = best.location_data.relative_bounding_box
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        return _crop_with_margin(frame_bgr, x, y, bw, bh, margin_ratio=0.35)

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(image)
    detections = getattr(result, "detections", None)
    if not detections:
        return None

    best = max(detections, key=_tasks_mediapipe_score)
    bbox = best.bounding_box
    return _crop_with_margin(
        frame_bgr,
        int(bbox.origin_x),
        int(bbox.origin_y),
        int(bbox.width),
        int(bbox.height),
        margin_ratio=0.35,
    )


def _crop_with_margin(
    frame_bgr: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    margin_ratio: float,
) -> Optional[np.ndarray]:
    if width <= 0 or height <= 0:
        return None

    h, w = frame_bgr.shape[:2]
    margin = int(max(width, height) * margin_ratio)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + width + margin)
    y2 = min(h, y + height + margin)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def _get_mediapipe_detector():
    global _MP_FACE_DETECTOR
    if _MP_FACE_DETECTOR is not None:
        return _MP_FACE_DETECTOR
    try:
        import mediapipe as mp  # type: ignore
    except Exception:
        return None

    if hasattr(mp, "solutions"):
        _MP_FACE_DETECTOR = (
            "solutions",
            mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.45,
            ),
        )
        return _MP_FACE_DETECTOR

    try:
        from mediapipe.tasks import python as mp_python  # type: ignore
        from mediapipe.tasks.python import vision  # type: ignore

        model_path = _ensure_mediapipe_face_model()
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.45,
        )
        _MP_FACE_DETECTOR = ("tasks", vision.FaceDetector.create_from_options(options))
    except Exception:
        return None
    return _MP_FACE_DETECTOR


def _ensure_mediapipe_face_model() -> Path:
    if MEDIAPIPE_FACE_MODEL_PATH.exists():
        return MEDIAPIPE_FACE_MODEL_PATH
    MEDIAPIPE_FACE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(MEDIAPIPE_FACE_MODEL_URL, MEDIAPIPE_FACE_MODEL_PATH)
    return MEDIAPIPE_FACE_MODEL_PATH


def _legacy_mediapipe_score(detection) -> float:
    scores = getattr(detection, "score", None)
    if not scores:
        return 0.0
    return float(scores[0])


def _tasks_mediapipe_score(detection) -> float:
    categories = getattr(detection, "categories", None)
    if categories:
        return max(float(getattr(category, "score", 0.0)) for category in categories)
    scores = getattr(detection, "score", None)
    if scores:
        return float(scores[0])
    return 0.0


def _extract_mtcnn_face(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    detector = _get_mtcnn_detector()
    if detector is None:
        return None

    try:
        import cv2  # type: ignore
    except ImportError:
        return None

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    boxes, probs = detector.detect(pil_image)
    if boxes is None or len(boxes) == 0:
        return None

    scores = probs if probs is not None else [0.0] * len(boxes)
    best_idx = int(np.argmax(scores))
    x1, y1, x2, y2 = boxes[best_idx]
    confidence = float(scores[best_idx]) if scores is not None else 0.0
    if confidence < 0.70:
        return None

    h, w = frame_bgr.shape[:2]
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    margin = int(max(bw, bh) * 0.32)
    left = max(0, int(x1) - margin)
    top = max(0, int(y1) - margin)
    right = min(w, int(x2) + margin)
    bottom = min(h, int(y2) + margin)
    crop = frame_bgr[top:bottom, left:right]
    if crop.size == 0:
        return None
    return crop


def _get_mtcnn_detector():
    global _MTCNN_DETECTOR
    if _MTCNN_DETECTOR is not None:
        return _MTCNN_DETECTOR
    try:
        from facenet_pytorch import MTCNN  # type: ignore
    except Exception:
        return None

    _MTCNN_DETECTOR = MTCNN(keep_all=True, device="cpu")
    return _MTCNN_DETECTOR


def _extract_haar_face(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return None

    cascade = _get_haar_cascade()
    if cascade is None:
        return None

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80))
    if len(faces) == 0:
        return None

    x, y, fw, fh = max(faces, key=lambda box: box[2] * box[3])
    margin = int(max(fw, fh) * 0.28)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame_bgr.shape[1], x + fw + margin)
    y2 = min(frame_bgr.shape[0], y + fh + margin)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def _get_haar_cascade():
    global _HAAR_CASCADE
    if _HAAR_CASCADE is not None:
        return _HAAR_CASCADE
    try:
        import cv2  # type: ignore
    except ImportError:
        return None

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        return None
    _HAAR_CASCADE = cascade
    return _HAAR_CASCADE


def _central_face_fallback(frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    size = int(min(h, w) * 0.72)
    cx = w // 2
    cy = int(h * 0.45)
    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)
    return frame_bgr[y1:y2, x1:x2]
