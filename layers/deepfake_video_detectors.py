"""DeepfakeBench 계열 영상 딥페이크 detector 앙상블.

Face crop sequence를 입력으로 받아 여러 공개 백본의 fake 확률을 구하고
하나의 영상 레이어 점수로 합친다.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Optional

import numpy as np
from PIL import Image


LEGACY_EFFB0_REPO = "Xicor9/efficientnet-b0-ffpp-c23"
LEGACY_EFFB0_FILE = "efficientnet_b0_ffpp_c23.pth"
BENCHMARK_REPO = "abraraltaf92/deepfake-detection-models"
XCEPTION_REPO = "huzaifanasirrr/faceforge-detector"
CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "models" / "video_calibration.json"

DEVICE = None
MODEL_CACHE: dict[str, object] = {}
CALIBRATION_CACHE: Optional[dict] = None
CALIBRATION_MTIME: Optional[float] = None
CALIBRATION_CACHE_PATH: Optional[Path] = None


@dataclass
class DetectorPrediction:
    name: str
    probability: float
    weight: float
    frame_std: Optional[float]
    detail: str


@dataclass
class DetectorSuiteResult:
    raw_probability: float
    calibrated_probability: float
    predictions: list[DetectorPrediction]
    errors: list[str]
    calibration_note: str

    @property
    def feature_map(self) -> dict[str, float]:
        data = {prediction.name: prediction.probability for prediction in self.predictions}
        data["ensemble_raw"] = self.raw_probability
        return data


def predict_video_detector_suite(
    face_crops: list[Image.Image],
    face_ratio: float,
) -> DetectorSuiteResult:
    """대표 얼굴 crop들에 여러 detector를 적용하고 앙상블 점수를 반환한다."""
    if not face_crops:
        raise ValueError("영상 detector에 사용할 얼굴 crop이 없습니다.")

    specs = [
        ("deepfakebench_efficientnet_b4", 0.30, _predict_deepfakebench_efficientnet_b4),
        ("faceforge_xception", 0.22, _predict_faceforge_xception),
        ("deepfakebench_r3d18", 0.24, _predict_deepfakebench_r3d18),
        ("deepfakebench_resnet18", 0.14, _predict_deepfakebench_resnet18),
        ("legacy_efficientnet_b0", 0.10, _predict_legacy_efficientnet_b0),
    ]

    predictions: list[DetectorPrediction] = []
    errors: list[str] = []
    for name, weight, predictor in specs:
        try:
            probability, frame_std, detail = predictor(face_crops)
            predictions.append(
                DetectorPrediction(
                    name=name,
                    probability=_clamp01(probability),
                    weight=weight,
                    frame_std=frame_std,
                    detail=detail,
                )
            )
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    if not predictions:
        raise RuntimeError("; ".join(errors) if errors else "실행 가능한 영상 detector가 없습니다.")

    raw = _weighted_average(predictions)
    calibrated, calibration_note = _calibrate_probability(raw, face_ratio, predictions)
    return DetectorSuiteResult(
        raw_probability=raw,
        calibrated_probability=calibrated,
        predictions=predictions,
        errors=errors,
        calibration_note=calibration_note,
    )


def _predict_deepfakebench_efficientnet_b4(face_crops: list[Image.Image]) -> tuple[float, Optional[float], str]:
    model, device = _get_model("deepfakebench_efficientnet_b4", _build_deepfakebench_efficientnet_b4)
    tensor = _make_frame_batch(face_crops, size=224, norm="imagenet").to(device)
    probability = _predict_clip_model_probability(model, tensor)
    return probability, None, "DeepfakeBench-style EfficientNet-B4, 16-frame feature mean pooling"


def _predict_deepfakebench_resnet18(face_crops: list[Image.Image]) -> tuple[float, Optional[float], str]:
    model, device = _get_model("deepfakebench_resnet18", _build_deepfakebench_resnet18)
    tensor = _make_frame_batch(face_crops, size=224, norm="imagenet").to(device)
    probability = _predict_clip_model_probability(model, tensor)
    return probability, None, "DeepfakeBench-style ResNet18, 16-frame feature mean pooling"


def _predict_deepfakebench_r3d18(face_crops: list[Image.Image]) -> tuple[float, Optional[float], str]:
    model, device = _get_model("deepfakebench_r3d18", _build_deepfakebench_r3d18)
    tensor = _make_frame_batch(face_crops, size=112, norm="kinetics")
    tensor = tensor.permute(0, 2, 1, 3, 4).contiguous().to(device)
    probability = _predict_clip_model_probability(model, tensor)
    return probability, None, "DeepfakeBench-style R3D-18 temporal RGB clip detector"


def _predict_faceforge_xception(face_crops: list[Image.Image]) -> tuple[float, Optional[float], str]:
    model, device = _get_model("faceforge_xception", _build_faceforge_xception)
    tensor = _make_flat_frame_batch(face_crops, size=224, norm="half").to(device)
    probabilities = _predict_frame_model_probabilities(model, tensor)
    probability = _aggregate_frame_probabilities(probabilities)
    return probability, float(np.std(probabilities)), "XceptionNet face detector, frame probabilities aggregated"


def _predict_legacy_efficientnet_b0(face_crops: list[Image.Image]) -> tuple[float, Optional[float], str]:
    model, device = _get_model("legacy_efficientnet_b0", _build_legacy_efficientnet_b0)
    tensor = _make_flat_frame_batch(face_crops, size=224, norm="imagenet").to(device)
    probabilities = _predict_frame_model_probabilities(model, tensor)
    probability = _aggregate_frame_probabilities(probabilities)
    return probability, float(np.std(probabilities)), "기존 FF++ EfficientNet-B0 frame detector"


def _build_deepfakebench_efficientnet_b4():
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from torchvision import models

    model = _FrameFeatureMeanDetector(
        backbone=models.efficientnet_b4(weights=None),
        feature_dim=1792,
    )
    model.backbone.classifier = nn.Identity()
    checkpoint_path = hf_hub_download(BENCHMARK_REPO, "efficientnet_b4_best.pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def _build_deepfakebench_resnet18():
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from torchvision import models

    model = _FrameFeatureMeanDetector(
        backbone=models.resnet18(weights=None),
        feature_dim=512,
    )
    model.backbone.fc = nn.Identity()
    checkpoint_path = hf_hub_download(BENCHMARK_REPO, "resnet18_best.pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def _build_deepfakebench_r3d18():
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from torchvision.models.video import r3d_18

    model = _VideoClipDetector(backbone=r3d_18(weights=None), feature_dim=512)
    model.backbone.fc = nn.Identity()
    checkpoint_path = hf_hub_download(BENCHMARK_REPO, "r3d18_best.pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def _build_faceforge_xception():
    import torch
    import torch.nn as nn
    import timm
    from huggingface_hub import hf_hub_download

    class FaceForgeXception(nn.Module):
        def __init__(self):
            super().__init__()
            self.xception = timm.create_model("xception", pretrained=False, num_classes=0)
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 2),
            )

        def forward(self, x):
            return self.classifier(self.xception(x))

    model = FaceForgeXception()
    checkpoint_path = hf_hub_download(XCEPTION_REPO, "detector_best.pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _build_legacy_efficientnet_b0():
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from torchvision import models

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    checkpoint_path = hf_hub_download(LEGACY_EFFB0_REPO, LEGACY_EFFB0_FILE)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def _get_model(name: str, builder: Callable[[], object]):
    import torch

    cached = MODEL_CACHE.get(name)
    device = _get_device()
    if cached is None:
        model = builder()
        model.to(device)
        model.eval()
        MODEL_CACHE[name] = model
    else:
        model = cached
    return model, device


def _get_device():
    global DEVICE
    if DEVICE is not None:
        return DEVICE

    import torch

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    return DEVICE


class _FrameFeatureMeanDetector:
    def __init__(self, backbone, feature_dim: int):
        import torch.nn as nn

        super().__setattr__("module", nn.Module())
        self.module.backbone = backbone
        self.module.head = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(feature_dim, 2))

    def __getattr__(self, name):
        return getattr(self.module, name)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.module.to(*args, **kwargs)
        return self

    def eval(self):
        self.module.eval()
        return self

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.module.load_state_dict(*args, **kwargs)

    def forward(self, x):
        import torch

        if x.ndim == 5:
            batch, frames, channels, height, width = x.shape
            flat = x.reshape(batch * frames, channels, height, width)
            features = self.module.backbone(flat)
            features = features.reshape(batch, frames, -1).mean(dim=1)
        else:
            features = self.module.backbone(x)
        return self.module.head(features)


class _VideoClipDetector:
    def __init__(self, backbone, feature_dim: int):
        import torch.nn as nn

        super().__setattr__("module", nn.Module())
        self.module.backbone = backbone
        self.module.head = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(feature_dim, 2))

    def __getattr__(self, name):
        return getattr(self.module, name)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.module.to(*args, **kwargs)
        return self

    def eval(self):
        self.module.eval()
        return self

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.module.load_state_dict(*args, **kwargs)

    def forward(self, x):
        features = self.module.backbone(x)
        return self.module.head(features)


def _make_frame_batch(face_crops: list[Image.Image], size: int, norm: str, frames: int = 16):
    import torch

    selected = _select_clip_frames(face_crops, frames)
    tensors = [_transform_image(img, size=size, norm=norm) for img in selected]
    return torch.stack(tensors, dim=0).unsqueeze(0)


def _make_flat_frame_batch(face_crops: list[Image.Image], size: int, norm: str):
    import torch

    tensors = [_transform_image(img, size=size, norm=norm) for img in face_crops]
    return torch.stack(tensors, dim=0)


def _transform_image(image: Image.Image, size: int, norm: str):
    from torchvision import transforms

    if norm == "imagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif norm == "kinetics":
        normalize = transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
    elif norm == "half":
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        raise ValueError(f"알 수 없는 정규화 방식: {norm}")

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ])
    return transform(image.convert("RGB"))


def _select_clip_frames(face_crops: list[Image.Image], frames: int) -> list[Image.Image]:
    if len(face_crops) == frames:
        return face_crops
    if len(face_crops) > frames:
        indices = np.linspace(0, len(face_crops) - 1, frames, dtype=int)
        return [face_crops[int(idx)] for idx in indices]
    result = list(face_crops)
    while len(result) < frames:
        result.append(face_crops[-1])
    return result


def _predict_clip_model_probability(model, tensor) -> float:
    import torch

    with torch.no_grad():
        logits = model(tensor)
        probability = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()[0]
    return _clamp01(float(probability))


def _predict_frame_model_probabilities(model, tensor) -> list[float]:
    import torch

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    return [_clamp01(float(probability)) for probability in probabilities]


def _aggregate_frame_probabilities(probabilities: list[float]) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    return _clamp01(
        0.60 * float(np.mean(probs))
        + 0.25 * float(np.percentile(probs, 90))
        + 0.15 * float(np.max(probs))
    )


def _weighted_average(predictions: list[DetectorPrediction]) -> float:
    total_weight = sum(prediction.weight for prediction in predictions)
    if total_weight <= 0:
        return 0.5
    return _clamp01(
        sum(prediction.probability * prediction.weight for prediction in predictions)
        / total_weight
    )


def _calibrate_probability(
    raw_probability: float,
    face_ratio: float,
    predictions: list[DetectorPrediction],
) -> tuple[float, str]:
    calibration_path = _active_calibration_path()
    calibration = _load_calibration()
    if calibration:
        method = calibration.get("method")
        if method == "logistic":
            feature_map = {prediction.name: prediction.probability for prediction in predictions}
            feature_map["ensemble_raw"] = raw_probability
            feature_order = calibration.get("feature_order", [])
            coefficients = calibration.get("coefficients", [])
            if len(feature_order) == len(coefficients):
                z = float(calibration.get("intercept", 0.0))
                for name, coefficient in zip(feature_order, coefficients):
                    z += float(coefficient) * float(feature_map.get(name, 0.5))
                return _sigmoid(z), f"local logistic calibration: {calibration_path.name}"
        elif method == "threshold_shift":
            threshold = float(calibration.get("decision_threshold", 0.5))
            temperature = float(calibration.get("temperature", 0.75))
            return _threshold_shift(raw_probability, threshold, temperature), (
                f"local threshold calibration threshold={threshold:.3f}"
            )

    coverage = _clamp01(face_ratio)
    conservative = 0.5 + (raw_probability - 0.5) * (0.72 + 0.28 * coverage)
    return _clamp01(conservative), "default conservative ensemble calibration"


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
    override = os.environ.get("CAVE_VIDEO_CALIBRATION", "").strip()
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


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))
