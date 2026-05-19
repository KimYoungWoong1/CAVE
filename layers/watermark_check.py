"""레이어 2: 워터마크/메타 마커 탐지.

SynthID, Firefly 등 주요 생성 도구의 실제 invisible watermark 탐지는
대부분 비공개 API에 의존한다. 이 프로토타입은 공개적으로 접근 가능한
XMP/EXIF/바이너리 문자열에서 워터마크·Content Credentials·생성 도구
마커를 찾는 제한적 구현이다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image


_AI_MARKERS = (
    "synthid",
    "firefly",
    "adobe firefly",
    "stable diffusion",
    "midjourney",
    "dall-e",
    "dall·e",
    "imagen",
    "imagefx",
    "runway",
    "openai",
    "stability ai",
)

_WATERMARK_MARKERS = (
    "watermark",
    "synthid",
    "content credentials",
    "contentcredentials",
    "c2pa",
    "cai",
    "xmp",
)


@dataclass
class WatermarkResult:
    watermark_present: Optional[bool]   # None = 분석 미수행
    ai_watermark: Optional[bool]
    confidence: float
    ai_score: Optional[float]
    notes: str


def run(file_path: str) -> WatermarkResult:
    """공개 메타데이터/바이너리 마커 기반 워터마크 탐지."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    text_sources = _collect_text_sources(path)
    joined = "\n".join(text_sources).lower()
    normalized = _normalize(joined)

    watermark_hits = _find_hits(normalized, _WATERMARK_MARKERS)
    ai_hits = _find_hits(normalized, _AI_MARKERS)

    watermark_present = bool(watermark_hits)
    ai_watermark = bool(ai_hits)

    if ai_watermark:
        ai_score = 1.0
        confidence = 0.65 if watermark_present else 0.45
    elif watermark_present:
        ai_score = 0.35
        confidence = 0.45
    else:
        ai_score = 0.0
        confidence = 0.25

    notes = _build_notes(watermark_hits, ai_hits, len(text_sources))

    return WatermarkResult(
        watermark_present=watermark_present,
        ai_watermark=ai_watermark,
        confidence=confidence,
        ai_score=ai_score,
        notes=notes,
    )


def _collect_text_sources(path: Path) -> list[str]:
    sources: list[str] = []
    sources.extend(_image_metadata_text(path))
    sources.append(_binary_text(path))
    return [source for source in sources if source]


def _image_metadata_text(path: Path) -> list[str]:
    chunks: list[str] = []
    try:
        with Image.open(path) as img:
            for key, value in img.info.items():
                chunks.append(f"{key}: {value}")
            exif = img.getexif()
            for key, value in exif.items():
                chunks.append(f"{key}: {value}")
    except Exception:
        return chunks
    return chunks


def _binary_text(path: Path, max_bytes: int = 2_000_000) -> str:
    data = path.read_bytes()[:max_bytes]
    return data.decode("utf-8", errors="ignore")


def _find_hits(text: str, markers: tuple[str, ...]) -> list[str]:
    return sorted({marker for marker in markers if _normalize(marker) in text})


def _normalize(value: str) -> str:
    return " ".join(value.replace("_", " ").replace("-", " ").lower().split())


def _build_notes(watermark_hits: list[str], ai_hits: list[str], source_count: int) -> str:
    if ai_hits:
        return (
            "공개 메타데이터/바이너리 마커에서 AI 생성 도구 또는 워터마크 관련 신호 감지. "
            f"AI hits={', '.join(ai_hits)}; watermark hits={', '.join(watermark_hits) or '없음'}. "
            "실제 invisible watermark 검출이 아닌 제한적 마커 탐지 결과."
        )
    if watermark_hits:
        return (
            "워터마크 또는 Content Credentials 관련 마커는 감지됐지만 AI 워터마크로 단정 불가. "
            f"watermark hits={', '.join(watermark_hits)}. "
            "실제 invisible watermark 검출이 아닌 제한적 마커 탐지 결과."
        )
    return (
        f"공개 메타데이터/바이너리 텍스트 {source_count}개 소스에서 워터마크 마커를 찾지 못함. "
        "비공개 SynthID/Firefly invisible watermark 부재를 증명하지는 않음."
    )
