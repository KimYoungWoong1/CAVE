"""레이어 1: C2PA 출처 검증 (Content Provenance and Authenticity).

C2PA는 출처(provenance) 기록 메커니즘일 뿐, 진정성(authenticity) 보증 불가.
메타데이터 삭제·타임스탬프 조작 취약점 존재 (Golaszewski et al., arXiv:2604.24890, 2026).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# AI 생성 도구로 알려진 claim_generator 키워드
_AI_GENERATORS = [
    "adobe firefly", "dall-e", "dall·e", "midjourney",
    "stable diffusion", "imagefx", "imagen", "runway",
    "openai", "anthropic", "stability ai", "google deepmind",
    "sora", "kling", "pika", "heygen",
]


@dataclass
class C2PAResult:
    manifest_present: bool
    signature_valid: str        # 'Valid' | 'Invalid' | 'None'
    ai_usage_declared: bool
    edit_history: list[str]
    claim_generator: Optional[str]
    raw_manifest: Optional[dict]
    ai_score: float             # 0.0 = 인간 제작 신호, 1.0 = AI 생성 신호
    notes: str


def run(file_path: str) -> C2PAResult:
    """C2PA 매니페스트 검증 실행."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    manifest_data = _try_read_c2pa(str(path))

    if manifest_data is None:
        return C2PAResult(
            manifest_present=False,
            signature_valid="None",
            ai_usage_declared=False,
            edit_history=[],
            claim_generator=None,
            raw_manifest=None,
            ai_score=0.0,
            notes=(
                "C2PA 매니페스트 없음. "
                "출처 정보 확인 불가 — 다른 레이어 결과와 종합 판단 필요. "
                "매니페스트 부재가 인간 제작을 증명하지 않음 (의도적 삭제 가능)."
            ),
        )

    return _parse_manifest(manifest_data)


# ─── 내부 헬퍼 ────────────────────────────────────────────────────────────────


def _try_read_c2pa(file_path: str) -> Optional[dict]:
    """c2pa-python 라이브러리로 매니페스트 읽기 시도; 실패 시 XMP 폴백."""
    # 1순위: c2pa-python 공식 라이브러리
    try:
        import c2pa  # type: ignore

        result_json: Optional[str] = None

        if hasattr(c2pa, "read_file"):
            result_json = c2pa.read_file(file_path, None)
        elif hasattr(c2pa, "Reader"):
            mime = _guess_mime(file_path)
            with open(file_path, "rb") as fh:
                reader = c2pa.Reader(mime, fh)
                result_json = reader.json()

        if result_json:
            return json.loads(result_json)
    except Exception:
        pass

    # 2순위: XMP/EXIF에서 C2PA 마커 탐지 (Pillow 폴백)
    return _try_xmp_fallback(file_path)


def _try_xmp_fallback(file_path: str) -> Optional[dict]:
    """PIL로 C2PA XMP 마커 존재 여부 확인 (간이 탐지)."""
    try:
        from PIL import Image  # type: ignore

        with Image.open(file_path) as img:
            xmp_bytes: bytes = img.info.get("xmp", b"")

        if not xmp_bytes:
            return None

        xmp_str = xmp_bytes.decode("utf-8", errors="ignore").lower()
        if "c2pa" in xmp_str or "contentcredentials" in xmp_str:
            return {
                "_detected_via": "xmp_marker",
                "_note": "c2pa-python 미설치로 완전 파싱 불가 — XMP 마커만 감지됨",
                "active_manifest": "",
                "manifests": {},
            }
    except Exception:
        pass
    return None


def _guess_mime(file_path: str) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
    }.get(Path(file_path).suffix.lower(), "application/octet-stream")


def _parse_manifest(manifest_data: dict) -> C2PAResult:
    """C2PA 매니페스트 JSON → C2PAResult 변환."""
    detected_via_xmp = manifest_data.get("_detected_via") == "xmp_marker"

    if detected_via_xmp:
        return C2PAResult(
            manifest_present=True,
            signature_valid="None",
            ai_usage_declared=False,
            edit_history=[],
            claim_generator=None,
            raw_manifest=manifest_data,
            ai_score=0.3,
            notes=(
                "XMP 마커로 C2PA 존재 확인됨. "
                "c2pa-python 미설치로 서명 검증 불가. "
                "`pip install c2pa-python` 설치 후 재실행 권장."
            ),
        )

    active_label = manifest_data.get("active_manifest", "")
    manifests: dict = manifest_data.get("manifests", {})
    active: dict = manifests.get(active_label, {}) if active_label else {}

    # 서명 유효성
    sig_info = active.get("signature_info", {})
    validation_errors: list = active.get("validation_status", [])
    if sig_info:
        signature_valid = "Invalid" if validation_errors else "Valid"
    else:
        signature_valid = "None"

    # claim_generator (생성 소프트웨어)
    claim_generator: str = active.get("claim_generator", "")

    # 어서션 파싱 (AI 선언, 편집 이력)
    assertions: list[dict] = active.get("assertions", [])
    ai_usage_declared = False
    edit_history: list[str] = []

    for assertion in assertions:
        label: str = assertion.get("label", "")
        data: dict = assertion.get("data", {})

        label_lower = label.lower()
        # AI 사용 선언 어서션
        if any(kw in label_lower for kw in ("ai.generated", "c2pa.ai", "training", "generative")):
            ai_usage_declared = True

        # 편집 이력 (c2pa.actions)
        if label == "c2pa.actions":
            for action in data.get("actions", []):
                edit_history.append(action.get("action", "unknown"))

    # claim_generator 기반 AI 판정
    gen_normalized = _normalize_generator_text(claim_generator)
    if any(_normalize_generator_text(g) in gen_normalized for g in _AI_GENERATORS):
        ai_usage_declared = True

    # ai_score 산출
    if ai_usage_declared and signature_valid == "Valid":
        ai_score = 1.0
    elif ai_usage_declared:
        ai_score = 0.85  # AI 선언됐으나 서명 이상
    elif signature_valid == "Invalid":
        ai_score = 0.4   # 서명 위조 가능성
    else:
        ai_score = 0.0

    # notes 조합
    notes_parts = [f"서명 상태: {signature_valid}."]
    if claim_generator:
        notes_parts.append(f"생성 도구: {claim_generator}.")
    if ai_usage_declared:
        notes_parts.append("매니페스트에 AI 생성 정보 기록됨.")
    if edit_history:
        notes_parts.append(f"편집 이력: {', '.join(edit_history)}.")
    if validation_errors:
        notes_parts.append(f"서명 오류 {len(validation_errors)}건 — 위변조 의심.")

    return C2PAResult(
        manifest_present=True,
        signature_valid=signature_valid,
        ai_usage_declared=ai_usage_declared,
        edit_history=edit_history,
        claim_generator=claim_generator or None,
        raw_manifest=manifest_data,
        ai_score=ai_score,
        notes=" ".join(notes_parts),
    )


def _normalize_generator_text(value: str) -> str:
    """생성 도구 표기의 공백/구분자 차이를 흡수한다."""
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
