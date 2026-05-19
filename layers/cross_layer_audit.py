"""레이어 6: Cross-layer Audit — Integrity Clash 탐지.

핵심 개념: Integrity Clash (Nemecek et al., arXiv:2603.02378, 2026)
  - C2PA와 워터마킹이 비동기 작동 → 두 검증 결과가 모순 가능
  - metadata washing workflow로 암호학적 침해 없이 인증된 가짜 생성 가능

구현:
  - 출처 레이어(1,2)와 탐지 레이어(3,4,5)를 분리하여 방향성 일치 확인
  - cosine similarity로 충돌 강도 측정
  - CONTEXT.md 판정 매트릭스 적용
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# cosine similarity 임계값: 이 미만이면 Integrity Clash
CLASH_COS_THRESHOLD = 0.40

# ai_score 판정 임계값
AI_HIGH = 0.60    # 이상 → AI 생성 신호
AI_LOW  = 0.30    # 이하 → 인간 제작 신호
VIDEO_SOFT_SUPPORT = 0.60
VIDEO_MID_SUPPORT = 0.45
VIDEO_RPPG_STRONG_SUPPORT = 0.70
IMAGE_DETECTOR_STRONG_SUPPORT = 0.75
IMAGE_FINGERPRINT_STRONG_SUPPORT = 0.80
IMAGE_MID_SUPPORT = 0.55
LOW_CONSISTENCY_REVIEW = 0.35


@dataclass
class CrossLayerResult:
    verdict: str            # 영문 키
    verdict_kr: str         # 한국어 판정문
    integrity_clash: bool
    expert_review_needed: bool
    consistency_score: float            # 0~1 (1 = 완전 일관)
    provenance_avg: Optional[float]     # 출처 레이어 평균 AI 점수
    detection_avg: Optional[float]      # 탐지 레이어 평균 AI 점수
    clash_details: list[str]
    layer_scores: dict[str, Optional[float]]
    notes: str


def run(layer_results: dict) -> CrossLayerResult:
    """Cross-layer Audit 실행.

    Args:
        layer_results: 키 'c2pa', 'watermark', 'ai_detection', 'rppg', 'fingerprint'
                       각 값은 .ai_score 속성을 가진 결과 객체 (float 0~1 또는 None)
    """
    layer_scores = {
        key: _extract_score(key, layer_results.get(key))
        for key in ("c2pa", "watermark", "ai_detection", "rppg", "fingerprint")
    }

    # 출처 레이어(1,2) vs 탐지 레이어(3,4,5) 분리
    provenance_scores = _valid_scores(layer_scores, ("c2pa", "watermark"))
    detection_scores  = _valid_scores(layer_scores, ("ai_detection", "rppg", "fingerprint"))

    provenance_avg = float(np.mean(provenance_scores)) if provenance_scores else None
    detection_avg  = float(np.mean(detection_scores))  if detection_scores  else None

    all_scores = [v for v in layer_scores.values() if v is not None]
    consistency_score = _compute_consistency(all_scores)

    clash_details: list[str] = []
    integrity_clash = False

    # ── 규칙 기반 충돌 탐지 ──────────────────────────────────────────────────
    c2pa_s = layer_scores["c2pa"]
    wm_s   = layer_scores["watermark"]
    ai_s   = layer_scores["ai_detection"]
    rppg_s = layer_scores["rppg"]
    fp_s   = layer_scores["fingerprint"]
    is_video_input = rppg_s is not None

    # Case: C2PA 인간 ↔ 워터마크 AI
    if c2pa_s is not None and wm_s is not None:
        if c2pa_s <= AI_LOW and wm_s >= AI_HIGH:
            integrity_clash = True
            clash_details.append(
                f"[C2PA ↔ 워터마크 충돌] C2PA={c2pa_s:.2f}(인간 지시) vs "
                f"워터마크={wm_s:.2f}(AI 지시) — Integrity Clash"
            )

    # Case: 워터마크 없음 + AI 탐지 양성 → 워터마크 손상·제거 의심
    if wm_s is not None and ai_s is not None:
        if wm_s <= AI_LOW and ai_s >= AI_HIGH:
            integrity_clash = True
            clash_details.append(
                f"[워터마크 손상 의심] AI 탐지={ai_s:.2f}(AI 지시)이나 "
                f"워터마크={wm_s:.2f}(없음) — 워터마크 손상·제거 가능성"
            )

    # Case: AI 탐지 모델 단독 음성이나 다른 레이어가 강한 AI 신호
    if ai_s is not None and ai_s <= AI_LOW:
        supporting_ai = [
            name for name, score in (
                ("C2PA", c2pa_s),
                ("워터마크/메타", wm_s),
                ("핑거프린트", fp_s),
            )
            if score is not None and score >= AI_HIGH
        ]
        if len(supporting_ai) >= 2:
            clash_details.append(
                f"[모델 탐지 불일치] AI 탐지 모델={ai_s:.2f}(낮음)이나 "
                f"{', '.join(supporting_ai)} 레이어가 AI 생성 신호를 제시"
            )

    # Case: 출처 레이어는 AI이나 탐지 레이어 평균이 낮음
    if provenance_avg is not None and detection_avg is not None:
        if provenance_avg >= AI_HIGH and detection_avg <= AI_LOW:
            clash_details.append(
                f"[탐지 레이어 약함] 출처 레이어 평균={provenance_avg:.2f}(AI)이나 "
                f"탐지 레이어 평균={detection_avg:.2f}(낮음) — 모델 일반화 한계 가능성"
            )

    # Case: 영상성 탐지 양성이나 출처 인증 부재
    if (
        c2pa_s is None
        and is_video_input
        and ai_s is not None and ai_s >= AI_HIGH
        and (
            (fp_s is not None and fp_s >= AI_HIGH)
            or (rppg_s is not None and rppg_s >= AI_HIGH)
        )
    ):
        clash_details.append(
            "[영상 탐지 보수 판정] 출처 인증 없이 얼굴 탐지/생체·흔적 분석이 양성. "
            "공개 detector의 도메인 편향 가능성을 고려해 확정 판정 대신 정밀 감정 필요로 처리"
        )

    if (
        c2pa_s is None
        and is_video_input
        and ai_s is not None and ai_s >= AI_HIGH
        and fp_s is not None and fp_s >= VIDEO_SOFT_SUPPORT
    ):
        clash_details.append(
            "[영상 탐지 강화 판정] 얼굴 detector가 강한 양성이고 얼굴 fingerprint가 보조 양성. "
            "출처 인증 부재 상태이므로 확정 대신 딥페이크 의심 및 정밀 감정 필요로 처리"
        )

    if (
        c2pa_s is None
        and is_video_input
        and fp_s is not None and fp_s >= VIDEO_SOFT_SUPPORT
        and (
            (ai_s is not None and ai_s >= VIDEO_MID_SUPPORT)
            or (rppg_s is not None and rppg_s >= VIDEO_RPPG_STRONG_SUPPORT)
        )
    ):
        support = []
        if ai_s is not None and ai_s >= VIDEO_MID_SUPPORT:
            support.append(f"얼굴 detector={ai_s:.2f}")
        if rppg_s is not None and rppg_s >= VIDEO_RPPG_STRONG_SUPPORT:
            support.append(f"rPPG={rppg_s:.2f}")
        clash_details.append(
            f"[영상 조합 강화] fingerprint={fp_s:.2f}(강함) + "
            f"{', '.join(support)}(보조) — 출처 인증 부재 상태에서 딥페이크 의심으로 상향"
        )

    # Case: 이미지 탐지 모델과 fingerprint attribution이 함께 양성이나 출처 인증 부재
    if (
        c2pa_s is None
        and not is_video_input
        and ai_s is not None and ai_s >= IMAGE_DETECTOR_STRONG_SUPPORT
        and fp_s is not None and fp_s >= IMAGE_FINGERPRINT_STRONG_SUPPORT
    ):
        clash_details.append(
            "[이미지 탐지 일치] C2PA 출처 인증은 없지만 이미지 AI detector와 "
            "fingerprint attribution이 모두 AI 생성 신호를 제시"
        )

    # Case: 영상 레이어에서 약한 신호가 복수로 모이는 경우
    video_soft_signals = [
        name for name, score in (
            ("얼굴 탐지", ai_s),
            ("rPPG", rppg_s),
            ("얼굴 핑거프린트", fp_s),
        )
        if score is not None and score >= VIDEO_SOFT_SUPPORT
    ]
    if c2pa_s is None and rppg_s is not None and len(video_soft_signals) >= 2:
        clash_details.append(
            f"[영상 약신호 클러스터] {', '.join(video_soft_signals)} 레이어가 "
            "중간 이상 신호를 보여 확정 대신 영상 정밀 감정 권고"
        )

    # Case: 출처 vs 탐지 그룹 방향성 충돌 (cosine similarity)
    if provenance_scores and detection_scores:
        if provenance_avg <= AI_LOW and detection_avg >= AI_HIGH:
            integrity_clash = True
            clash_details.append(
                f"[그룹 방향 충돌] 출처 레이어 평균={provenance_avg:.2f}(인간) vs "
                f"탐지 레이어 평균={detection_avg:.2f}(AI)"
            )

        # 벡터 길이가 동일할 때 cosine similarity 계산
        prov_arr = np.array(provenance_scores)
        det_arr  = np.array(detection_scores)
        if len(prov_arr) == len(det_arr):
            cos_sim = _cosine_similarity(prov_arr, det_arr)
            if cos_sim < CLASH_COS_THRESHOLD:
                integrity_clash = True
                clash_details.append(
                    f"[Cosine Similarity 충돌] 출처-탐지 레이어 간 유사도 "
                    f"{cos_sim:.3f} < 임계값 {CLASH_COS_THRESHOLD}"
                )

    # ── 판정 매트릭스 적용 ──────────────────────────────────────────────────
    verdict, verdict_kr, expert_needed = _apply_judgment_matrix(
        c2pa_s, wm_s, ai_s, rppg_s, fp_s,
        provenance_avg, detection_avg,
        integrity_clash, clash_details, layer_scores, consistency_score,
    )

    # notes
    notes_parts: list[str] = []
    if integrity_clash:
        notes_parts.append("Integrity Clash 감지: 단순 기술 분석으로 진위 판별 불충분, 정밀 감정 요청 권고.")
    elif verdict == "ai_suspected_unverified":
        notes_parts.append("출처 인증 없이 탐지 모듈이 AI/딥페이크 신호를 제시함. 모델 편향 가능성을 고려해 정밀 감정 권고.")
    elif verdict == "video_review_recommended":
        notes_parts.append("영상 레이어에서 복수의 약한 신호가 모였으나 확정 수준은 아니므로 정밀 감정 권고.")
    elif verdict == "image_review_recommended":
        notes_parts.append("이미지 탐지 신호가 있으나 일반 이미지 오탐 가능성을 고려해 확정 대신 정밀 감정을 권고.")
    elif any("이미지 탐지 일치" in d for d in clash_details):
        notes_parts.append("이미지 레이어 3과 레이어 5가 같은 방향의 AI 생성 신호를 제시함.")
    elif any("모델 탐지 불일치" in d for d in clash_details):
        notes_parts.append("최종 판정은 AI 생성 쪽으로 수렴하나, AI 탐지 모델 단독 결과가 낮아 보조 레이어 근거를 함께 제시함.")
    elif consistency_score < LOW_CONSISTENCY_REVIEW:
        notes_parts.append("레이어 간 점수 편차가 커서 결과 해석 시 주의 필요.")
    elif not clash_details:
        notes_parts.append("레이어 간 판정 일관성 확인됨.")
    if not all_scores:
        notes_parts.append("구현된 레이어 데이터 부족 — 1단계 스텁 결과 포함됨.")

    signal_summary = _signal_summary(layer_scores)
    notes_parts.append(
        f"신호 요약: AI={signal_summary['ai']}개, 인간/비AI={signal_summary['human']}개, "
        f"불확실={signal_summary['uncertain']}개, 미실행={signal_summary['missing']}개."
    )

    return CrossLayerResult(
        verdict=verdict,
        verdict_kr=verdict_kr,
        integrity_clash=integrity_clash,
        expert_review_needed=expert_needed,
        consistency_score=consistency_score,
        provenance_avg=provenance_avg,
        detection_avg=detection_avg,
        clash_details=clash_details,
        layer_scores=layer_scores,
        notes=" ".join(notes_parts),
    )


# ─── 내부 헬퍼 ────────────────────────────────────────────────────────────────


def _extract_score(key: str, obj) -> Optional[float]:
    if obj is None:
        return None

    # C2PA manifest 부재는 인간 제작 증명이 아니라 출처 정보 부재다.
    if key == "c2pa" and getattr(obj, "manifest_present", True) is False:
        return None

    # 공개 메타 마커 탐지에서 아무 마커도 없는 저신뢰 결과는
    # 워터마크 부재 증명이 아니므로 종합 판정에서 제외한다.
    if (
        key == "watermark"
        and getattr(obj, "watermark_present", None) is False
        and getattr(obj, "confidence", 1.0) < 0.4
    ):
        return None

    score = getattr(obj, "ai_score", None)
    if score is None:
        return None
    return float(score)


def _valid_scores(layer_scores: dict, keys: tuple) -> list[float]:
    return [layer_scores[k] for k in keys if layer_scores.get(k) is not None]


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 1.0  # 영벡터 → 정의 불가, 충돌 없음으로 처리
    return float(np.dot(v1, v2) / (n1 * n2))


def _compute_consistency(scores: list[float]) -> float:
    """점수 분산으로 일관성 측정 (1 = 완전 일관, 0 = 최대 불일치)."""
    if len(scores) < 2:
        return 1.0
    arr = np.array(scores)
    # 이진 분포 최대 분산 = 0.25
    return float(max(0.0, 1.0 - np.var(arr) / 0.25))


def _apply_judgment_matrix(
    c2pa_s: Optional[float],
    wm_s:   Optional[float],
    ai_s:   Optional[float],
    rppg_s: Optional[float],
    fp_s:   Optional[float],
    provenance_avg: Optional[float],
    detection_avg:  Optional[float],
    integrity_clash: bool,
    clash_details: list[str],
    layer_scores: dict[str, Optional[float]],
    consistency_score: float,
) -> tuple[str, str, bool]:
    """CONTEXT.md 판정 매트릭스 적용."""

    # Integrity Clash 우선 처리
    if integrity_clash:
        # 워터마크 손상 vs 일반 충돌 구분
        watermark_removed = any("워터마크 손상" in d for d in clash_details)
        if watermark_removed:
            return (
                "watermark_compromised",
                "AI 생성 의심 / 워터마크 손상·제거 의심",
                True,
            )
        return (
            "integrity_clash",
            "Integrity Clash — 출처와 탐지 결과 모순: 정밀 감정 필요",
            True,
        )

    summary = _signal_summary(layer_scores)
    strong_ai_layers = [
        name for name, score in layer_scores.items()
        if score is not None and score >= AI_HIGH
    ]
    strong_human_layers = [
        name for name, score in layer_scores.items()
        if score is not None and score <= AI_LOW
    ]
    model_disagreement = any("모델 탐지 불일치" in d for d in clash_details)
    provenance_ai_support = any(
        score is not None and score >= AI_HIGH
        for score in (c2pa_s, wm_s)
    )
    video_soft_signals = [
        score for score in (ai_s, rppg_s, fp_s)
        if score is not None and score >= VIDEO_SOFT_SUPPORT
    ]
    detector_mid = ai_s is not None and ai_s >= VIDEO_MID_SUPPORT
    rppg_weak = rppg_s is not None and rppg_s >= VIDEO_MID_SUPPORT
    rppg_strong = rppg_s is not None and rppg_s >= VIDEO_RPPG_STRONG_SUPPORT
    fingerprint_strong = fp_s is not None and fp_s >= VIDEO_SOFT_SUPPORT

    # 영상 입력에서는 calibration된 얼굴 detector를 1차 분류기로 둔다.
    # rPPG/fingerprint는 보조 레이어라 detector 음성을 뒤집지 않는다.
    if rppg_s is not None and not provenance_ai_support:
        if fingerprint_strong and (detector_mid or rppg_strong):
            return (
                "ai_suspected_unverified",
                "AI/딥페이크 의심 — 영상 fingerprint 강함 + 보조 신호",
                True,
            )
        if detector_mid and fp_s is not None and fp_s >= VIDEO_MID_SUPPORT and rppg_weak:
            return (
                "ai_suspected_unverified",
                "AI/딥페이크 의심 — 영상 복수 중간 신호",
                True,
            )
        if ai_s is not None and ai_s >= AI_HIGH:
            return (
                "ai_suspected_unverified",
                "AI/딥페이크 의심 — 출처 인증 없는 영상 detector 양성",
                True,
            )
        if (
            (ai_s is None or ai_s < VIDEO_MID_SUPPORT)
            and (fp_s is None or fp_s < VIDEO_MID_SUPPORT)
        ):
            return (
                "authentic_likely",
                "진본 가능성 높음 — 영상 detector·fingerprint 음성",
                False,
            )

    # 이미지 입력에서는 general_aigc/fingerprint의 도메인 편향을 고려해
    # 단순 0.60 기준이 아니라 더 강한 detector+fingerprint 동의를 요구한다.
    if rppg_s is None and not provenance_ai_support:
        detector_strong = ai_s is not None and ai_s >= IMAGE_DETECTOR_STRONG_SUPPORT
        fingerprint_image_strong = fp_s is not None and fp_s >= IMAGE_FINGERPRINT_STRONG_SUPPORT
        detector_mid = ai_s is not None and ai_s >= IMAGE_MID_SUPPORT
        fingerprint_mid = fp_s is not None and fp_s >= IMAGE_MID_SUPPORT
        if detector_strong and fingerprint_image_strong:
            return (
                "ai_suspected_unverified",
                "AI/딥페이크 의심 — 이미지 detector와 fingerprint 강한 동의",
                True,
            )
        if (detector_mid and fingerprint_image_strong) or (detector_strong and fingerprint_mid):
            return (
                "image_review_recommended",
                "이미지 정밀 감정 권고 — 탐지 신호는 있으나 단정 기준 미달",
                True,
            )
        if detector_mid and fingerprint_mid:
            return (
                "image_review_recommended",
                "이미지 정밀 감정 권고 — 중간 탐지 신호가 복수 존재",
                True,
            )
        if ai_s is not None and ai_s < 0.55 and (fp_s is None or fp_s < IMAGE_FINGERPRINT_STRONG_SUPPORT):
            return (
                "authentic_likely",
                "진본 가능성 높음 — 이미지 detector 강한 양성 없음",
                False,
            )
        if len(strong_ai_layers) == 1 and len(strong_human_layers) == 0:
            return ("uncertain", "판정 불확실 — 단일 이미지 AI 신호만 존재", True)
        if detector_mid or fingerprint_mid:
            return ("uncertain", "판정 불확실 — 이미지 단일/약한 탐지 신호", True)
        return (
            "authentic_likely",
            "진본 가능성 높음 — 이미지 탐지 신호 약함",
            False,
        )

    if (
        rppg_s is not None
        and ai_s is not None and ai_s >= AI_HIGH
        and fp_s is not None and fp_s >= VIDEO_SOFT_SUPPORT
        and not provenance_ai_support
    ):
        return (
            "ai_suspected_unverified",
            "AI/딥페이크 의심 — 출처 인증 없는 영상 탐지 양성",
            True,
        )

    if rppg_s is not None and len(video_soft_signals) >= 2 and not provenance_ai_support:
        return (
            "video_review_recommended",
            "영상 정밀 감정 권고 — 복수 약한 탐지 신호",
            True,
        )

    # 설명 가능한 조합 규칙 우선 적용
    if summary["ai"] >= 2 and summary["human"] == 0:
        return ("ai_generated_likely", "AI 생성 가능성 높음 — 다수 레이어 일치", False)

    if summary["ai"] >= 3 and summary["human"] >= 1:
        return (
            "ai_generated_with_disagreement",
            "AI 생성 가능성 높음 — 일부 레이어 불일치",
            consistency_score < LOW_CONSISTENCY_REVIEW and not model_disagreement,
        )

    if provenance_avg is not None and detection_avg is not None:
        if provenance_avg >= AI_HIGH and detection_avg >= AI_HIGH:
            return ("ai_generated_likely", "AI 생성 가능성 높음 — 출처·탐지 레이어 일치", False)
        if provenance_avg <= AI_LOW and detection_avg <= AI_LOW:
            return ("authentic_likely", "진본 가능성 높음 — 출처·탐지 레이어 일치", False)

    if c2pa_s is not None and c2pa_s >= AI_HIGH and fp_s is not None and fp_s >= AI_HIGH:
        return (
            "ai_generated_likely",
            "AI 생성 가능성 높음 — C2PA와 핑거프린트 일치",
            False,
        )

    if ai_s is not None and ai_s >= AI_HIGH and fp_s is not None and fp_s >= AI_HIGH:
        return (
            "ai_generated_likely",
            "AI 생성 가능성 높음 — 탐지 모델과 핑거프린트 일치",
            False,
        )

    if len(strong_ai_layers) == 1 and len(strong_human_layers) == 0:
        return ("uncertain", "판정 불확실 — 단일 AI 신호만 존재", True)

    # 판정에 사용할 대표 점수 결정
    representative_scores = [
        s for s in (provenance_avg, detection_avg, c2pa_s, ai_s) if s is not None
    ]

    if not representative_scores:
        return ("insufficient_data", "분석 데이터 불충분 — 추가 레이어 구현 필요", False)

    avg = float(np.mean(representative_scores))

    # CONTEXT.md 매트릭스: 인간 / AI / 불확실
    if avg >= AI_HIGH:
        return ("ai_generated_likely", "AI 생성 가능성 높음", False)
    elif avg <= AI_LOW:
        return ("authentic_likely", "진본 가능성 높음", False)
    else:
        return ("uncertain", "판정 불확실 — 추가 분석 필요", True)


def _signal_summary(layer_scores: dict[str, Optional[float]]) -> dict[str, int]:
    summary = {"ai": 0, "human": 0, "uncertain": 0, "missing": 0}
    for score in layer_scores.values():
        if score is None:
            summary["missing"] += 1
        elif score >= AI_HIGH:
            summary["ai"] += 1
        elif score <= AI_LOW:
            summary["human"] += 1
        else:
            summary["uncertain"] += 1
    return summary
