"""레이어 7: 피해 규모 정량화.

산출식 (CONTEXT.md):
  피해 규모 점수
    = w_dr × (확산 점수 + 재유포 위험 점수)
    + w_is × (식별 위험 점수 + 침해 심각도 점수)
    + w_es × (경제적 피해 점수 + 사회적 영향 점수)

기본 가중치: w_dr=1.5, w_is=1.0, w_es=0.5  →  최대 30점
각 하위 점수: 0~5 범위

GNN 출력 시뮬레이션:
  실제 환경에서는 Bi-GCN (Bian et al., AAAI 2020) 또는
  시간 변화형 GNN (Song et al., IPM 2021)의 출력값을 입력.
  현 단계에서는 소셜미디어 지표를 기반으로 점수를 근사 계산.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─── 피해 등급 ────────────────────────────────────────────────────────────────
_GRADES = [
    (24, "광범위·고위험 피해", "Widespread High-Risk Harm"),
    (16, "심각한 피해",        "Severe Harm"),
    ( 8, "중간 수준 피해",     "Moderate Harm"),
    ( 0, "제한적 피해",        "Limited Harm"),
]

# ─── 범죄 유형별 가중치 ────────────────────────────────────────────────────────
_CRIME_WEIGHTS: dict[str, dict[str, float]] = {
    "deepfake_sexual": {
        "w_dr": 1.5, "w_is": 1.0, "w_es": 0.5,
    },
    "financial_fraud": {
        "w_dr": 1.5, "w_is": 1.0, "w_es": 1.5,  # 경제적 피해 가중치 상향
    },
    "election_manipulation": {
        "w_dr": 1.5, "w_is": 1.0, "w_es": 0.5,
        "w_social_override": 1.5,               # 사회적 영향만 별도 상향
    },
    "default": {
        "w_dr": 1.5, "w_is": 1.0, "w_es": 0.5,
    },
}

GNN_BLEND_WEIGHT = 0.65
HEURISTIC_BLEND_WEIGHT = 0.35
REDISTRIBUTION_BLEND_WEIGHT = 0.70
REDISTRIBUTION_HEURISTIC_WEIGHT = 0.30


# ─── 입력 데이터클래스 ────────────────────────────────────────────────────────

@dataclass
class DamageInputs:
    """피해 규모 계산에 필요한 모든 입력값.

    GNN 관련 필드(확산·재유포)는 실제 소셜미디어 수집 데이터 또는
    수사 기관이 제공한 통계를 입력한다.
    """
    # ── GNN 입력: 확산 점수 ──────────────────────────────────────────────
    num_posts: int = 0              # 게시물 수
    num_platforms: int = 0          # 플랫폼 수
    num_shares: int = 0             # 공유 수
    num_views: int = 0              # 조회수
    spread_speed_hours: float = 0.0 # 1,000 인게이지먼트 도달 소요 시간 (시간)
    platform_names: str = ""        # 발견 플랫폼/커뮤니티 메모
    first_seen_at: str = ""         # 최초 발견 시각/기간
    evidence_source: str = ""       # URL·캡처·제보 경로 메모
    evidence_capture_count: int = 0 # 보전한 URL/캡처 수

    # ── GNN 입력: 재유포 위험 점수 ──────────────────────────────────────
    has_variants: bool = False          # 변형본 존재 여부
    on_closed_platforms: bool = False   # 텔레그램·다크웹 등 폐쇄형 유통
    reappeared_after_deletion: bool = False  # 삭제 후 재등장

    # ── 식별 위험 점수 ────────────────────────────────────────────────────
    face_match_score: float = 0.0   # 피해자 얼굴 일치도 (0~1)
    voice_match_score: float = 0.0  # 피해자 음성 일치도 (0~1)
    victim_identifiable: bool = False # 피해자 특정 가능
    real_name_mentioned: bool = False
    affiliation_revealed: bool = False  # 소속·직장 등 개인정보 노출

    # ── 침해 심각도 점수 ──────────────────────────────────────────────────
    is_sexual_manipulation: bool = False
    is_threatening: bool = False
    is_defamatory: bool = False

    # ── 경제적 피해 점수 ──────────────────────────────────────────────────
    financial_loss_krw: float = 0.0     # 직접 금전 피해 (원)
    reputation_damaged: bool = False    # 기업·직업적 평판 손실

    # ── 사회적 영향 점수 ──────────────────────────────────────────────────
    election_manipulation_risk: float = 0.0  # 선거·여론 조작 위험도 (0~1)
    public_opinion_impact: float = 0.0       # 여론 영향 범위 (0~1)

    @classmethod
    def example_deepfake_sexual(cls) -> "DamageInputs":
        """딥페이크 성범죄 전형적 시나리오 예시."""
        return cls(
            num_posts=80, num_platforms=3, num_shares=1200, num_views=45000,
            spread_speed_hours=4.0,
            platform_names="Telegram, X, online community",
            first_seen_at="사건 접수 전 24시간 이내",
            evidence_source="게시물 URL 2건, 캡처 5건 보전",
            evidence_capture_count=7,
            has_variants=True, on_closed_platforms=True, reappeared_after_deletion=False,
            face_match_score=0.92, voice_match_score=0.0, victim_identifiable=True,
            real_name_mentioned=True, affiliation_revealed=True,
            is_sexual_manipulation=True, is_threatening=False, is_defamatory=True,
            financial_loss_krw=0.0, reputation_damaged=True,
            election_manipulation_risk=0.0, public_opinion_impact=0.15,
        )

    @classmethod
    def example_financial_fraud(cls) -> "DamageInputs":
        """AI 합성 금융사기 시나리오 예시."""
        return cls(
            num_posts=20, num_platforms=2, num_shares=300, num_views=8000,
            spread_speed_hours=12.0,
            platform_names="messenger, email",
            first_seen_at="신고 당일",
            evidence_source="피싱 메시지 캡처 및 송금 내역",
            evidence_capture_count=3,
            has_variants=False, on_closed_platforms=False, reappeared_after_deletion=False,
            face_match_score=0.85, voice_match_score=0.80, victim_identifiable=True,
            real_name_mentioned=True, affiliation_revealed=True,
            is_sexual_manipulation=False, is_threatening=True, is_defamatory=True,
            financial_loss_krw=50_000_000, reputation_damaged=True,
            election_manipulation_risk=0.0, public_opinion_impact=0.05,
        )


# ─── 결과 데이터클래스 ────────────────────────────────────────────────────────

@dataclass
class DamageResult:
    diffusion_score: float          # 확산 점수        (0~5)
    redistribution_score: float     # 재유포 위험 점수  (0~5)
    id_risk_score: float            # 식별 위험 점수    (0~5)
    severity_score: float           # 침해 심각도 점수  (0~5)
    economic_score: float           # 경제적 피해 점수  (0~5)
    social_score: float             # 사회적 영향 점수  (0~5)
    total_score: float              # 최종 합산 점수    (0~30+)
    grade_kr: str                   # 피해 등급 (한국어)
    grade_en: str                   # 피해 등급 (영어)
    crime_type: str
    weights: dict[str, float]
    notes: str
    gnn_risk: Optional[float] = None
    gnn_converted_score: Optional[float] = None
    heuristic_diffusion: float = 0.0
    blend_ratio: str = ""
    gnn_test_acc: Optional[float] = None
    gnn_fallback_reason: Optional[str] = None
    redistribution_learned_prob: Optional[float] = None
    redistribution_converted_score: Optional[float] = None
    redistribution_heuristic_score: float = 0.0
    redistribution_blend_ratio: str = ""
    redistribution_model_type: Optional[str] = None
    redistribution_test_auc: Optional[float] = None
    redistribution_fallback_reason: Optional[str] = None
    spread_context: dict[str, str] = field(default_factory=dict)


# ─── 실행 함수 ────────────────────────────────────────────────────────────────

def run(inputs: DamageInputs, crime_type: str = "default") -> DamageResult:
    """피해 규모 점수 산출."""
    weights = dict(_CRIME_WEIGHTS.get(crime_type, _CRIME_WEIGHTS["default"]))
    heuristic_diffusion = _diffusion_score(inputs)

    # GNN 전파 위험도: 학습된 모델이 있으면 heuristic과 섞어 과도한 포화를 방지
    gnn_risk, gnn_meta, gnn_fallback_reason = (None, None, None)
    if _has_spread_activity(inputs):
        gnn_risk, gnn_meta, gnn_fallback_reason = _try_gnn_spread_risk(inputs)
    else:
        gnn_fallback_reason = "입력 없음"

    if gnn_risk is not None:
        gnn_diffusion = round(gnn_risk * 5.0, 3)
        diffusion = round(GNN_BLEND_WEIGHT * gnn_diffusion + HEURISTIC_BLEND_WEIGHT * heuristic_diffusion, 3)
        gnn_note = (
            f"GNN 전파 위험도={gnn_risk:.3f}, "
            f"GNN 환산={gnn_diffusion:.3f}, heuristic={heuristic_diffusion:.3f}, "
            f"blend={_blend_ratio(GNN_BLEND_WEIGHT, HEURISTIC_BLEND_WEIGHT)} — {_gnn_source_label(gnn_meta)}."
        )
    else:
        gnn_diffusion = None
        diffusion = heuristic_diffusion
        if _has_spread_activity(inputs):
            gnn_note = f"GNN fallback={gnn_fallback_reason or '알 수 없음'} — heuristic 확산 점수 사용."
        else:
            gnn_note = "GNN fallback=입력 없음 — GNN 생략 및 heuristic 확산 점수 0 적용."

    redistrib, redistrib_meta, redistrib_note = _redistribution_score_learned(inputs)
    id_risk    = _id_risk_score(inputs)
    severity   = _severity_score(inputs)
    economic   = _economic_score(inputs)
    social     = _social_score(inputs)

    w_dr = weights["w_dr"]
    w_is = weights["w_is"]
    w_es = weights["w_es"]
    # 사회적 영향 별도 가중치 (election_manipulation 전용)
    w_social = weights.get("w_social_override", w_es)

    total = (
        w_dr * (diffusion + redistrib)
        + w_is * (id_risk + severity)
        + w_es * economic
        + w_social * social
    )
    total = round(total, 2)

    grade_kr, grade_en = _get_grade(total)

    notes_parts: list[str] = [f"범죄 유형: {crime_type}.", gnn_note]
    if redistrib_note:
        notes_parts.append(redistrib_note)
    if total >= 16:
        notes_parts.append("고위험 피해 — 즉각적 증거 보전 및 피해 구제 절차 권고.")
    if inputs.on_closed_platforms:
        notes_parts.append("폐쇄형 플랫폼 유포 확인 — 삭제·차단 집행 어려움.")

    return DamageResult(
        diffusion_score=diffusion,
        redistribution_score=redistrib,
        id_risk_score=id_risk,
        severity_score=severity,
        economic_score=economic,
        social_score=social,
        total_score=total,
        grade_kr=grade_kr,
        grade_en=grade_en,
        crime_type=crime_type,
        weights={"w_dr": w_dr, "w_is": w_is, "w_es": w_es, "w_social": w_social},
        notes=" ".join(notes_parts),
        gnn_risk=round(gnn_risk, 3) if gnn_risk is not None else None,
        gnn_converted_score=gnn_diffusion,
        heuristic_diffusion=heuristic_diffusion,
        blend_ratio=_blend_ratio(GNN_BLEND_WEIGHT, HEURISTIC_BLEND_WEIGHT) if gnn_risk is not None else "heuristic-only",
        gnn_test_acc=_meta_float(gnn_meta, "test_accuracy"),
        gnn_fallback_reason=gnn_fallback_reason if gnn_risk is None else None,
        redistribution_learned_prob=redistrib_meta.get("learned_prob"),
        redistribution_converted_score=redistrib_meta.get("converted_score"),
        redistribution_heuristic_score=redistrib_meta["heuristic_score"],
        redistribution_blend_ratio=redistrib_meta["blend_ratio"],
        redistribution_model_type=redistrib_meta.get("model_type"),
        redistribution_test_auc=redistrib_meta.get("test_auc"),
        redistribution_fallback_reason=redistrib_meta.get("fallback_reason"),
        spread_context=_spread_context(inputs),
    )


def _try_gnn_spread_risk(inputs: DamageInputs) -> tuple[Optional[float], Optional[dict], Optional[str]]:
    """GNN 전파 위험도 (0~1). 실패 시 fallback 원인을 반환한다."""
    try:
        from layers import gnn_spread_model
    except Exception as exc:
        return None, None, f"로딩 실패: {type(exc).__name__}"

    if not Path(gnn_spread_model.MODEL_PATH).exists():
        return None, gnn_spread_model.load_model_metadata(), "모델 없음"

    try:
        risk = gnn_spread_model.predict_spread_risk(inputs)
        if risk is None:
            return None, gnn_spread_model.load_model_metadata(), "모델 없음"
        return risk, gnn_spread_model.load_model_metadata(), None
    except Exception as exc:
        return None, gnn_spread_model.load_model_metadata(), f"로딩 실패: {type(exc).__name__}"


def _gnn_source_label(meta: Optional[dict]) -> str:
    if not meta:
        return "모델 메타데이터 없음"

    source = str(meta.get("data_source", "unknown"))
    if source == "synthetic_risk":
        label = "합성 전파 그래프 기반 GCN"
    elif source == "upfd":
        label = "UPFD 구조 벤치마크 GCN"
    else:
        label = f"{source} GCN"

    test_acc = meta.get("test_accuracy")
    if isinstance(test_acc, (int, float)):
        label += f", test_acc={float(test_acc):.3f}"
    return label


def _redistribution_score_learned(inp: DamageInputs) -> tuple[float, dict, str]:
    heuristic = _redistribution_score(inp)
    meta: dict = {
        "heuristic_score": heuristic,
        "learned_prob": None,
        "converted_score": None,
        "blend_ratio": "heuristic-only",
        "model_type": None,
        "test_auc": None,
        "fallback_reason": None,
    }

    try:
        from layers import redistribution_risk
    except Exception as exc:
        meta["fallback_reason"] = f"로딩 실패: {type(exc).__name__}"
        return heuristic, meta, f"재유포 learned fallback={meta['fallback_reason']} — heuristic 재유포 점수 사용."

    if not redistribution_risk.MODEL_PATH.exists():
        meta["fallback_reason"] = "모델 없음"
        return heuristic, meta, "재유포 learned fallback=모델 없음 — heuristic 재유포 점수 사용."

    try:
        pred = redistribution_risk.predict_redistribution_risk(inp)
    except Exception as exc:
        meta["fallback_reason"] = f"로딩 실패: {type(exc).__name__}"
        return heuristic, meta, f"재유포 learned fallback={meta['fallback_reason']} — heuristic 재유포 점수 사용."

    if pred is None:
        meta["fallback_reason"] = "모델 없음"
        return heuristic, meta, "재유포 learned fallback=모델 없음 — heuristic 재유포 점수 사용."

    converted = pred.converted_score
    score = round(REDISTRIBUTION_BLEND_WEIGHT * converted + REDISTRIBUTION_HEURISTIC_WEIGHT * heuristic, 3)
    meta.update({
        "learned_prob": round(pred.probability, 3),
        "converted_score": converted,
        "blend_ratio": _blend_ratio(REDISTRIBUTION_BLEND_WEIGHT, REDISTRIBUTION_HEURISTIC_WEIGHT),
        "model_type": pred.model_type,
        "test_auc": _meta_float(pred.metadata, "test_auc"),
        "fallback_reason": None,
    })
    note = (
        f"재유포 learned risk={pred.probability:.3f}, 환산={converted:.3f}, "
        f"heuristic={heuristic:.3f}, blend={meta['blend_ratio']}."
    )
    return score, meta, note


# ─── 하위 점수 산출 함수 ──────────────────────────────────────────────────────

def _diffusion_score(inp: DamageInputs) -> float:
    """확산 점수 (0~5) — Bi-GCN 출력 시뮬레이션."""
    if not _has_spread_activity(inp):
        return 0.0

    post_f     = _log_norm(inp.num_posts,   saturation=2000)
    platform_f = min(inp.num_platforms / 8.0, 1.0)
    share_f    = _log_norm(inp.num_shares,  saturation=500_000)
    view_f     = _log_norm(inp.num_views,   saturation=5_000_000)
    # 확산 속도: 빠를수록 1에 가까움 (168시간 = 1주일 기준).
    # 값이 0이고 확산 지표가 존재하면 "미상"으로 보고 중립값을 둔다.
    speed_f = max(0.0, 1.0 - inp.spread_speed_hours / 168.0) if inp.spread_speed_hours > 0 else 0.5

    raw = 0.25 * post_f + 0.20 * platform_f + 0.25 * share_f + 0.20 * view_f + 0.10 * speed_f
    return round(min(raw * 5.0, 5.0), 3)


def _has_spread_activity(inp: DamageInputs) -> bool:
    return any((
        inp.num_posts > 0,
        inp.num_platforms > 0,
        inp.num_shares > 0,
        inp.num_views > 0,
    ))


def _redistribution_score(inp: DamageInputs) -> float:
    """재유포 위험 점수 (0~5) — GNN 출력 시뮬레이션."""
    score = 0.0
    if inp.has_variants:               score += 2.0
    if inp.on_closed_platforms:        score += 2.0
    if inp.reappeared_after_deletion:  score += 1.0
    return round(min(score, 5.0), 3)


def _blend_ratio(primary: float, secondary: float) -> str:
    return f"{int(round(primary * 100))}:{int(round(secondary * 100))}"


def _meta_float(meta: Optional[dict], key: str) -> Optional[float]:
    if not meta:
        return None
    value = meta.get(key)
    if isinstance(value, (int, float)):
        return round(float(value), 3)
    return None


def _id_risk_score(inp: DamageInputs) -> float:
    """식별 위험 점수 (0~5)."""
    score = inp.face_match_score * 2.0 + inp.voice_match_score * 1.0
    if inp.victim_identifiable: score += 0.7
    if inp.real_name_mentioned:   score += 1.0
    if inp.affiliation_revealed:  score += 1.0
    return round(min(score, 5.0), 3)


def _severity_score(inp: DamageInputs) -> float:
    """침해 심각도 점수 (0~5)."""
    score = 0.0
    if inp.is_sexual_manipulation:  score += 3.0
    if inp.is_threatening:          score += 1.5
    if inp.is_defamatory:           score += 1.0
    return round(min(score, 5.0), 3)


def _economic_score(inp: DamageInputs) -> float:
    """경제적 피해 점수 (0~5)."""
    # 1억 원에서 포화 (log 스케일)
    loss_f = _log_norm(inp.financial_loss_krw, saturation=100_000_000)
    score  = loss_f * 4.0 + (1.0 if inp.reputation_damaged else 0.0)
    return round(min(score, 5.0), 3)


def _social_score(inp: DamageInputs) -> float:
    """사회적 영향 점수 (0~5)."""
    score = inp.election_manipulation_risk * 2.5 + inp.public_opinion_impact * 2.5
    return round(min(score, 5.0), 3)


def _log_norm(value: float, saturation: float) -> float:
    """log1p 기반 0~1 정규화."""
    if saturation <= 0 or value <= 0:
        return 0.0
    return min(math.log1p(value) / math.log1p(saturation), 1.0)


def _get_grade(score: float) -> tuple[str, str]:
    for threshold, kr, en in _GRADES:
        if score >= threshold:
            return kr, en
    return _GRADES[-1][1], _GRADES[-1][2]


def _spread_context(inp: DamageInputs) -> dict[str, str]:
    return {
        "platform_names": inp.platform_names.strip(),
        "first_seen_at": inp.first_seen_at.strip(),
        "evidence_source": inp.evidence_source.strip(),
        "evidence_capture_count": str(int(inp.evidence_capture_count)),
        "victim_identifiable": "예" if inp.victim_identifiable else "아니오",
    }
