"""CAVE 로컬 웹 데모.

실행:
  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from layers import (
    ai_detection,
    c2pa_check,
    cross_layer_audit,
    damage_score,
    fingerprint,
    rppg_check,
    watermark_check,
)


CRIME_TYPES = {
    "기본": "default",
    "딥페이크 성범죄": "deepfake_sexual",
    "AI 금융사기": "financial_fraud",
    "선거·여론 조작": "election_manipulation",
}


def main() -> None:
    st.set_page_config(page_title="CAVE Demo", page_icon="CAVE", layout="wide")
    _inject_css()

    _render_header()

    with st.sidebar:
        st.header("분석 설정")
        uploaded = st.file_uploader(
            "이미지 또는 영상 파일",
            type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi", "mkv", "webm"],
        )
        crime_label = st.selectbox("범죄 유형", list(CRIME_TYPES.keys()), index=1)
        use_demo_damage = st.toggle("유포 정황 데모값 사용", value=True)

        if use_demo_damage:
            st.caption("선택한 범죄 유형에 맞는 예시 유포 정황을 GNN 피해산정 입력으로 사용합니다.")
            damage_inputs = _demo_damage_inputs(CRIME_TYPES[crime_label])
            with st.expander("데모 유포 정황", expanded=False):
                st.markdown(_rows_to_markdown_table(_damage_input_rows(damage_inputs)))
        else:
            damage_inputs = _damage_form(CRIME_TYPES[crime_label])

        analyze = st.button("분석 실행", type="primary", use_container_width=True)

    if uploaded is None:
        _empty_state()
        return

    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        _section_heading("감정 대상")
        if _is_video_file(uploaded.name):
            st.video(uploaded)
        else:
            st.image(uploaded, use_container_width=True)
        _file_meta(uploaded.name, uploaded.size)

    if not analyze:
        with right:
            st.info("왼쪽 설정을 확인한 뒤 분석 실행을 누르세요.")
        return

    suffix = Path(uploaded.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        file_path = tmp.name

    try:
        results = _run_analysis(
            file_path=file_path,
            original_name=uploaded.name,
            crime_type=CRIME_TYPES[crime_label],
            damage_inputs=damage_inputs,
        )
    except Exception as exc:
        st.error(f"분석 중 오류가 발생했습니다: {type(exc).__name__}: {exc}")
        return

    with right:
        _render_summary(results)

    st.divider()
    _render_layers(results)


def _run_analysis(
    file_path: str,
    original_name: str,
    crime_type: str,
    damage_inputs: damage_score.DamageInputs,
) -> dict:
    progress = st.progress(0, text="출처 인증 검사 중...")

    file_hash = _sha256(file_path)
    l1 = c2pa_check.run(file_path)
    progress.progress(15, text="워터마크·메타 신호 검사 중...")

    l2 = watermark_check.run(file_path)
    progress.progress(25, text="AI 생성 탐지 모델 실행 중...")

    l3 = ai_detection.run(file_path)
    progress.progress(55, text="생체 신호 일관성 검사 중...")

    l4 = rppg_check.run(file_path)
    progress.progress(65, text="생성 모델 흔적 분석 중...")

    l5 = fingerprint.run(file_path)
    progress.progress(78, text="다층 증거 일관성 감사 중...")

    l6 = cross_layer_audit.run({
        "c2pa": l1,
        "watermark": l2,
        "ai_detection": l3,
        "rppg": l4,
        "fingerprint": l5,
    })
    progress.progress(88, text="피해 규모 산정 중...")

    l7 = damage_score.run(damage_inputs, crime_type=crime_type)
    progress.progress(100, text="분석 완료")

    return {
        "original_name": original_name,
        "file_path": file_path,
        "file_hash": file_hash,
        "crime_type": crime_type,
        "c2pa": l1,
        "watermark": l2,
        "ai_detection": l3,
        "rppg": l4,
        "fingerprint": l5,
        "audit": l6,
        "damage": l7,
        "damage_inputs": damage_inputs,
    }


def _render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
          <div>
            <div class="app-kicker">Credibility Audit for AI-Generated Evidence</div>
            <h1>CAVE</h1>
            <p>AI 생성물 증거 검증 및 피해 정량화 로컬 데모</p>
          </div>
          <div class="app-header-badge">Layered Evidence Audit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section_heading(title: str) -> None:
    st.markdown(f'<div class="section-heading">{title}</div>', unsafe_allow_html=True)


def _file_meta(filename: str, size: int) -> None:
    st.markdown(
        f"""
        <div class="file-meta">
          <span>파일명</span><strong>{_escape_html(filename)}</strong>
          <span>크기</span><strong>{size / 1024:.1f} KB</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_summary(results: dict) -> None:
    audit = results["audit"]
    damage = results["damage"]

    _section_heading("종합 판정")
    verdict_class = "danger" if audit.integrity_clash else "ok"
    if "불확실" in audit.verdict_kr or audit.expert_review_needed:
        verdict_class = "warn"

    st.markdown(
        f"""
        <div class="verdict {verdict_class}">
          <div class="label">최종 의견</div>
          <div class="value">{audit.verdict_kr}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    cols[0].metric("Integrity Clash", "감지" if audit.integrity_clash else "없음")
    cols[1].metric("일관성", f"{audit.consistency_score:.3f}")
    cols[2].metric("정밀 감정", "필요" if audit.expert_review_needed else "불필요")
    cols[3].metric("피해 점수", f"{damage.total_score:.1f} / 30")

    st.markdown('<div class="mini-heading">핵심 근거</div>', unsafe_allow_html=True)
    _finding_list(_key_findings(results))

    review_label, review_body = _review_brief(audit)
    if audit.expert_review_needed:
        st.warning(f"**{review_label}**  \n{review_body}")
    else:
        st.success(f"**{review_label}**  \n{review_body}")

    st.markdown('<div class="mini-heading">피해·확산 근거</div>', unsafe_allow_html=True)
    damage_cols = st.columns(3)
    damage_cols[0].metric("피해 등급", damage.grade_kr)
    damage_cols[1].metric("GNN 전파 위험도", _fmt_score(getattr(damage, "gnn_risk", None)))
    damage_cols[2].metric("재유포 learned", _fmt_score(getattr(damage, "redistribution_learned_prob", None)))
    st.caption(_damage_brief(damage))


def _render_layers(results: dict) -> None:
    _section_heading("세부 분석")

    c2pa = results["c2pa"]
    watermark = results["watermark"]
    ai_det = results["ai_detection"]
    fp = results["fingerprint"]
    rppg = results["rppg"]
    audit = results["audit"]
    damage = results["damage"]
    damage_inputs = results.get("damage_inputs")

    tab_evidence, tab_damage = st.tabs(["증거 레이어", "피해 산정"])

    with tab_evidence:
        _render_evidence_layers(c2pa, watermark, ai_det, rppg, fp, audit)

    with tab_damage:
        _render_damage_layer(damage, damage_inputs)


def _render_evidence_layers(c2pa, watermark, ai_det, rppg, fp, audit) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        _result_card(
            "출처 인증 검사",
            [
                ("매니페스트", "존재" if c2pa.manifest_present else "없음"),
                ("서명", c2pa.signature_valid),
                ("AI 선언", "예" if c2pa.ai_usage_declared else "아니오"),
                ("생성 도구", c2pa.claim_generator or "정보 없음"),
                ("AI 점수", f"{c2pa.ai_score:.3f}"),
            ],
            c2pa.notes,
        )

    with col2:
        _result_card(
            "워터마크·메타 신호 검사",
            [
                ("워터마크", "감지" if watermark.watermark_present else "없음"),
                ("AI 마커", "감지" if watermark.ai_watermark else "없음"),
                ("신뢰도", _fmt_score(watermark.confidence)),
                ("AI 점수", _fmt_score(watermark.ai_score)),
            ],
            watermark.notes,
        )

    with col3:
        _result_card(
            "AI 생성 탐지 모델",
            [
                ("AI 확률", _fmt_score(ai_det.ai_probability)),
                ("판정", ai_det.verdict or "미실행"),
                ("얼굴 합성", ai_det.face_synthesis_level or "N/A"),
                ("AI 점수", _fmt_score(ai_det.ai_score)),
            ],
            ai_det.notes,
        )

    col_rppg, col_fp, col_audit = st.columns(3)
    with col_rppg:
        rppg_rows = [
            ("분석 모드", _rppg_analysis_mode(rppg)),
            ("심박 신호 자연스러움", _fmt_score(rppg.heartbeat_naturalness)),
            ("생체 신호 일치", "예" if rppg.biometric_match else ("아니오" if rppg.biometric_match is not None else "N/A")),
            ("AI 점수", _fmt_score(rppg.ai_score)),
        ]
        rppg_rows.extend(_rppg_evidence_rows(rppg))
        _result_card(
            "생체 신호 일관성 검사",
            rppg_rows,
            rppg.notes,
        )

    with col_fp:
        fp_rows = [
            ("분석 방식", _fingerprint_analysis_mode(fp)),
            ("AI 가능성", fp.ai_likelihood or "미실행"),
            ("조작 계열", fp.generation_method or "N/A"),
            ("Attribution", fp.model_family or "N/A"),
            ("신뢰도", _fmt_score(fp.confidence)),
            ("AI 점수", _fmt_score(fp.ai_score)),
        ]
        fp_rows.extend(_fingerprint_evidence_rows(fp))
        _result_card(
            "생성 모델 흔적 attribution",
            fp_rows,
            fp.notes,
        )

    with col_audit:
        audit_rows = [
            ("판정", audit.verdict_kr),
            ("Clash", "감지" if audit.integrity_clash else "없음"),
            ("출처 평균", _fmt_score(audit.provenance_avg)),
            ("탐지 평균", _fmt_score(audit.detection_avg)),
            ("일관성", f"{audit.consistency_score:.3f}"),
        ]
        if getattr(rppg, "ai_score", None) is not None:
            audit_rows.append(("영상 신호 조합", _video_signal_summary(ai_det, rppg, fp)))
        _result_card(
            "다층 증거 일관성 감사",
            audit_rows,
            audit.notes,
        )
        if audit.clash_details:
            st.warning("\n".join(audit.clash_details))


def _render_damage_layer(damage, damage_inputs=None) -> None:
    damage_rows = [
        ("확산", f"{damage.diffusion_score:.2f} / 5"),
        ("재유포", f"{damage.redistribution_score:.2f} / 5"),
        ("식별 위험", f"{damage.id_risk_score:.2f} / 5"),
        ("침해 심각도", f"{damage.severity_score:.2f} / 5"),
        ("경제·사회", f"{damage.economic_score:.2f} / {damage.social_score:.2f}"),
        ("총점", f"{damage.total_score:.1f} / 30"),
    ]
    damage_rows.extend(_damage_evidence_rows(damage))
    _result_card(
        "피해 규모 산정",
        damage_rows,
        damage.notes,
    )
    if damage_inputs is not None:
        _result_card(
            "유포 정황 입력 요약",
            _damage_input_rows(damage_inputs),
            "이미지·영상 자체의 픽셀 분석값이 아니라, 발견 경로와 확산 정황을 GNN 피해산정 그래프 구성에 반영한 입력값입니다.",
        )


def _result_card(title: str, rows: list[tuple[str, str]], note: str) -> None:
    st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(_rows_to_markdown_table(rows))
    with st.expander("분석 메모"):
        st.write(note)


def _damage_form(crime_type: str) -> damage_score.DamageInputs:
    st.subheader("유포 정황 입력")
    st.caption("이미지 단독 판단이 아니라 발견·확산·재유포 정황을 함께 넣어 GNN 피해산정을 수행합니다.")

    is_deepfake_sexual = crime_type == "deepfake_sexual"
    is_financial = crime_type == "financial_fraud"
    is_election = crime_type == "election_manipulation"

    with st.expander("1. 발견·증거 보전", expanded=True):
        first_seen_at = st.text_input(
            "최초 발견 시각/기간",
            placeholder="예: 2026-05-19 13:20 또는 최초 발견 후 24시간 이내",
        )
        platform_options = [
            "X",
            "Instagram",
            "TikTok",
            "YouTube",
            "Telegram",
            "Discord",
            "온라인 커뮤니티",
            "메신저",
            "이메일",
            "다크웹/비공개 포럼",
        ]
        default_platforms = ["Telegram", "온라인 커뮤니티"] if is_deepfake_sexual else []
        selected_platforms = st.multiselect("발견 플랫폼", platform_options, default=default_platforms)
        extra_platforms = st.text_input("기타 플랫폼/커뮤니티", placeholder="쉼표로 구분")
        platform_names = _join_platforms(selected_platforms, extra_platforms)
        evidence_source = st.text_area(
            "URL·캡처·제보 경로 메모",
            placeholder="예: 원 게시물 URL 1건, 재업로드 캡처 3건, 피해자 제보",
            height=76,
        )
        evidence_capture_count = st.number_input("보전한 URL/캡처 수", min_value=0, value=0)

    with st.expander("2. 확산 규모", expanded=True):
        num_posts = st.number_input("동일·유사 게시물 수", min_value=0, value=0)
        derived_platform_count = _count_platforms(platform_names)
        num_platforms = st.number_input("플랫폼 수", min_value=0, value=derived_platform_count)
        num_shares = st.number_input("공유·리포스트 수", min_value=0, value=0)
        num_views = st.number_input("조회수", min_value=0, value=0)
        spread_speed_hours = st.number_input(
            "1,000 인게이지먼트 도달 시간",
            min_value=0.0,
            value=0.0,
            help="값이 작을수록 빠른 확산으로 해석됩니다. 모르면 0으로 둡니다.",
        )

    with st.expander("3. 재유포 위험", expanded=True):
        has_variants = st.checkbox("변형본 존재", value=is_deepfake_sexual)
        on_closed_platforms = st.checkbox("폐쇄형 플랫폼 유포", value=is_deepfake_sexual)
        reappeared_after_deletion = st.checkbox("삭제 후 재등장")

    with st.expander("4. 피해자 특정성", expanded=True):
        face_match_score = st.slider("피해자 얼굴 동일성 점수", 0.0, 1.0, 0.0)
        voice_match_score = st.slider("피해자 음성 동일성 점수", 0.0, 1.0, 0.0)
        victim_identifiable = st.checkbox("피해자 특정 가능", value=is_deepfake_sexual)
        real_name_mentioned = st.checkbox("실명 언급")
        affiliation_revealed = st.checkbox("학교·직장 등 소속 정보 노출")

    with st.expander("5. 침해·사회 영향", expanded=True):
        is_sexual_manipulation = st.checkbox("성적 조작 콘텐츠", value=is_deepfake_sexual)
        is_threatening = st.checkbox("협박성", value=is_financial)
        is_defamatory = st.checkbox("명예훼손성", value=is_deepfake_sexual)
        financial_loss_krw = st.number_input(
            "직접 금전 피해액",
            min_value=0.0,
            value=50_000_000.0 if is_financial else 0.0,
            step=1_000_000.0,
        )
        reputation_damaged = st.checkbox("평판·직업상 손실", value=is_deepfake_sexual or is_financial)
        election_manipulation_risk = st.slider("선거 조작 위험도", 0.0, 1.0, 0.7 if is_election else 0.0)
        public_opinion_impact = st.slider("여론 영향 범위", 0.0, 1.0, 0.5 if is_election else 0.0)

    return damage_score.DamageInputs(
        num_posts=int(num_posts),
        num_platforms=int(num_platforms),
        num_shares=int(num_shares),
        num_views=int(num_views),
        spread_speed_hours=float(spread_speed_hours),
        platform_names=platform_names,
        first_seen_at=first_seen_at,
        evidence_source=evidence_source,
        evidence_capture_count=int(evidence_capture_count),
        has_variants=has_variants,
        on_closed_platforms=on_closed_platforms,
        reappeared_after_deletion=reappeared_after_deletion,
        face_match_score=face_match_score,
        voice_match_score=voice_match_score,
        victim_identifiable=victim_identifiable,
        real_name_mentioned=real_name_mentioned,
        affiliation_revealed=affiliation_revealed,
        is_sexual_manipulation=is_sexual_manipulation,
        is_threatening=is_threatening,
        is_defamatory=is_defamatory,
        financial_loss_krw=financial_loss_krw,
        reputation_damaged=reputation_damaged,
        election_manipulation_risk=election_manipulation_risk,
        public_opinion_impact=public_opinion_impact,
    )


def _demo_damage_inputs(crime_type: str) -> damage_score.DamageInputs:
    if crime_type == "financial_fraud":
        return damage_score.DamageInputs.example_financial_fraud()
    if crime_type == "deepfake_sexual":
        return damage_score.DamageInputs.example_deepfake_sexual()
    return damage_score.DamageInputs()


def _join_platforms(selected: list[str], extra: str) -> str:
    values = [item.strip() for item in selected if item.strip()]
    values.extend(item.strip() for item in extra.split(",") if item.strip())
    return ", ".join(dict.fromkeys(values))


def _count_platforms(platform_names: str) -> int:
    if not platform_names.strip():
        return 0
    return len([part for part in platform_names.split(",") if part.strip()])


def _damage_input_rows(inputs: damage_score.DamageInputs) -> list[tuple[str, str]]:
    return [
        ("발견 플랫폼", inputs.platform_names or "미입력"),
        ("최초 발견", inputs.first_seen_at or "미입력"),
        ("보전 자료", inputs.evidence_source or "미입력"),
        ("URL/캡처 수", str(inputs.evidence_capture_count)),
        ("게시물/플랫폼", f"{inputs.num_posts}건 / {inputs.num_platforms}개"),
        ("공유/조회", f"{inputs.num_shares}회 / {inputs.num_views}회"),
        ("확산 속도", f"{inputs.spread_speed_hours:.1f}시간"),
        ("변형·폐쇄·재등장", _bool_chain([
            ("변형본", inputs.has_variants),
            ("폐쇄형", inputs.on_closed_platforms),
            ("재등장", inputs.reappeared_after_deletion),
        ])),
        ("피해자 특정성", _bool_chain([
            ("특정 가능", inputs.victim_identifiable),
            ("실명", inputs.real_name_mentioned),
            ("소속", inputs.affiliation_revealed),
        ])),
        ("얼굴/음성 동일성", f"{inputs.face_match_score:.2f} / {inputs.voice_match_score:.2f}"),
        ("침해 성격", _bool_chain([
            ("성적 조작", inputs.is_sexual_manipulation),
            ("협박성", inputs.is_threatening),
            ("명예훼손성", inputs.is_defamatory),
        ])),
    ]


def _bool_chain(items: list[tuple[str, bool]]) -> str:
    active = [label for label, enabled in items if enabled]
    return ", ".join(active) if active else "해당 없음"


def _empty_state() -> None:
    st.info("왼쪽 사이드바에서 이미지 또는 영상 파일을 업로드하면 분석을 시작할 수 있습니다.")
    st.markdown(
        """
        현재 구현된 분석 모듈:
        - 출처 인증 검사
        - 워터마크·메타 신호 검사
        - AI 생성 탐지 모델
        - 생성 모델 흔적 분석
        - 다층 증거 일관성 감사
        - 피해 규모 산정
        """
    )


def _sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt_score(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def _finding_list(findings: list[str]) -> None:
    items = "".join(f"<li>{_escape_html(finding)}</li>" for finding in findings)
    st.markdown(f'<ul class="finding-list">{items}</ul>', unsafe_allow_html=True)


def _key_findings(results: dict) -> list[str]:
    c2pa = results["c2pa"]
    watermark = results["watermark"]
    ai_det = results["ai_detection"]
    fp = results["fingerprint"]
    rppg = results["rppg"]
    audit = results["audit"]
    damage = results["damage"]

    findings: list[str] = []
    if getattr(c2pa, "manifest_present", False):
        findings.append("C2PA 출처 매니페스트가 존재해 출처 레이어 근거가 확인됩니다.")
    else:
        findings.append("C2PA 출처 매니페스트가 없어 탐지 레이어와 교차 검증이 필요합니다.")

    ai_score = getattr(ai_det, "ai_score", None)
    if ai_score is not None:
        findings.append(f"AI 탐지 모델 점수는 {_fmt_score(ai_score)}이며 판정은 `{ai_det.verdict or '미실행'}`입니다.")

    fp_score = getattr(fp, "ai_score", None)
    if fp_score is not None:
        method = getattr(fp, "generation_method", None) or "unknown"
        findings.append(f"생성 흔적 attribution은 {_fmt_score(fp_score)}로 `{method}` 계열 신호를 제시합니다.")

    if getattr(rppg, "ai_score", None) is not None:
        findings.append(_video_signal_summary(ai_det, rppg, fp))

    if getattr(watermark, "watermark_present", False) or getattr(watermark, "ai_watermark", False):
        findings.append(f"워터마크·메타 신호 점수는 {_fmt_score(getattr(watermark, 'ai_score', None))}입니다.")

    findings.append(
        f"피해 규모는 {damage.total_score:.1f}/30점, `{damage.grade_kr}`이며 "
        f"GNN 전파 위험도는 {_fmt_score(getattr(damage, 'gnn_risk', None))}입니다."
    )

    if audit.clash_details:
        findings.append(_clean_detail(audit.clash_details[0]))

    return findings[:3]


def _review_brief(audit) -> tuple[str, str]:
    if audit.integrity_clash:
        detail = _clean_detail(audit.clash_details[0]) if audit.clash_details else "출처와 탐지 결과가 서로 충돌합니다."
        return "정밀 감정 필요", detail
    if audit.expert_review_needed:
        if audit.clash_details:
            return "정밀 감정 권고", _clean_detail(audit.clash_details[0])
        return "정밀 감정 권고", "출처 인증 또는 탐지 신호가 확정 수준에 이르지 않아 전문가 검토가 필요합니다."
    return "정밀 감정 불필요", "현재 레이어 조합에서는 강한 충돌 없이 판정 방향이 일관됩니다."


def _damage_brief(damage) -> str:
    gnn_risk = getattr(damage, "gnn_risk", None)
    if gnn_risk is None:
        return f"GNN fallback={getattr(damage, 'gnn_fallback_reason', 'N/A')} · 총점 {damage.total_score:.1f}/30"
    return (
        f"GNN 전파 위험도 {_fmt_score(gnn_risk)} · "
        f"GNN 환산 {_fmt_score(getattr(damage, 'gnn_converted_score', None))} / "
        f"heuristic {_fmt_score(getattr(damage, 'heuristic_diffusion', None))} / "
        f"blend {getattr(damage, 'blend_ratio', '')} · "
        f"재유포 learned {_fmt_score(getattr(damage, 'redistribution_learned_prob', None))}"
    )


def _clean_detail(detail: str) -> str:
    text = detail.strip()
    if text.startswith("[") and "]" in text:
        text = text.split("]", 1)[1].strip()
    return text


def _rows_to_markdown_table(rows: list[tuple[str, str]]) -> str:
    table = ["| 항목 | 값 |", "|---|---|"]
    for label, value in rows:
        table.append(f"| {label} | {_escape_table(value)} |")
    return "\n".join(table)


def _escape_table(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _escape_html(value) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _damage_evidence_rows(result) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    gnn_risk = getattr(result, "gnn_risk", None)
    if gnn_risk is not None:
        rows.append(("GNN 전파 위험도", _fmt_score(gnn_risk)))
        rows.append((
            "GNN 확산 근거",
            (
                f"GNN 환산 {_fmt_score(getattr(result, 'gnn_converted_score', None))} / "
                f"heuristic {_fmt_score(getattr(result, 'heuristic_diffusion', None))} / "
                f"blend {getattr(result, 'blend_ratio', '')}"
            ),
        ))
        rows.append(("GNN test acc", _fmt_score(getattr(result, "gnn_test_acc", None))))
    else:
        reason = getattr(result, "gnn_fallback_reason", None)
        if reason:
            rows.append(("GNN fallback", str(reason)))

    learned_prob = getattr(result, "redistribution_learned_prob", None)
    if learned_prob is not None:
        rows.append(("재유포 learned prob", _fmt_score(learned_prob)))
        rows.append((
            "재유포 근거",
            (
                f"learned 환산 {_fmt_score(getattr(result, 'redistribution_converted_score', None))} / "
                f"heuristic {_fmt_score(getattr(result, 'redistribution_heuristic_score', None))} / "
                f"blend {getattr(result, 'redistribution_blend_ratio', '')}"
            ),
        ))
        rows.append(("재유포 model", str(getattr(result, "redistribution_model_type", "") or "N/A")))
        rows.append(("재유포 test AUC", _fmt_score(getattr(result, "redistribution_test_auc", None))))
    else:
        reason = getattr(result, "redistribution_fallback_reason", None)
        if reason:
            rows.append(("재유포 fallback", str(reason)))
    return rows


def _fingerprint_analysis_mode(result) -> str:
    notes = getattr(result, "notes", "") or ""
    if "학습 기반 video fingerprint classifier" in notes:
        return "학습 기반 영상 fingerprint classifier"
    if "image fingerprint ensemble" in notes:
        return "GenImage + RedFace 이미지 ensemble"
    if "GenImage fingerprint classifier" in notes:
        return "GenImage generator fingerprint"
    if "학습 기반 fingerprint classifier" in notes:
        return "학습 기반 이미지 attribution"
    if "learned_classifier=image-trained" in notes:
        return "영상 heuristic (이미지 classifier 미적용)"
    if "영상용 learned fingerprint classifier 없음" in notes:
        return "영상 heuristic fallback"
    if "학습 classifier 없음" in notes or "FFT/노이즈 기반 보조 분석" in notes:
        return "FFT/노이즈 heuristic fallback"
    if getattr(result, "ai_score", None) is None:
        return "미실행"
    return "fingerprint 분석"


def _rppg_analysis_mode(result) -> str:
    if getattr(result, "ai_score", None) is None:
        return "미실행"
    notes = getattr(result, "notes", "") or ""
    if "학습 기반 rPPG feature classifier" in notes:
        return "RF rPPG feature classifier"
    if "CHROM rPPG" in notes:
        return "CHROM rPPG heuristic"
    return getattr(result, "analysis_mode", "none")


def _rppg_evidence_rows(result) -> list[tuple[str, str]]:
    evidence = getattr(result, "evidence", {}) or {}
    keys = [
        ("learned_prob", "RF prob"),
        ("calibrated_prob", "보정 확률"),
        ("threshold", "rPPG threshold"),
        ("heuristic", "CHROM heuristic"),
        ("face_detected", "Face detected"),
        ("peak_bpm", "Peak BPM"),
        ("test_auc", "Layer 4 AUC"),
    ]
    return [
        (label, str(evidence[key]))
        for key, label in keys
        if evidence.get(key) not in (None, "", "N/A")
    ]


def _fingerprint_evidence_rows(result) -> list[tuple[str, str]]:
    evidence = getattr(result, "evidence", {}) or {}
    keys = [
        ("learned_prob", "Learned prob"),
        ("genimage_prob", "GenImage prob"),
        ("redface_prob", "RedFace prob"),
        ("generator", "Generator"),
        ("redface_method", "RedFace method"),
        ("calibrated_prob", "보정 확률"),
        ("threshold", "영상 threshold"),
        ("heuristic", "Heuristic"),
        ("consensus", "Consensus"),
        ("face_detected", "Face crop"),
        ("temporal_delta", "Temporal delta"),
        ("genimage_auc", "GenImage AUC"),
        ("redface_auc", "RedFace AUC"),
        ("test_auc", "Layer 5 AUC"),
    ]
    return [
        (label, str(evidence[key]))
        for key, label in keys
        if evidence.get(key) not in (None, "", "N/A")
    ]


def _video_signal_summary(ai_det, rppg, fp) -> str:
    detector = _signal_level(getattr(ai_det, "ai_score", None))
    rppg_level = _signal_level(getattr(rppg, "ai_score", None))
    fingerprint = _signal_level(getattr(fp, "ai_score", None))

    ai_score = getattr(ai_det, "ai_score", None)
    rppg_score = getattr(rppg, "ai_score", None)
    fp_score = getattr(fp, "ai_score", None)
    mid_count = sum(1 for score in (ai_score, rppg_score, fp_score) if score is not None and score >= 0.45)

    if fp_score is not None and fp_score >= 0.60 and (
        (ai_score is not None and ai_score >= 0.45)
        or (rppg_score is not None and rppg_score >= 0.70)
    ):
        decision = "딥페이크 의심 상향"
    elif ai_score is not None and ai_score >= 0.60 and fp_score is not None and fp_score >= 0.60:
        decision = "정밀감정 권고"
    elif mid_count >= 2:
        decision = "복수 보조 신호"
    elif ai_score is not None and ai_score < 0.45 and (fp_score is None or fp_score < 0.45):
        decision = "진본 가능성 우세"
    else:
        decision = "보조 검토"

    return f"Detector {detector} + rPPG {rppg_level} + Fingerprint {fingerprint} -> {decision}"


def _signal_level(score: float | None) -> str:
    if score is None:
        return "미실행"
    if score >= 0.60:
        return "강함"
    if score >= 0.45:
        return "중간"
    return "낮음"


def _is_video_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --cave-bg: #f6f8fb;
            --cave-surface: #ffffff;
            --cave-surface-alt: #eef3f7;
            --cave-border: #d7dee8;
            --cave-text: #172033;
            --cave-muted: #607086;
            --cave-teal: #0f766e;
            --cave-amber: #a16207;
            --cave-red: #b42318;
            --cave-indigo: #334155;
        }
        .stApp {
            background: var(--cave-bg);
            color: var(--cave-text);
        }
        header[data-testid="stHeader"] {
            height: 0;
            visibility: hidden;
        }
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        #MainMenu,
        footer {
            display: none;
            visibility: hidden;
        }
        .block-container {
            max-width: 1280px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebarContent"] {
            padding-top: 2.1rem;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--cave-border);
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--cave-text);
        }
        .app-header {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
            background: var(--cave-surface);
            border: 1px solid var(--cave-border);
            border-left: 5px solid var(--cave-teal);
            border-radius: 8px;
            padding: 18px 20px 16px;
            margin-bottom: 18px;
        }
        .app-header h1 {
            margin: 2px 0 2px;
            font-size: 2.1rem;
            line-height: 1.05;
            letter-spacing: 0;
            color: var(--cave-text);
        }
        .app-header p {
            margin: 0;
            color: var(--cave-muted);
            font-size: 0.98rem;
        }
        .app-kicker {
            color: var(--cave-teal);
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0;
        }
        .app-header-badge {
            border: 1px solid #bfd8d5;
            background: #eff8f6;
            color: #0b5f58;
            border-radius: 999px;
            padding: 7px 10px;
            font-size: 0.78rem;
            font-weight: 700;
            white-space: nowrap;
        }
        .section-heading {
            margin: 0.15rem 0 0.65rem;
            color: var(--cave-text);
            font-size: 1.08rem;
            font-weight: 760;
            letter-spacing: 0;
        }
        .mini-heading {
            margin: 1rem 0 0.35rem;
            color: var(--cave-text);
            font-size: 0.95rem;
            font-weight: 760;
            letter-spacing: 0;
        }
        .file-meta {
            display: grid;
            grid-template-columns: max-content minmax(0, 1fr);
            gap: 5px 10px;
            margin-top: 10px;
            padding: 11px 12px;
            background: var(--cave-surface);
            border: 1px solid var(--cave-border);
            border-radius: 8px;
            font-size: 0.88rem;
        }
        .file-meta span {
            color: var(--cave-muted);
        }
        .file-meta strong {
            min-width: 0;
            overflow-wrap: anywhere;
            color: var(--cave-text);
        }
        .verdict {
            border-radius: 8px;
            padding: 18px 20px;
            margin: 8px 0 14px;
            color: white;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.12);
        }
        .verdict .label {
            font-size: 0.85rem;
            opacity: 0.86;
            margin-bottom: 4px;
        }
        .verdict .value {
            font-size: 1.3rem;
            font-weight: 700;
            line-height: 1.35;
            letter-spacing: 0;
        }
        .verdict.ok { background: #247245; }
        .verdict.warn { background: var(--cave-amber); }
        .verdict.danger { background: var(--cave-red); }
        .finding-list {
            margin: 0.2rem 0 0.75rem;
            padding: 10px 12px 10px 28px;
            background: var(--cave-surface);
            border: 1px solid var(--cave-border);
            border-left: 4px solid var(--cave-indigo);
            border-radius: 8px;
        }
        .finding-list li {
            margin: 0.28rem 0;
            color: var(--cave-text);
            line-height: 1.45;
        }
        div[data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid var(--cave-border);
        }
        [data-testid="stMetric"] {
            background: var(--cave-surface);
            border: 1px solid var(--cave-border);
            border-radius: 8px;
            padding: 10px 12px;
            min-height: 78px;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.15rem;
            line-height: 1.25;
            color: var(--cave-text);
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.82rem;
            color: var(--cave-muted);
            font-weight: 650;
        }
        .card-title {
            margin: 0.6rem 0 0.35rem;
            color: var(--cave-text);
            font-size: 0.96rem;
            font-weight: 760;
            letter-spacing: 0;
        }
        [data-testid="stTabs"] [role="tablist"] {
            gap: 4px;
            border-bottom: 1px solid var(--cave-border);
        }
        [data-testid="stTabs"] [role="tab"] {
            border-radius: 8px 8px 0 0;
            color: var(--cave-muted);
            font-weight: 650;
            padding: 8px 12px;
        }
        [data-testid="stTabs"] [aria-selected="true"] {
            background: var(--cave-surface);
            color: var(--cave-text);
            border: 1px solid var(--cave-border);
            border-bottom-color: var(--cave-surface);
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--cave-border);
            border-radius: 8px;
            background: var(--cave-surface);
        }
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 8px;
            font-weight: 700;
            border: 1px solid #0f6b64;
        }
        .stButton > button[kind="primary"] {
            background: var(--cave-teal);
            border-color: var(--cave-teal);
        }
        .stButton > button[kind="primary"]:hover {
            background: #0b5f58;
            border-color: #0b5f58;
        }
        table {
            font-size: 0.9rem;
            line-height: 1.35;
            background: var(--cave-surface);
            border: 1px solid var(--cave-border);
            border-radius: 8px;
            overflow: hidden;
            width: 100%;
            table-layout: fixed;
        }
        th, td {
            padding: 0.34rem 0.45rem !important;
            vertical-align: top;
            border-bottom: 1px solid #e7ecf2 !important;
            overflow-wrap: anywhere;
        }
        th {
            background: var(--cave-surface-alt);
            color: var(--cave-text);
            font-weight: 750;
        }
        td:first-child {
            width: 36%;
            color: var(--cave-muted);
            font-weight: 650;
        }
        hr {
            margin: 1.4rem 0;
            border-color: var(--cave-border);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
