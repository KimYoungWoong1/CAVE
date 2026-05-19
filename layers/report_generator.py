"""기술 감정 보고서 PDF 생성.

라이브러리: fpdf2 (pip install fpdf2)
한국어 폰트: AppleGothic.ttf, NanumGothic.ttf, NotoSansCJK 계열 등
"""
from __future__ import annotations

import platform
from datetime import datetime
from pathlib import Path
from typing import Optional


# ─── 폰트 탐색 ────────────────────────────────────────────────────────────────

_FONT_CANDIDATES: dict[str, list[str]] = {
    "Darwin": [
        # fpdf2/PDF 뷰어 조합에 따라 TTC 컬렉션 폰트가 한글 깨짐을 만들 수 있어
        # 단일 TTF/OTF를 우선 사용한다. 단, AppleGothic.ttf는 fpdf2에서
        # OS/2 table 부재로 실패하는 macOS 버전이 있어 자동 검증으로 걸러낸다.
        "/Library/Fonts/NanumGothic.ttf",
        "/Library/Fonts/NotoSansCJKkr-Regular.otf",
        "/Library/Fonts/NotoSansKR-Regular.otf",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/Library/Fonts/AppleGothic.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ],
    "Linux": [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ],
    "Windows": [
        "C:\\Windows\\Fonts\\malgun.ttf",
        "C:\\Windows\\Fonts\\gulim.ttc",
    ],
}


def _find_korean_font() -> Optional[str]:
    root = Path(__file__).resolve().parents[1]
    local_candidates = [
        root / "fonts" / "NanumGothic.ttf",
        root / "fonts" / "NotoSansKR-Regular.otf",
        root / "assets" / "fonts" / "NanumGothic.ttf",
        Path(__file__).resolve().parent / "NanumGothic.ttf",
    ]
    candidates = [str(path) for path in local_candidates]
    candidates.extend(_FONT_CANDIDATES.get(platform.system(), []))
    for raw_path in candidates:
        path = Path(raw_path).expanduser()
        if path.exists() and _font_is_usable(path):
            return str(path)
    return None


def _font_is_usable(path: Path) -> bool:
    """Check whether fpdf2 can register the font."""
    try:
        from fpdf import FPDF  # type: ignore

        pdf = FPDF()
        if path.suffix.lower() == ".ttc":
            pdf.add_font("KoreanProbe", fname=str(path), collection_font_number=0)
        else:
            pdf.add_font("KoreanProbe", fname=str(path))
        return True
    except Exception:
        return False


def _register_pdf_font(pdf, font_path: str) -> str:
    """Register a Unicode Korean font and return the family name.

    Prefer TTF/OTF. TTC is kept as a last resort and registered with the first
    collection face.
    """
    path = Path(font_path)
    if path.suffix.lower() == ".ttc":
        pdf.add_font("Korean", style="", fname=str(path), collection_font_number=0)
        pdf.add_font("Korean", style="B", fname=str(path), collection_font_number=0)
    else:
        pdf.add_font("Korean", style="", fname=str(path))
        pdf.add_font("Korean", style="B", fname=str(path))
    return "Korean"


# ─── 색상 상수 ────────────────────────────────────────────────────────────────

COLOR_TITLE    = (20,  60, 120)   # 짙은 파랑
COLOR_SECTION  = (40, 100, 160)   # 중간 파랑
COLOR_WARN     = (180,  40,  40)  # 위험 빨강
COLOR_OK       = (30, 120,  60)   # 안전 초록
COLOR_GRAY     = (100, 100, 100)
COLOR_LIGHTBG  = (245, 248, 252)


# ─── 메인 진입점 ──────────────────────────────────────────────────────────────

def run(
    file_path: str,
    file_hash: str,
    c2pa_result,
    watermark_result,
    ai_det_result,
    rppg_result,
    fp_result,
    audit_result,
    damage_result,
    output_dir: str = "output",
    case_number: str = "",
    analyst_name: str = "CAVE 자동 분석 시스템",
) -> str:
    """PDF 보고서를 생성하고 저장 경로를 반환."""
    try:
        from fpdf import FPDF  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "fpdf2가 설치되지 않았습니다. `pip install fpdf2` 후 재시도하세요."
        ) from exc

    font_path = _find_korean_font()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # 한국어 폰트 등록
    if font_path:
        _FONT = _register_pdf_font(pdf, font_path)
        if hasattr(pdf, "set_text_shaping"):
            pdf.set_text_shaping(False)
    else:
        _FONT = "Helvetica"
        print(
            "[경고] 한국어 폰트를 찾을 수 없습니다. "
            "한국어 문자가 올바르게 표시되지 않을 수 있습니다.\n"
            "NanumGothic.ttf를 fonts/ 또는 layers/ 디렉토리에 복사하세요."
        )

    # ── 표지 ─────────────────────────────────────────────────────────────────
    pdf.add_page()
    _cover_page(pdf, _FONT, file_path, file_hash, case_number, analyst_name)

    # ── 분석 모듈별 결과 페이지 ─────────────────────────────────────────────
    pdf.add_page()
    _section_file_info(pdf, _FONT, file_path, file_hash)
    _section_c2pa(pdf, _FONT, c2pa_result)
    _section_watermark(pdf, _FONT, watermark_result)
    _section_ai_detection(pdf, _FONT, ai_det_result)
    _section_rppg(pdf, _FONT, rppg_result)
    _section_fingerprint(pdf, _FONT, fp_result)

    # ── 다층 증거 일관성 감사 ───────────────────────────────────────────────
    pdf.add_page()
    _section_audit(pdf, _FONT, audit_result)

    # ── 피해 규모 ────────────────────────────────────────────────────────────
    _section_damage(pdf, _FONT, damage_result)

    # ── 최종 의견 및 제한사항 ────────────────────────────────────────────────
    pdf.add_page()
    _section_final_opinion(pdf, _FONT, audit_result, damage_result)
    _section_limitations(pdf, _FONT)

    # ── 저장 ────────────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(file_path).stem
    out_path = str(Path(output_dir) / f"CAVE_report_{stem}_{timestamp}.pdf")
    pdf.output(out_path)
    return out_path


# ─── 페이지 구성 함수 ─────────────────────────────────────────────────────────

def _cover_page(pdf, font, file_path, file_hash, case_number, analyst_name):
    _set_font(pdf, font, "B", 20)
    pdf.set_text_color(*COLOR_TITLE)
    pdf.ln(30)
    pdf.cell(0, 12, "CAVE 기술 감정 보고서", align="C", new_x="LMARGIN", new_y="NEXT")

    _set_font(pdf, font, "", 12)
    pdf.set_text_color(*COLOR_GRAY)
    pdf.cell(0, 8, "AI 생성물 증거 검증 및 피해 정량화 파이프라인", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "CAVE: Credibility Audit for AI-Generated Evidence", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    pdf.set_draw_color(*COLOR_SECTION)
    pdf.set_line_width(0.5)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(10)

    rows = [
        ("감정 대상 파일",  Path(file_path).name),
        ("SHA-256 해시",   file_hash[:32] + "…"),
        ("사건 번호",       case_number or "미기재"),
        ("작성 기관",       "성균관대학교 I2S Lab"),
        ("작성자",          analyst_name),
        ("작성일",          datetime.now().strftime("%Y년 %m월 %d일 %H:%M")),
        ("파이프라인 버전",  "v1.0-stage1"),
    ]
    _set_font(pdf, font, "", 11)
    pdf.set_text_color(30, 30, 30)
    for label, value in rows:
        pdf.set_font(style="B")
        pdf.cell(55, 9, f"  {label}", border=0)
        pdf.set_font(style="")
        pdf.cell(0, 9, value, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(15)
    _set_font(pdf, font, "", 9)
    pdf.set_text_color(*COLOR_WARN)
    pdf.multi_cell(
        0, 6,
        "【주의】본 보고서는 기술적 분석 결과를 제공하며, 법적 판단을 대체하지 않습니다.\n"
        "최종 진위 판정은 법원의 감정인 의견 및 전문가 정밀 감정을 통해 이루어져야 합니다.",
        align="C",
    )


def _section_file_info(pdf, font, file_path, file_hash):
    _heading(pdf, font, "1. 감정 대상 파일 정보")
    path = Path(file_path)
    size_kb = path.stat().st_size / 1024 if path.exists() else 0
    rows = [
        ("파일명",          path.name),
        ("전체 경로",       str(path.resolve())),
        ("파일 크기",       f"{size_kb:.1f} KB"),
        ("확장자",          path.suffix.upper()),
        ("SHA-256",        file_hash),
    ]
    _key_value_table(pdf, font, rows)


def _section_c2pa(pdf, font, result):
    _heading(pdf, font, "2. 출처 인증 검사")
    if result is None:
        _note(pdf, font, "분석 결과 없음")
        return

    manifest_txt = "있음" if result.manifest_present else "없음"
    rows = [
        ("매니페스트 존재", manifest_txt),
        ("서명 유효성",     result.signature_valid),
        ("AI 사용 선언",    "예" if result.ai_usage_declared else "아니오"),
        ("생성 도구",       result.claim_generator or "정보 없음"),
        ("편집 이력",       ", ".join(result.edit_history) if result.edit_history else "없음"),
        ("AI 점수",         f"{result.ai_score:.3f}"),
    ]
    _key_value_table(pdf, font, rows, highlight_last=True)
    _note(pdf, font, result.notes)

    # 한계 경고
    _warn(pdf, font,
          "한계: C2PA는 출처 기록 메커니즘이며 진정성 보증 불가. "
          "메타데이터 삭제·타임스탬프 조작으로 우회 가능 (Golaszewski et al., 2026).")


def _section_watermark(pdf, font, result):
    _heading(pdf, font, "3. 워터마크·메타 신호 검사")
    if result is None or result.ai_score is None:
        _note(pdf, font, "분석 결과 없음. 다층 증거 일관성 감사에서 제외.")
        return
    rows = [
        ("워터마크 존재",    str(result.watermark_present)),
        ("AI 워터마크",      str(result.ai_watermark)),
        ("신뢰도",           f"{result.confidence:.3f}"),
        ("AI 점수",          f"{result.ai_score:.3f}"),
    ]
    _key_value_table(pdf, font, rows, highlight_last=True)
    _note(pdf, font, result.notes)


def _section_ai_detection(pdf, font, result):
    _heading(pdf, font, "4. AI 생성 탐지 모델")
    if result is None or result.ai_score is None:
        _note(pdf, font, "분석 결과 없음. 다층 증거 일관성 감사에서 제외.")
        return
    rows = [
        ("AI 생성 가능성",   f"{result.ai_probability:.3f}" if result.ai_probability else "-"),
        ("얼굴 합성 수준",   result.face_synthesis_level or "-"),
        ("프레임 불일치",    str(result.frame_inconsistency)),
        ("조명 불일치",      str(result.lighting_inconsistency)),
        ("판정",             result.verdict or "-"),
        ("AI 점수",          f"{result.ai_score:.3f}"),
    ]
    _key_value_table(pdf, font, rows, highlight_last=True)
    _note(pdf, font, result.notes)


def _section_rppg(pdf, font, result):
    _heading(pdf, font, "5. 생체 신호 일관성 검사")
    if result is None or result.ai_score is None:
        _note(pdf, font, "분석 불가 (정지 이미지는 시간축이 없어 rPPG 제외). 다층 감사에서 제외.")
        return
    mode = getattr(result, "analysis_mode", "none")
    rows = [
        ("분석 모드",          _rppg_analysis_mode(result) if mode == "video" else mode),
        ("심박 신호 자연스러움", f"{result.heartbeat_naturalness:.3f}" if result.heartbeat_naturalness is not None else "-"),
        ("생체 신호 일치",     str(result.biometric_match)),
        ("AI 점수",            f"{result.ai_score:.3f}"),
    ]
    _key_value_table(pdf, font, rows, highlight_last=True)
    evidence_rows = _rppg_evidence_rows(result)
    if evidence_rows:
        _key_value_table(pdf, font, evidence_rows)
    _note(pdf, font, result.notes)


def _section_fingerprint(pdf, font, result):
    _heading(pdf, font, "6. 생성 모델 흔적 분석")
    if result is None or result.ai_score is None:
        _note(pdf, font, "분석 결과 없음. 다층 증거 일관성 감사에서 제외.")
        return
    rows = [
        ("분석 방식",       _fingerprint_analysis_mode(result)),
        ("AI 생성 가능성",  result.ai_likelihood or "-"),
        ("조작 계열 추정",   result.generation_method or "-"),
        ("Attribution",     result.model_family or "-"),
        ("신뢰도",          f"{result.confidence:.3f}"),
        ("AI 점수",         f"{result.ai_score:.3f}"),
    ]
    _key_value_table(pdf, font, rows, highlight_last=True)
    evidence_rows = _fingerprint_evidence_rows(result)
    if evidence_rows:
        _key_value_table(pdf, font, evidence_rows)
    _note(pdf, font, result.notes)


def _section_audit(pdf, font, result):
    _heading(pdf, font, "7. 다층 증거 일관성 감사")
    if result is None:
        _note(pdf, font, "분석 결과 없음")
        return

    # 판정 강조 박스
    verdict_color = COLOR_WARN if result.integrity_clash else (
        COLOR_OK if "진본" in result.verdict_kr else COLOR_SECTION
    )
    pdf.set_fill_color(*verdict_color)
    pdf.set_text_color(255, 255, 255)
    _set_font(pdf, font, "B", 12)
    pdf.cell(0, 10, f"  최종 판정: {result.verdict_kr}", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(30, 30, 30)

    pdf.ln(3)
    rows = [
        ("Integrity Clash",     "감지됨" if result.integrity_clash else "없음"),
        ("정밀 감정 필요",       "예" if result.expert_review_needed else "아니오"),
        ("일관성 점수",          f"{result.consistency_score:.3f} / 1.000"),
        ("출처 레이어 평균",     f"{result.provenance_avg:.3f}" if result.provenance_avg is not None else "N/A"),
        ("탐지 레이어 평균",     f"{result.detection_avg:.3f}"  if result.detection_avg  is not None else "N/A"),
    ]
    if result.layer_scores.get("rppg") is not None:
        rows.append(("영상 신호 조합", _video_signal_summary_from_scores(result.layer_scores)))
    _key_value_table(pdf, font, rows)

    if result.clash_details:
        pdf.ln(3)
        _set_font(pdf, font, "B", 10)
        pdf.cell(0, 7, "충돌 상세 내역:", new_x="LMARGIN", new_y="NEXT")
        _set_font(pdf, font, "", 9)
        pdf.set_text_color(*COLOR_WARN)
        for detail in result.clash_details:
            pdf.multi_cell(pdf.epw, 6, f"  - {detail}")
        pdf.set_text_color(30, 30, 30)

    _note(pdf, font, result.notes)

    # 분석 모듈별 점수 표
    pdf.ln(3)
    _set_font(pdf, font, "B", 10)
    pdf.cell(0, 7, "분석 모듈별 AI 점수:", new_x="LMARGIN", new_y="NEXT")
    _layer_score_table(pdf, font, result.layer_scores)


def _section_damage(pdf, font, result):
    _heading(pdf, font, "8. 피해 규모 산정")
    if result is None:
        _note(pdf, font, "분석 결과 없음")
        return

    # 피해 등급 강조
    grade_color = (
        COLOR_WARN    if result.total_score >= 16 else
        (200, 100, 0) if result.total_score >= 8  else
        COLOR_OK
    )
    pdf.set_fill_color(*grade_color)
    pdf.set_text_color(255, 255, 255)
    _set_font(pdf, font, "B", 12)
    pdf.cell(
        0, 10,
        f"  피해 규모 점수: {result.total_score:.1f}점  |  등급: {result.grade_kr}",
        fill=True, new_x="LMARGIN", new_y="NEXT",
    )
    pdf.set_text_color(30, 30, 30)
    pdf.ln(3)

    context_rows = _damage_context_rows(result)
    if context_rows:
        _set_font(pdf, font, "B", 10)
        pdf.cell(0, 7, "유포 정황 입력 요약:", new_x="LMARGIN", new_y="NEXT")
        _key_value_table(pdf, font, context_rows)
        pdf.ln(2)

    # 하위 점수 표
    sub_scores = [
        ("확산 점수 (×1.5)",           result.diffusion_score),
        ("재유포 위험 점수 (×1.5)",     result.redistribution_score),
        ("식별 위험 점수 (×1.0)",       result.id_risk_score),
        ("침해 심각도 점수 (×1.0)",     result.severity_score),
        ("경제적 피해 점수 (×0.5)",     result.economic_score),
        ("사회적 영향 점수 (×0.5)",     result.social_score),
    ]
    _score_bar_table(pdf, font, sub_scores, max_val=5.0)

    evidence_rows = _damage_evidence_rows(result)
    if evidence_rows:
        pdf.ln(2)
        _key_value_table(pdf, font, evidence_rows)

    _note(pdf, font, result.notes)

    # 피해 등급 기준 안내
    _set_font(pdf, font, "", 8)
    pdf.set_text_color(*COLOR_GRAY)
    pdf.multi_cell(
        pdf.epw, 5,
        "피해 등급 기준: 0~7점 제한적 / 8~15점 중간 / 16~23점 심각 / 24~30점 광범위·고위험",
    )
    pdf.set_text_color(30, 30, 30)


def _section_final_opinion(pdf, font, audit_result, damage_result):
    _heading(pdf, font, "9. 최종 의견")

    # 판정 결론
    _set_font(pdf, font, "B", 11)
    verdict = audit_result.verdict_kr if audit_result else "분석 불가"
    pdf.multi_cell(pdf.epw, 8, f"기술 분석 결론: {verdict}")
    pdf.ln(3)

    if audit_result and audit_result.expert_review_needed:
        _set_font(pdf, font, "B", 10)
        pdf.set_text_color(*COLOR_WARN)
        pdf.multi_cell(
            pdf.epw, 7,
            "[권고] 레이어 간 분석 결과 상충 또는 불확실 판정 발생. "
            "법원 감정인에 의한 정밀 감정을 요청합니다.",
        )
        pdf.set_text_color(30, 30, 30)

    if damage_result:
        pdf.ln(3)
        _set_font(pdf, font, "", 10)
        pdf.multi_cell(
            0, 7,
            f"피해 규모 점수 {damage_result.total_score:.1f}점 ({damage_result.grade_kr})은 "
            "소송촉진 등에 관한 특례법상 배상명령 신청 시 손해 규모 입증의 기술적 근거 자료로 활용될 수 있습니다."
        )


def _section_limitations(pdf, font):
    _heading(pdf, font, "10. 기술적 한계 및 고지 사항")
    _set_font(pdf, font, "", 9)
    pdf.set_text_color(60, 60, 60)
    limitations = [
        "C2PA는 출처 기록 메커니즘으로 진정성 보증 불가; metadata washing으로 우회 가능 (Golaszewski et al., 2026).",
        "워터마킹은 압축·크롭·재인코딩 시 손상 가능; 단 1장 샘플로 제거·위조 가능 (arXiv:2502.06418, 2025).",
        "AI 탐지 모델은 cross-dataset 환경에서 AUC 최대 50% 급락 가능 (Chandra et al., 2025).",
        "rPPG 분석은 조명 변화·낮은 해상도에서 신뢰도 저하.",
        "핑거프린트 추정은 확률적·보조적 지표로 특정 모델명 단정 불가.",
        "피해 확산 GNN은 DamageInputs 기반 approximate propagation graph와 합성 학습 데이터에 근거하므로 실제 플랫폼 원시 전파 로그와 동일하지 않음.",
        "본 보고서는 출처 인증, 워터마크·메타 신호, AI 생성 탐지, 생성 모델 흔적, 다층 감사, 피해 산정 모듈을 기반으로 작성됨.",
        "최종 법적 판단은 법원 및 공인 감정인의 전문적 의견을 통해 이루어져야 함.",
    ]
    for item in limitations:
        pdf.multi_cell(pdf.epw, 6, f"  - {item}")
    pdf.set_text_color(30, 30, 30)

    pdf.ln(8)
    _set_font(pdf, font, "", 8)
    pdf.set_text_color(*COLOR_GRAY)
    pdf.cell(
        0, 6,
        f"본 보고서는 CAVE 파이프라인 v1.0에 의해 자동 생성되었습니다. "
        f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        align="C",
    )


# ─── UI 헬퍼 ─────────────────────────────────────────────────────────────────

def _set_font(pdf, font, style="", size=10):
    pdf.set_font(font, style=style, size=size)


def _heading(pdf, font, text: str):
    pdf.ln(6)
    pdf.set_fill_color(*COLOR_SECTION)
    pdf.set_text_color(255, 255, 255)
    _set_font(pdf, font, "B", 11)
    pdf.cell(0, 9, f"  {text}", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(30, 30, 30)
    pdf.ln(2)


def _note(pdf, font, text: str):
    _set_font(pdf, font, "", 9)
    pdf.set_text_color(*COLOR_GRAY)
    pdf.multi_cell(pdf.epw, 5, text)
    pdf.set_text_color(30, 30, 30)
    pdf.ln(1)


def _warn(pdf, font, text: str):
    _set_font(pdf, font, "", 9)
    pdf.set_text_color(*COLOR_WARN)
    pdf.multi_cell(pdf.epw, 5, f"[!] {text}")
    pdf.set_text_color(30, 30, 30)
    pdf.ln(1)


def _fingerprint_analysis_mode(result) -> str:
    notes = getattr(result, "notes", "") or ""
    if "학습 기반 video fingerprint classifier" in notes:
        return "학습 기반 영상 fingerprint classifier"
    if "학습 기반 fingerprint classifier" in notes:
        return "학습 기반 이미지 attribution"
    if "learned_classifier=image-trained" in notes:
        return "영상 heuristic (이미지 classifier 미적용)"
    if "영상용 learned fingerprint classifier 없음" in notes:
        return "영상 heuristic fallback"
    if "학습 classifier 없음" in notes or "FFT/노이즈 기반 보조 분석" in notes:
        return "FFT/노이즈 heuristic fallback"
    return "fingerprint 분석"


def _damage_evidence_rows(result) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    gnn_risk = getattr(result, "gnn_risk", None)
    if gnn_risk is not None:
        rows.append(("GNN 전파 위험도", _fmt_optional(gnn_risk)))
        rows.append((
            "GNN 확산 근거",
            (
                f"GNN 환산 {_fmt_optional(getattr(result, 'gnn_converted_score', None))} / "
                f"heuristic {_fmt_optional(getattr(result, 'heuristic_diffusion', None))} / "
                f"blend {getattr(result, 'blend_ratio', '')}"
            ),
        ))
        rows.append(("GNN test acc", _fmt_optional(getattr(result, "gnn_test_acc", None))))
    else:
        reason = getattr(result, "gnn_fallback_reason", None)
        if reason:
            rows.append(("GNN fallback", str(reason)))

    learned_prob = getattr(result, "redistribution_learned_prob", None)
    if learned_prob is not None:
        rows.append(("재유포 learned prob", _fmt_optional(learned_prob)))
        rows.append((
            "재유포 근거",
            (
                f"learned 환산 {_fmt_optional(getattr(result, 'redistribution_converted_score', None))} / "
                f"heuristic {_fmt_optional(getattr(result, 'redistribution_heuristic_score', None))} / "
                f"blend {getattr(result, 'redistribution_blend_ratio', '')}"
            ),
        ))
        rows.append(("재유포 model", str(getattr(result, "redistribution_model_type", "") or "N/A")))
        rows.append(("재유포 test AUC", _fmt_optional(getattr(result, "redistribution_test_auc", None))))
    else:
        reason = getattr(result, "redistribution_fallback_reason", None)
        if reason:
            rows.append(("재유포 fallback", str(reason)))
    return rows


def _damage_context_rows(result) -> list[tuple[str, str]]:
    context = getattr(result, "spread_context", {}) or {}
    rows: list[tuple[str, str]] = []

    labels = [
        ("platform_names", "발견 플랫폼"),
        ("first_seen_at", "최초 발견"),
        ("evidence_capture_count", "URL/캡처 수"),
        ("victim_identifiable", "피해자 특정 가능"),
    ]
    for key, label in labels:
        value = str(context.get(key, "")).strip()
        if value and value not in {"0", "미입력"}:
            rows.append((label, value))

    source = str(context.get("evidence_source", "")).strip()
    if source:
        rows.append(("보전 자료", _shorten(source, 80)))
    return rows


def _fmt_optional(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return str(value)


def _rppg_analysis_mode(result) -> str:
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
        ("calibrated_prob", "보정 확률"),
        ("threshold", "영상 threshold"),
        ("heuristic", "Heuristic"),
        ("face_detected", "Face crop"),
        ("temporal_delta", "Temporal delta"),
        ("test_auc", "Layer 5 AUC"),
    ]
    return [
        (label, str(evidence[key]))
        for key, label in keys
        if evidence.get(key) not in (None, "", "N/A")
    ]


def _video_signal_summary_from_scores(scores: dict[str, float | None]) -> str:
    ai_score = scores.get("ai_detection")
    rppg_score = scores.get("rppg")
    fp_score = scores.get("fingerprint")
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

    return (
        f"Detector {_signal_level(ai_score)} + rPPG {_signal_level(rppg_score)} + "
        f"Fingerprint {_signal_level(fp_score)} -> {decision}"
    )


def _signal_level(score: float | None) -> str:
    if score is None:
        return "미실행"
    if score >= 0.60:
        return "강함"
    if score >= 0.45:
        return "중간"
    return "낮음"


def _shorten(text: str, limit: int) -> str:
    text = str(text).strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _key_value_table(pdf, font, rows: list[tuple[str, str]], highlight_last=False):
    _set_font(pdf, font, "", 10)
    for i, (label, value) in enumerate(rows):
        is_last = highlight_last and i == len(rows) - 1
        if i % 2 == 0:
            pdf.set_fill_color(*COLOR_LIGHTBG)
        else:
            pdf.set_fill_color(255, 255, 255)

        if is_last:
            pdf.set_text_color(*COLOR_SECTION)
            pdf.set_font(style="B")

        pdf.cell(60, 7, f"  {label}", border=0, fill=True)
        pdf.cell(0,  7, str(value),  border=0, fill=True, new_x="LMARGIN", new_y="NEXT")

        if is_last:
            pdf.set_text_color(30, 30, 30)
            pdf.set_font(style="")


def _score_bar_table(pdf, font, items: list[tuple[str, float]], max_val: float = 5.0):
    """이름 + 점수 + 막대 그래프."""
    _set_font(pdf, font, "", 9)
    bar_max_w = 80.0
    for i, (label, score) in enumerate(items):
        pdf.set_fill_color(*COLOR_LIGHTBG if i % 2 == 0 else (255, 255, 255))
        pdf.cell(80, 6, f"  {label}", border=0, fill=True)
        pdf.cell(15, 6, f"{score:.2f}/5", border=0, fill=True)

        # 막대
        bar_w = (score / max_val) * bar_max_w
        bar_color = (
            COLOR_WARN if score >= 3.5 else
            (200, 150, 0) if score >= 2.0 else
            COLOR_OK
        )
        x, y = pdf.get_x(), pdf.get_y()
        pdf.set_fill_color(*bar_color)
        if bar_w > 0:
            pdf.rect(x, y + 1, bar_w, 4, "F")
        pdf.ln(6)
    pdf.set_fill_color(255, 255, 255)


def _layer_score_table(pdf, font, scores: dict):
    labels = {
        "c2pa":        "출처 인증 검사",
        "watermark":   "워터마크·메타 신호 검사",
        "ai_detection":"AI 생성 탐지 모델",
        "rppg":        "생체 신호 일관성 검사",
        "fingerprint": "생성 모델 흔적 분석",
    }
    _set_font(pdf, font, "", 9)
    for key, label in labels.items():
        score = scores.get(key)
        score_str = f"{score:.3f}" if score is not None else "미구현 (N/A)"
        pdf.set_fill_color(*COLOR_LIGHTBG)
        pdf.cell(70, 6, f"  {label}", border=0, fill=True)
        pdf.cell(0,  6, score_str,   border=0, fill=True, new_x="LMARGIN", new_y="NEXT")
