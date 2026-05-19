"""CAVE Pipeline — AI 생성물 증거 검증 및 피해 정량화 파이프라인.

실행 예시:
  python main.py path/to/image_or_video
  python main.py test_data/deepfake/video.mp4 --crime-type deepfake_sexual
  python main.py image.jpg --case-number "2025고단1234" --analyst "홍길동"
  python main.py image.jpg --demo   # 데모 입력값 사용 (실제 파일 없어도 실행)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from layers import (
    c2pa_check,
    watermark_check,
    ai_detection,
    rppg_check,
    fingerprint,
    cross_layer_audit,
    damage_score,
    report_generator,
)


# ─── 해시 ────────────────────────────────────────────────────────────────────

def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── 피해 입력값 수집 ─────────────────────────────────────────────────────────

def collect_damage_inputs(
    crime_type: str,
    interactive: bool,
    demo: bool,
) -> damage_score.DamageInputs:
    """피해 입력값 수집: 데모 → 자동 예시 / interactive → 터미널 입력 / 기본 → 0."""
    if demo:
        if crime_type == "financial_fraud":
            return damage_score.DamageInputs.example_financial_fraud()
        return damage_score.DamageInputs.example_deepfake_sexual()

    if interactive:
        return _collect_interactive()

    # 기본값 (모두 0)
    return damage_score.DamageInputs()


def _collect_interactive() -> damage_score.DamageInputs:
    """터미널에서 피해 지표를 입력받는다."""
    print("\n[피해 규모 산정] 피해 규모 입력 (Enter = 기본값 0 / False)")
    print("─" * 50)

    def ask_int(prompt: str, default: int = 0) -> int:
        raw = input(f"  {prompt} [{default}]: ").strip()
        return int(raw) if raw.isdigit() else default

    def ask_float(prompt: str, default: float = 0.0) -> float:
        raw = input(f"  {prompt} [{default}]: ").strip()
        try:
            return float(raw)
        except ValueError:
            return default

    def ask_bool(prompt: str) -> bool:
        raw = input(f"  {prompt} [y/N]: ").strip().lower()
        return raw in ("y", "yes", "예")

    return damage_score.DamageInputs(
        num_posts            = ask_int("게시물 수"),
        num_platforms        = ask_int("플랫폼 수"),
        num_shares           = ask_int("공유 수"),
        num_views            = ask_int("조회수"),
        spread_speed_hours   = ask_float("1,000 인게이지먼트 도달 시간 (시간)"),
        has_variants         = ask_bool("변형본 존재?"),
        on_closed_platforms  = ask_bool("폐쇄형 플랫폼(텔레그램 등) 유포?"),
        reappeared_after_deletion = ask_bool("삭제 후 재등장?"),
        face_match_score     = ask_float("얼굴 일치도 (0~1)"),
        voice_match_score    = ask_float("음성 일치도 (0~1)"),
        real_name_mentioned  = ask_bool("실명 언급?"),
        affiliation_revealed = ask_bool("소속·직장 정보 노출?"),
        is_sexual_manipulation = ask_bool("성적 조작 콘텐츠?"),
        is_threatening       = ask_bool("협박성?"),
        is_defamatory        = ask_bool("명예훼손성?"),
        financial_loss_krw   = ask_float("금전 피해 금액 (원)"),
        reputation_damaged   = ask_bool("평판 손실?"),
        election_manipulation_risk = ask_float("선거 조작 위험도 (0~1)"),
        public_opinion_impact = ask_float("여론 영향 범위 (0~1)"),
    )


# ─── 파이프라인 실행 ─────────────────────────────────────────────────────────

def run_pipeline(
    file_path: str,
    crime_type: str = "default",
    case_number: str = "",
    analyst_name: str = "CAVE 자동 분석 시스템",
    output_dir: str = "output",
    interactive: bool = False,
    demo: bool = False,
) -> None:
    sep = "=" * 64

    print(f"\n{sep}")
    print(" CAVE — AI 생성물 증거 검증 파이프라인")
    print(f"{sep}")
    print(f"  분석 파일  : {file_path}")
    print(f"  범죄 유형  : {crime_type}")
    if case_number:
        print(f"  사건 번호  : {case_number}")
    print(f"  SHA-256    : {compute_sha256(file_path)}")
    print()

    # ── 출처 인증 검사 ───────────────────────────────────────────────────────
    print("[출처 인증 검사] C2PA 검증...")
    l1 = c2pa_check.run(file_path)
    print(f"  매니페스트 : {'존재' if l1.manifest_present else '없음'}")
    print(f"  서명 상태  : {l1.signature_valid}")
    print(f"  AI 선언    : {'예' if l1.ai_usage_declared else '아니오'}")
    print(f"  AI 점수    : {l1.ai_score:.3f}")

    # ── 분석 모듈 ───────────────────────────────────────────────────────────
    print("[워터마크·메타 신호 검사] 마커 탐지...")
    l2 = watermark_check.run(file_path)
    print(f"  워터마크  : {'감지' if l2.watermark_present else '없음'}")
    print(f"  AI 마커    : {'감지' if l2.ai_watermark else '없음'}")
    print(f"  신뢰도     : {l2.confidence:.3f}")
    print(f"  AI 점수    : {l2.ai_score:.3f}")

    print("[AI 생성 탐지 모델] 이미지/프레임 분류...")
    l3 = ai_detection.run(file_path)
    if l3.ai_probability is None:
        print(f"  상태       : 미실행 ({l3.notes})")
    else:
        print(f"  AI 확률    : {l3.ai_probability:.3f}")
        print(f"  판정       : {l3.verdict}")
        print(f"  AI 점수    : {l3.ai_score:.3f}")

    print("[생체 신호 일관성 검사] rPPG 분석...")
    l4 = rppg_check.run(file_path)
    if l4.ai_score is None:
        print(f"  상태       : 미실행 ({l4.notes})")
    else:
        print(f"  분석 모드  : {l4.analysis_mode}")
        print(f"  자연스러움 : {l4.heartbeat_naturalness:.3f}")
        print(f"  생체 일치  : {'예' if l4.biometric_match else '아니오'}")
        print(f"  AI 점수    : {l4.ai_score:.3f}")

    print("[생성 모델 흔적 분석] 핑거프린트 추정...")
    l5 = fingerprint.run(file_path)
    if l5.ai_score is None:
        print(f"  상태       : 미실행 ({l5.notes})")
    else:
        print(f"  AI 가능성  : {l5.ai_likelihood}")
        print(f"  생성 방식  : {l5.generation_method}")
        print(f"  모델 계열  : {l5.model_family}")
        print(f"  신뢰도     : {l5.confidence:.3f}")
        print(f"  AI 점수    : {l5.ai_score:.3f}")

    # ── 다층 증거 일관성 감사 ───────────────────────────────────────────────
    print("[다층 증거 일관성 감사] 종합 판정...")
    layer_results = {
        "c2pa":         l1,
        "watermark":    l2,
        "ai_detection": l3,
        "rppg":         l4,
        "fingerprint":  l5,
    }
    l6 = cross_layer_audit.run(layer_results)
    print(f"  판정       : {l6.verdict_kr}")
    print(f"  Clash      : {'감지됨' if l6.integrity_clash else '없음'}")
    print(f"  일관성     : {l6.consistency_score:.3f}")
    print(f"  정밀 감정  : {'필요' if l6.expert_review_needed else '불필요'}")
    if l6.clash_details:
        for detail in l6.clash_details:
            print(f"    ⚠ {detail}")

    # ── 피해 규모 산정 ───────────────────────────────────────────────────────
    print("[피해 규모 산정] 피해 점수 계산...")
    d_inputs = collect_damage_inputs(crime_type, interactive, demo)
    l7 = damage_score.run(d_inputs, crime_type=crime_type)
    print(f"  확산 점수       : {l7.diffusion_score:.2f} / 5")
    print(f"  재유포 위험     : {l7.redistribution_score:.2f} / 5")
    print(f"  식별 위험       : {l7.id_risk_score:.2f} / 5")
    print(f"  침해 심각도     : {l7.severity_score:.2f} / 5")
    print(f"  경제적 피해     : {l7.economic_score:.2f} / 5")
    print(f"  사회적 영향     : {l7.social_score:.2f} / 5")
    print(f"  ─────────────────────────────")
    print(f"  피해 규모 점수  : {l7.total_score:.1f} / 30")
    print(f"  피해 등급       : {l7.grade_kr}")

    # ── 기술 감정 보고서 ─────────────────────────────────────────────────────
    print("[기술 감정 보고서] PDF 생성...")
    try:
        report_path = report_generator.run(
            file_path     = file_path,
            file_hash     = compute_sha256(file_path),
            c2pa_result   = l1,
            watermark_result  = l2,
            ai_det_result = l3,
            rppg_result   = l4,
            fp_result     = l5,
            audit_result  = l6,
            damage_result = l7,
            output_dir    = output_dir,
            case_number   = case_number,
            analyst_name  = analyst_name,
        )
        print(f"  보고서 저장    : {report_path}")
    except ImportError as e:
        print(f"  [경고] PDF 생성 실패: {e}")
        report_path = None

    # ── 최종 요약 ────────────────────────────────────────────────────────────
    print()
    print(f"{sep}")
    print(" 분석 완료 요약")
    print(f"{sep}")
    print(f"  파일        : {Path(file_path).name}")
    print(f"  C2PA 판정   : {l1.signature_valid} (AI 점수 {l1.ai_score:.3f})")
    print(f"  종합 판정   : {l6.verdict_kr}")
    print(f"  피해 점수   : {l7.total_score:.1f}점 — {l7.grade_kr}")
    if report_path:
        print(f"  보고서      : {report_path}")
    print(f"{sep}\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CAVE — AI 생성물 증거 검증 및 피해 정량화 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("file", help="분석할 파일 경로 (이미지 또는 영상)")
    parser.add_argument(
        "--crime-type",
        default="default",
        choices=["default", "deepfake_sexual", "financial_fraud", "election_manipulation"],
        help="범죄 유형 (피해 가중치 조정, 기본값: default)",
    )
    parser.add_argument("--case-number", default="", help="사건 번호 (예: 2025고단1234)")
    parser.add_argument("--analyst",     default="CAVE 자동 분석 시스템", help="감정인 이름")
    parser.add_argument("--output-dir",  default="output", help="보고서 출력 디렉토리")
    parser.add_argument(
        "--interactive", action="store_true",
        help="피해 규모 입력값을 터미널에서 직접 입력"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="예시 피해 입력값으로 데모 실행 (실제 파일 없어도 가능)"
    )

    args = parser.parse_args()

    file_path = args.file
    if not Path(file_path).exists():
        if args.demo:
            # 데모: 더미 파일 생성
            dummy = Path("test_data/local_demo/demo_sample.jpg")
            dummy.parent.mkdir(parents=True, exist_ok=True)
            if not dummy.exists():
                dummy.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # 최소 JPEG 헤더
            file_path = str(dummy)
            print(f"[데모] 더미 파일 생성: {file_path}")
        else:
            print(f"[오류] 파일을 찾을 수 없습니다: {file_path}")
            print("  --demo 플래그로 데모를 실행하거나 올바른 파일 경로를 지정하세요.")
            sys.exit(1)

    run_pipeline(
        file_path    = file_path,
        crime_type   = args.crime_type,
        case_number  = args.case_number,
        analyst_name = args.analyst,
        output_dir   = args.output_dir,
        interactive  = args.interactive,
        demo         = args.demo,
    )


if __name__ == "__main__":
    main()
