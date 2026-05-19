"""Generate README-safe demo output images.

The generated asset summarizes actual local CAVE pipeline scores without
embedding the underlying media files. This keeps the README useful while
avoiding redistribution of dataset images or sensitive samples.
"""
from __future__ import annotations

import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layers import ai_detection, cross_layer_audit, damage_score, fingerprint  # noqa: E402


ASSET_PATH = ROOT / "docs" / "assets" / "readme_demo_summary.png"
AI_SAMPLE = ROOT / "test_data" / "genimage" / "holdout" / "ai" / "GLIDE__tiny_genimage_0000831.jpg"
REAL_SAMPLE = ROOT / "test_data" / "genimage" / "holdout" / "real" / "Real__tiny_genimage_0000532.jpg"


@dataclass
class MediaSnapshot:
    label: str
    sample_name: str
    verdict_kr: str
    ai_score: float
    fingerprint_score: float
    consistency: float
    review_needed: bool
    fingerprint_method: str


def main() -> None:
    ai_snapshot = analyze_media("AI 생성/조작 의심", AI_SAMPLE)
    real_snapshot = analyze_media("진본 가능성 높음", REAL_SAMPLE)
    high_damage = damage_score.run(
        damage_score.DamageInputs.example_deepfake_sexual(),
        crime_type="deepfake_sexual",
    )
    low_damage = damage_score.run(
        damage_score.DamageInputs(),
        crime_type="deepfake_sexual",
    )

    ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    image = render_summary(ai_snapshot, real_snapshot, high_damage, low_damage)
    image.save(ASSET_PATH)
    print(f"Wrote {ASSET_PATH.relative_to(ROOT)}")


def analyze_media(label: str, path: Path) -> MediaSnapshot:
    if not path.exists():
        raise FileNotFoundError(
            f"README demo sample is missing: {path.relative_to(ROOT)}. "
            "Run the Tiny-GenImage preparation script first."
        )
    ai_result = ai_detection.run(str(path))
    fp_result = fingerprint.run(str(path))
    audit = cross_layer_audit.run(
        {
            "ai_detection": ai_result,
            "fingerprint": fp_result,
        }
    )
    return MediaSnapshot(
        label=label,
        sample_name=path.name,
        verdict_kr=audit.verdict_kr,
        ai_score=float(ai_result.ai_score or 0.0),
        fingerprint_score=float(fp_result.ai_score or 0.0),
        consistency=float(audit.consistency_score),
        review_needed=bool(audit.expert_review_needed),
        fingerprint_method=fp_result.generation_method or "unknown",
    )


def render_summary(
    ai_snapshot: MediaSnapshot,
    real_snapshot: MediaSnapshot,
    high_damage: damage_score.DamageResult,
    low_damage: damage_score.DamageResult,
) -> Image.Image:
    width, height = 1800, 1120
    img = Image.new("RGB", (width, height), "#f6f8fb")
    draw = ImageDraw.Draw(img)

    title = font(54)
    subtitle = font(25)
    small = font(21)
    body = font(26)
    label_font = font(22)

    draw.text((70, 56), "CAVE 실제 출력 요약", fill="#14213d", font=title)
    draw.text(
        (72, 128),
        "원본 미디어를 포함하지 않고, 로컬 파이프라인의 실제 레이어 점수만 README용으로 시각화했습니다.",
        fill="#526071",
        font=subtitle,
    )
    draw.text((70, 178), "Layer 3/5 탐지 대비", fill="#0f766e", font=small)
    draw.text((1216, 178), "Layer 7 피해 규모", fill="#0f766e", font=small)

    draw_media_card(
        draw,
        x=70,
        y=220,
        w=520,
        h=720,
        snapshot=ai_snapshot,
        accent="#b45309",
        badge="#fff7ed",
        title_fill="#7c2d12",
    )
    draw_media_card(
        draw,
        x=640,
        y=220,
        w=520,
        h=720,
        snapshot=real_snapshot,
        accent="#047857",
        badge="#ecfdf5",
        title_fill="#064e3b",
    )
    draw_damage_card(
        draw,
        x=1210,
        y=220,
        w=520,
        h=720,
        high_damage=high_damage,
        low_damage=low_damage,
    )

    footer = (
        "README에는 데이터셋 원본 이미지·영상이 포함되지 않습니다. "
        "점수는 Tiny-GenImage holdout 샘플과 딥페이크 성범죄 유포 데모 정황으로 산출했습니다."
    )
    draw.rounded_rectangle((70, 980, 1730, 1054), radius=22, fill="#e8f3f2", outline="#b8deda", width=2)
    draw.text((102, 1005), footer, fill="#0f4f4a", font=label_font)

    return img


def draw_media_card(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    snapshot: MediaSnapshot,
    accent: str,
    badge: str,
    title_fill: str,
) -> None:
    draw.rounded_rectangle((x + 8, y + 10, x + w + 8, y + h + 10), radius=26, fill="#d8e0ea")
    draw.rounded_rectangle((x, y, x + w, y + h), radius=26, fill="#ffffff", outline="#d5dee9", width=2)
    draw.rounded_rectangle((x, y, x + w, y + 16), radius=8, fill=accent)
    draw.rounded_rectangle((x + 32, y + 42, x + 276, y + 88), radius=22, fill=badge, outline=accent, width=2)
    draw.text((x + 54, y + 51), "실제 출력 예시", fill=accent, font=font(21))

    draw.text((x + 32, y + 118), snapshot.label, fill=title_fill, font=font(38))
    draw_wrapped(
        draw,
        snapshot.verdict_kr,
        x + 34,
        y + 176,
        w - 68,
        font(25),
        fill="#1f2937",
        line_spacing=8,
    )

    metric_y = y + 292
    draw_score_row(draw, x + 34, metric_y, "Layer 3 AI detector", snapshot.ai_score, accent)
    draw_score_row(draw, x + 34, metric_y + 108, "Layer 5 fingerprint", snapshot.fingerprint_score, accent)
    draw_score_row(draw, x + 34, metric_y + 216, "Cross-layer consistency", snapshot.consistency, "#0f766e")

    review = "필요" if snapshot.review_needed else "불필요"
    draw_metric_box(draw, x + 34, y + 598, 210, 84, "정밀 감정", review)
    draw_metric_box(draw, x + 270, y + 598, 216, 84, "Attribution", snapshot.fingerprint_method)


def draw_damage_card(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    high_damage: damage_score.DamageResult,
    low_damage: damage_score.DamageResult,
) -> None:
    accent = "#0f766e"
    draw.rounded_rectangle((x + 8, y + 10, x + w + 8, y + h + 10), radius=26, fill="#d8e0ea")
    draw.rounded_rectangle((x, y, x + w, y + h), radius=26, fill="#ffffff", outline="#d5dee9", width=2)
    draw.rounded_rectangle((x, y, x + w, y + 16), radius=8, fill=accent)
    draw.rounded_rectangle((x + 32, y + 42, x + 286, y + 88), radius=22, fill="#ecfdf5", outline=accent, width=2)
    draw.text((x + 54, y + 51), "GNN 피해 산정", fill=accent, font=font(21))

    draw.text((x + 32, y + 118), "유포 정황 반영", fill="#064e3b", font=font(38))
    draw_wrapped(
        draw,
        "파일 자체만 보지 않고 게시물 수, 플랫폼 수, 변형본, 폐쇄형 유포 여부를 함께 반영합니다.",
        x + 34,
        y + 176,
        w - 68,
        font(24),
        fill="#334155",
        line_spacing=8,
    )

    draw_metric_box(draw, x + 34, y + 286, 212, 94, "피해 등급", high_damage.grade_kr)
    draw_metric_box(draw, x + 270, y + 286, 216, 94, "피해 점수", f"{high_damage.total_score:.2f} / 30")
    draw_score_row(draw, x + 34, y + 414, "GNN 전파 위험도", high_damage.gnn_risk or 0.0, "#0f766e")
    draw_score_row(
        draw,
        x + 34,
        y + 522,
        "재유포 learned risk",
        high_damage.redistribution_learned_prob or 0.0,
        "#0f766e",
    )

    draw_metric_box(draw, x + 34, y + 612, 212, 86, "정황 없음", low_damage.grade_kr)
    draw_metric_box(draw, x + 270, y + 612, 216, 86, "피해 점수", f"{low_damage.total_score:.2f} / 30")


def draw_score_row(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    score: float,
    color: str,
) -> None:
    draw.text((x, y), label, fill="#526071", font=font(22))
    draw.text((x + 360, y - 4), f"{score:.3f}", fill="#172033", font=font(28))
    bar_x, bar_y, bar_w, bar_h = x, y + 42, 452, 22
    draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), radius=11, fill="#e8edf4")
    filled = max(0, min(bar_w, int(bar_w * score)))
    if filled > 0:
        draw.rounded_rectangle((bar_x, bar_y, bar_x + filled, bar_y + bar_h), radius=11, fill=color)


def draw_metric_box(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    value: str,
) -> None:
    draw.rounded_rectangle((x, y, x + w, y + h), radius=16, fill="#f8fafc", outline="#d7e1ed", width=2)
    draw.text((x + 18, y + 14), label, fill="#64748b", font=font(19))
    draw_wrapped(draw, value, x + 18, y + 42, w - 36, font(22), fill="#172033", line_spacing=4, max_lines=2)


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    width: int,
    text_font: ImageFont.FreeTypeFont,
    *,
    fill: str,
    line_spacing: int = 6,
    max_lines: int | None = None,
) -> int:
    lines: list[str] = []
    for paragraph in text.splitlines():
        wrapped = wrap_line(draw, paragraph, width, text_font)
        lines.extend(wrapped or [""])
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .") + "..."
    current_y = y
    for line in lines:
        draw.text((x, current_y), line, fill=fill, font=text_font)
        bbox = draw.textbbox((x, current_y), line or "가", font=text_font)
        current_y += (bbox[3] - bbox[1]) + line_spacing
    return current_y


def wrap_line(
    draw: ImageDraw.ImageDraw,
    text: str,
    width: int,
    text_font: ImageFont.FreeTypeFont,
) -> list[str]:
    if not text:
        return []
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=text_font)[2] <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    if not lines:
        return textwrap.wrap(text, width=18)
    return lines


def font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
