"""Generate README architecture diagrams for CAVE.

The assets are static visual explanations of the implemented pipeline and
Layer 7 harm propagation graph. They do not embed any dataset media.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layers import damage_score  # noqa: E402


ASSET_DIR = ROOT / "docs" / "assets"
PIPELINE_PATH = ASSET_DIR / "cave_pipeline.png"
GNN_PATH = ASSET_DIR / "gnn_spread_graph.png"


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    damage = damage_score.run(
        damage_score.DamageInputs.example_deepfake_sexual(),
        crime_type="deepfake_sexual",
    )
    render_pipeline().save(PIPELINE_PATH)
    render_gnn_graph(damage).save(GNN_PATH)
    print(f"Wrote {PIPELINE_PATH.relative_to(ROOT)}")
    print(f"Wrote {GNN_PATH.relative_to(ROOT)}")


def render_pipeline() -> Image.Image:
    img = Image.new("RGB", (1800, 1000), "#f6f8fb")
    draw = ImageDraw.Draw(img)

    title(draw, "CAVE 모델 파이프라인", "파일 신뢰성 감사와 피해 확산 산정을 하나의 로컬 분석 흐름으로 연결")

    media = (70, 445, 292, 595)
    group = (365, 190, 1145, 780)
    audit = (1235, 245, 1515, 395)
    harm = (1235, 555, 1515, 705)
    output = (1235, 790, 1730, 890)

    draw_node(draw, media, "Media Upload", "이미지·영상", "#ffffff", "#0f766e", icon="01")

    draw.rounded_rectangle((group[0] + 8, group[1] + 10, group[2] + 8, group[3] + 10), radius=30, fill="#d8e0ea")
    draw.rounded_rectangle(group, radius=30, fill="#ffffff", outline="#d5dee9", width=2)
    draw.text((400, 220), "Evidence Layers", fill="#14213d", font=font(34))
    draw.text((402, 265), "출처 신호와 AI 탐지 신호를 분리해 생성한 뒤 Layer 6에서 교차 검증합니다.", fill="#526071", font=font(22))

    draw.text((405, 330), "출처·무결성", fill="#526071", font=font(23))
    l1 = draw_compact_card(draw, (405, 365, 625, 475), "L1", "C2PA", "Provenance", "#64748b", "#ffffff")
    l2 = draw_compact_card(draw, (675, 365, 945, 475), "L2", "Watermark", "Metadata signal", "#64748b", "#ffffff")
    arrow(draw, mid_right(l1), mid_left(l2), "#94a3b8", width=4)

    draw.text((405, 520), "AI 기반 탐지", fill="#0f766e", font=font(23))
    l3 = draw_compact_card(draw, (405, 555, 605, 665), "L3", "AI Detector", "Image/video ensemble", "#0f766e", "#ecfdf5")
    l4 = draw_compact_card(draw, (645, 555, 845, 665), "L4", "rPPG", "RF classifier", "#0f766e", "#ecfdf5")
    l5 = draw_compact_card(draw, (885, 555, 1110, 665), "L5", "Fingerprint", "Generator attribution", "#0f766e", "#ecfdf5")
    arrow(draw, mid_right(l3), mid_left(l4), "#0f766e", width=4)
    arrow(draw, mid_right(l4), mid_left(l5), "#0f766e", width=4)

    draw.rounded_rectangle((405, 705, 1110, 745), radius=18, fill="#fff7ed", outline="#fed7aa", width=2)
    draw.text((430, 715), "Image: diffusion + GenImage + RedFace · Video: face crop + rPPG + fingerprint", fill="#9a3412", font=font(19))

    draw_node(draw, audit, "Layer 6", "Cross-layer Audit", "#eef2ff", "#475569", icon="AUD")
    draw_node(draw, harm, "Layer 7", "GNN Harm Assessment", "#ecfdf5", "#0f766e", icon="GNN")
    draw_node(draw, output, "Output", "Verdict · Evidence · Harm Score", "#ffffff", "#b45309", icon="UI")

    arrow(draw, mid_right(media), mid_left(group), "#94a3b8", width=5)
    arrow(draw, mid_right(group), mid_left(audit), "#64748b", width=5)
    arrow(draw, mid_bottom(audit), mid_top(harm), "#0f766e", width=5)
    arrow(draw, mid_bottom(harm), mid_top(output), "#0f766e", width=5)

    footer(draw, "AI 레이어: Layer 3, 4, 5, 7 · Layer 6은 출처/탐지 신호를 통합하는 판정 레이어", y=900)
    return img


def render_gnn_graph(damage: damage_score.DamageResult) -> Image.Image:
    img = Image.new("RGB", (1800, 1040), "#f6f8fb")
    draw = ImageDraw.Draw(img)

    title(draw, "Layer 7 GNN 피해 확산 시각화", "유포 정황 입력을 approximate propagation graph로 변환해 전파 위험도를 산정")

    graph_box = (70, 190, 1240, 880)
    metric_box = (1280, 190, 1730, 880)
    draw.rounded_rectangle(graph_box, radius=28, fill="#ffffff", outline="#d5dee9", width=2)
    draw.rounded_rectangle(metric_box, radius=28, fill="#ffffff", outline="#d5dee9", width=2)
    draw.text((100, 220), "입력 정황 기반 유포 그래프", fill="#0f4f4a", font=font(30))
    draw.text((1314, 220), "GNN/RF 산정 결과", fill="#0f4f4a", font=font(30))

    nodes = {
        "root": ((215, 515), "의심\n이미지·영상", "#fff7ed", "#b45309", 78),
        "telegram": ((515, 300), "Telegram\n폐쇄형", "#fee2e2", "#b91c1c", 70),
        "x": ((515, 515), "X\n공개 확산", "#e0f2fe", "#0369a1", 64),
        "community": ((515, 730), "온라인\n커뮤니티", "#ecfdf5", "#047857", 64),
        "variant": ((855, 260), "변형본\n존재", "#fee2e2", "#b91c1c", 64),
        "reupload": ((855, 515), "재업로드\n80건", "#fff7ed", "#b45309", 68),
        "evidence": ((855, 740), "URL·캡처\n7건 보전", "#eef2ff", "#475569", 64),
        "risk": ((1120, 515), "확산 위험\n집계", "#ecfdf5", "#0f766e", 76),
    }

    graph_edges = [
        ("root", "telegram", "폐쇄형 유포", 7, "#b91c1c"),
        ("root", "x", "공개 공유", 5, "#0369a1"),
        ("root", "community", "커뮤니티 게시", 5, "#047857"),
        ("telegram", "variant", "변형 생성", 6, "#b91c1c"),
        ("x", "reupload", "공유 1,200", 6, "#b45309"),
        ("community", "evidence", "조회 45,000", 4, "#64748b"),
        ("variant", "risk", "", 5, "#b91c1c"),
        ("reupload", "risk", "", 6, "#b45309"),
        ("evidence", "risk", "", 4, "#64748b"),
    ]

    for source, target, label, width, color in graph_edges:
        start = nodes[source][0]
        end = nodes[target][0]
        arrow(draw, start, end, color, width=width, shorten=72)
        if label:
            label_edge(draw, start, end, label, color)

    for center, text, fill, outline, radius in nodes.values():
        draw_circle_node(draw, center, radius, text, fill, outline)

    metric(draw, 1320, 300, "GNN 전파 위험도", f"{damage.gnn_risk:.3f}", "#0f766e")
    metric(draw, 1320, 405, "GNN 환산", f"{damage.gnn_converted_score:.3f} / 5", "#0f766e")
    metric(draw, 1320, 510, "Heuristic 확산", f"{damage.heuristic_diffusion:.3f} / 5", "#64748b")
    metric(draw, 1320, 615, "Blend", damage.blend_ratio, "#475569")
    metric(draw, 1320, 720, "피해 총점", f"{damage.total_score:.2f} / 30", "#b45309")

    draw.rounded_rectangle((100, 800, 1210, 852), radius=18, fill="#e8f3f2", outline="#b8deda", width=2)
    draw.text(
        (130, 814),
        "게시물 80건 · 플랫폼 3개 · 공유 1,200회 · 조회 45,000회 · 변형본/폐쇄형 유포 반영",
        fill="#0f4f4a",
        font=font(22),
    )

    footer(draw, "주의: 실제 SNS 크롤링 그래프가 아니라, 사용자가 입력한 유포 정황을 기반으로 구성한 분석용 graph representation입니다.")
    return img


def title(draw: ImageDraw.ImageDraw, main: str, sub: str) -> None:
    draw.text((70, 52), main, fill="#14213d", font=font(52))
    draw.text((74, 124), sub, fill="#526071", font=font(25))


def footer(draw: ImageDraw.ImageDraw, text: str, *, y: int = 912) -> None:
    draw.rounded_rectangle((70, y, 1730, y + 60), radius=20, fill="#e8f3f2", outline="#b8deda", width=2)
    draw.text((102, y + 18), text, fill="#0f4f4a", font=font(22))


def draw_compact_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    code: str,
    title_text: str,
    body: str,
    accent: str,
    fill: str,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle((x1 + 6, y1 + 7, x2 + 6, y2 + 7), radius=18, fill="#d8e0ea")
    draw.rounded_rectangle(box, radius=18, fill=fill, outline="#d5dee9", width=2)
    draw.rounded_rectangle((x1, y1, x1 + 54, y2), radius=18, fill=accent)
    draw.text((x1 + 15, y1 + 38), code, fill="#ffffff", font=font(20))
    draw.text((x1 + 74, y1 + 22), title_text, fill=accent, font=font(23))
    draw_wrapped(draw, body, x1 + 74, y1 + 56, x2 - x1 - 94, font(20), fill="#172033", line_spacing=3)
    return box


def draw_node(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    value: str,
    fill: str,
    outline: str,
    *,
    icon: str,
) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle((x1 + 8, y1 + 8, x2 + 8, y2 + 8), radius=24, fill="#d8e0ea")
    draw.rounded_rectangle(box, radius=24, fill=fill, outline="#d5dee9", width=2)
    draw.rounded_rectangle((x1, y1, x1 + 62, y2), radius=22, fill=outline)
    draw.text((x1 + 16, y1 + 48), icon, fill="#ffffff", font=font(22))
    draw.text((x1 + 86, y1 + 30), label, fill=outline, font=font(23))
    draw_wrapped(draw, value, x1 + 86, y1 + 68, x2 - x1 - 112, font(25), fill="#172033")


def callout(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    body: str,
    fill: str,
    outline: str,
) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=22, fill=fill, outline="#d5dee9", width=2)
    draw.text((x1 + 24, y1 + 22), label, fill=outline, font=font(23))
    draw_wrapped(draw, body, x1 + 24, y1 + 62, x2 - x1 - 48, font(21), fill="#334155")


def draw_circle_node(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    radius: int,
    text: str,
    fill: str,
    outline: str,
) -> None:
    x, y = center
    draw.ellipse((x - radius + 6, y - radius + 8, x + radius + 6, y + radius + 8), fill="#d8e0ea")
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=4)
    lines = text.splitlines()
    line_height = 27
    start_y = y - (len(lines) * line_height) // 2
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font(23))
        draw.text((x - (bbox[2] - bbox[0]) / 2, start_y + i * line_height), line, fill="#172033", font=font(23))


def metric(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, value: str, color: str) -> None:
    draw.text((x, y), label, fill="#526071", font=font(22))
    draw.text((x, y + 32), value, fill="#172033", font=font(34))
    draw.rounded_rectangle((x, y + 80, x + 342, y + 100), radius=10, fill="#e8edf4")
    ratio = metric_ratio(value)
    draw.rounded_rectangle((x, y + 80, x + int(342 * ratio), y + 100), radius=10, fill=color)


def metric_ratio(value: str) -> float:
    if ":" in value:
        first = value.split(":", 1)[0]
        try:
            return max(0.0, min(1.0, float(first) / 100.0))
        except ValueError:
            return 0.65
    try:
        number = float(value.split()[0])
    except (ValueError, IndexError):
        return 0.65
    if "/ 30" in value:
        return max(0.0, min(1.0, number / 30.0))
    if "/ 5" in value:
        return max(0.0, min(1.0, number / 5.0))
    return max(0.0, min(1.0, number))


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str,
    *,
    width: int = 4,
    shorten: int = 0,
) -> None:
    sx, sy = start
    ex, ey = end
    if shorten:
        dx = ex - sx
        dy = ey - sy
        dist = math.hypot(dx, dy) or 1.0
        sx += int(dx / dist * shorten)
        sy += int(dy / dist * shorten)
        ex -= int(dx / dist * shorten)
        ey -= int(dy / dist * shorten)
    draw.line((sx, sy, ex, ey), fill=color, width=width)
    angle = math.atan2(ey - sy, ex - sx)
    size = 14 + width
    points = [
        (ex, ey),
        (ex - size * math.cos(angle - 0.45), ey - size * math.sin(angle - 0.45)),
        (ex - size * math.cos(angle + 0.45), ey - size * math.sin(angle + 0.45)),
    ]
    draw.polygon(points, fill=color)


def label_edge(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], text: str, color: str) -> None:
    x = (start[0] + end[0]) // 2
    y = (start[1] + end[1]) // 2
    bbox = draw.textbbox((0, 0), text, font=font(18))
    pad_x, pad_y = 12, 5
    draw.rounded_rectangle(
        (x - (bbox[2] - bbox[0]) // 2 - pad_x, y - 17, x + (bbox[2] - bbox[0]) // 2 + pad_x, y + 17),
        radius=12,
        fill="#ffffff",
        outline="#d5dee9",
    )
    draw.text((x - (bbox[2] - bbox[0]) // 2, y - 12), text, fill=color, font=font(18))


def mid_left(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return (box[0], (box[1] + box[3]) // 2)


def mid_right(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return (box[2], (box[1] + box[3]) // 2)


def mid_bottom(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return ((box[0] + box[2]) // 2, box[3])


def mid_top(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return ((box[0] + box[2]) // 2, box[1])


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
) -> None:
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
    current_y = y
    for line in lines:
        draw.text((x, current_y), line, fill=fill, font=text_font)
        bbox = draw.textbbox((0, current_y), line, font=text_font)
        current_y += (bbox[3] - bbox[1]) + line_spacing


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
