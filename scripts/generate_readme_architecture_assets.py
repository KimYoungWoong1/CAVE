"""Generate README architecture diagrams for CAVE.

The assets are static visual explanations of the implemented pipeline and
Layer 7 harm propagation graph. They do not embed any dataset media.
"""
from __future__ import annotations

import math
import sys
import json
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
    metrics = read_pipeline_metrics()
    img = Image.new("RGB", (1600, 2380), "#050607")
    draw = ImageDraw.Draw(img)

    center_text(draw, 800, 52, "CAVE 모델 파이프라인", font(50), "#f8fafc")
    center_text(draw, 800, 118, "Credibility Audit for AI-Generated Evidence", font(24), "#94a3b8")

    x = 800
    input_box = (430, 180, 1170, 310)
    l1_box = (420, 380, 765, 510)
    l2_box = (835, 380, 1180, 510)
    l3_box = (430, 660, 1170, 790)
    l4_box = (430, 860, 1170, 990)
    l5_box = (430, 1060, 1170, 1190)
    l6_box = (430, 1260, 1170, 1390)
    authentic_box = (115, 1510, 465, 1635)
    clash_box = (600, 1510, 1000, 1635)
    suspected_box = (1135, 1510, 1485, 1635)
    l7_box = (430, 1815, 1170, 1945)
    output_box = (430, 2030, 1170, 2160)
    metrics_box = (110, 2190, 1490, 2345)

    pipeline_box(draw, input_box, "디지털 파일 입력", "이미지 · 영상", fill="#f5f2ea", outline="#e5e7eb")
    down_arrow(draw, x, 310, 380)

    layer_label(draw, 86, 440, "출처\n레이어")
    dashed_guide(draw, 190, 445, 405, 445)
    pipeline_box(draw, l1_box, "L1 · C2PA 검증", "출처 기록 · 편집 이력", fill="#e7f2ff", outline="#38bdf8", title_color="#12467a")
    pipeline_box(draw, l2_box, "L2 · 워터마킹", "AI 마커 · 메타 신호", fill="#e7f2ff", outline="#38bdf8", title_color="#12467a")
    center_text(draw, 592, 535, "암호화 표준", font(21), "#a3e635")
    center_text(draw, 1008, 535, "신호처리", font(21), "#a3e635")
    connect_to_center(draw, l1_box, x, 610)
    connect_to_center(draw, l2_box, x, 610)
    down_arrow(draw, x, 610, 660)

    layer_label(draw, 86, 725, "AI\n탐지")
    dashed_guide(draw, 190, 735, 405, 735)
    pipeline_box(draw, l3_box, "L3 · AI 탐지 모델", "SDXL detector · GenImage · EfficientNet/Xception/R3D", fill="#ecebff", outline="#8b5cf6", title_color="#4338ca")
    down_arrow(draw, x, 790, 860)
    pipeline_box(draw, l4_box, "L4 · rPPG 생체 신호", "CHROM rPPG + RandomForest · 영상 전용", fill="#ecebff", outline="#8b5cf6", title_color="#4338ca")
    down_arrow(draw, x, 990, 1060)
    pipeline_box(draw, l5_box, "L5 · 생성 모델 핑거프린트", "FFT · 잔차 노이즈 · RF attribution", fill="#ecebff", outline="#8b5cf6", title_color="#4338ca")
    down_arrow(draw, x, 1190, 1260)

    layer_label(draw, 86, 1328, "통합\n판정")
    dashed_guide(draw, 190, 1330, 405, 1330)
    pipeline_box(draw, l6_box, "L6 · Cross-layer Audit", "Cosine similarity · Integrity Clash 탐지", fill="#dcfce7", outline="#34d399", title_color="#047857")
    down_arrow(draw, x, 1390, 1452)
    draw.line((280, 1452, 1320, 1452), fill="#9ca3af", width=2)
    branch_arrow(draw, 280, 1452, center_top(authentic_box))
    branch_arrow(draw, x, 1452, center_top(clash_box))
    branch_arrow(draw, 1320, 1452, center_top(suspected_box))
    pipeline_box(draw, authentic_box, "진본 가능성", "레이어 일관 일치", fill="#dcfce7", outline="#34d399", title_color="#047857", title_size=29, body_size=22)
    pipeline_box(draw, clash_box, "Integrity Clash", "전문가 정밀 감정 권고", fill="#fef3c7", outline="#f59e0b", title_color="#92400e", title_size=28, body_size=22)
    pipeline_box(draw, suspected_box, "AI 생성 의심", "위조 가능성 높음", fill="#fee2e2", outline="#fb7185", title_color="#7f1d1d", title_size=28, body_size=22)
    draw.line((280, 1760, 1320, 1760), fill="#9ca3af", width=2)
    draw.line((280, 1635, 280, 1760), fill="#9ca3af", width=2)
    draw.line((800, 1635, 800, 1760), fill="#9ca3af", width=2)
    draw.line((1320, 1635, 1320, 1760), fill="#9ca3af", width=2)
    down_arrow(draw, x, 1760, 1815)

    layer_label(draw, 86, 1870, "피해\n산정")
    dashed_guide(draw, 190, 1875, 405, 1875)
    pipeline_box(draw, l7_box, "L7 · 피해 규모 정량화", "GNN 확산 추적 · 재유포 RF · 가중 합산 30점", fill="#ecebff", outline="#8b5cf6", title_color="#4338ca")
    down_arrow(draw, x, 1945, 2030)

    layer_label(draw, 86, 2085, "출력")
    dashed_guide(draw, 190, 2095, 405, 2095)
    pipeline_box(draw, output_box, "L8 · 최종 결과 요약", "판정 근거 · 피해 점수 · 정밀 감정 권고", fill="#f5f2ea", outline="#e5e7eb")

    draw.rounded_rectangle(metrics_box, radius=18, fill="#050607", outline="#e5e7eb", width=2)
    center_text(draw, 800, 2210, "주요 평가결과 (RedFace · FF++ C23 · Tiny-GenImage)", font(20), "#a3e635")
    metric_text(draw, 290, 2250, "L3 이미지 탐지", metrics["l3_auc"])
    metric_text(draw, 560, 2250, "L5 핑거프린트", metrics["l5_auc"])
    metric_text(draw, 860, 2250, "L4 rPPG", metrics["l4_auc"])
    metric_text(draw, 1160, 2250, "Attribution", metrics["attribution"])
    center_text(draw, 800, 2320, "보조 신호로만 활용 / 법적 확정 판정기 아님", font(20), "#a3e635")

    return img


def read_pipeline_metrics() -> dict[str, str]:
    image_calibration = read_json("image_calibration.json")
    fingerprint_meta = read_json("fingerprint_classifier.meta.json")
    rppg_meta = read_json("rppg_classifier.meta.json")
    return {
        "l3_auc": f"AUC {image_calibration.get('training_metrics', {}).get('auc', 0.0):.3f}",
        "l5_auc": f"AUC {fingerprint_meta.get('test_auc', 0.0):.3f}",
        "l4_auc": f"AUC {rppg_meta.get('test_auc', 0.0):.3f}",
        "attribution": f"{fingerprint_meta.get('method_accuracy', 0.0) * 100:.1f}%",
    }


def read_json(filename: str) -> dict:
    path = ROOT / "models" / filename
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def pipeline_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title_text: str,
    body: str,
    *,
    fill: str,
    outline: str,
    title_color: str = "#111827",
    title_size: int = 32,
    body_size: int = 24,
) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle((x1 + 5, y1 + 6, x2 + 5, y2 + 6), radius=18, fill="#27272a")
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=2)
    center_text(draw, (x1 + x2) // 2, y1 + 30, title_text, font(title_size), title_color)
    center_text(draw, (x1 + x2) // 2, y1 + 76, body, font(body_size), "#4b5563")


def center_text(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    text_font: ImageFont.FreeTypeFont,
    fill: str,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=text_font)
    draw.text((x - (bbox[2] - bbox[0]) / 2, y), text, fill=fill, font=text_font)


def metric_text(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, value: str) -> None:
    center_text(draw, x, y, label, font(19), "#a3e635")
    center_text(draw, x, y + 32, value, font(27), "#2563eb")


def layer_label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str) -> None:
    for index, line in enumerate(text.splitlines()):
        draw.text((x, y + index * 34), line, fill="#a3e635", font=font(22))


def dashed_guide(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int) -> None:
    segment = 12
    gap = 12
    x = x1
    while x < x2:
        draw.line((x, y1, min(x + segment, x2), y2), fill="#d1d5db", width=2)
        x += segment + gap


def down_arrow(draw: ImageDraw.ImageDraw, x: int, y1: int, y2: int) -> None:
    arrow(draw, (x, y1), (x, y2), "#6b7280", width=3)


def connect_to_center(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], x: int, y: int) -> None:
    sx = (box[0] + box[2]) // 2
    sy = box[3]
    draw.line((sx, sy, sx, y - 50), fill="#9ca3af", width=2)
    draw.line((sx, y - 50, x, y - 50), fill="#9ca3af", width=2)


def branch_arrow(draw: ImageDraw.ImageDraw, x: int, y: int, end: tuple[int, int]) -> None:
    ex, ey = end
    draw.line((x, y, x, ey - 24), fill="#9ca3af", width=2)
    arrow(draw, (x, ey - 24), (ex, ey), "#9ca3af", width=2)


def center_top(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return ((box[0] + box[2]) // 2, box[1])


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
