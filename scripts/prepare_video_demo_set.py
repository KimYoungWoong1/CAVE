"""영상 웹 데모용 샘플 세트를 symlink로 구성.

실행:
  python scripts/prepare_video_demo_set.py --overwrite
"""
from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass
class DemoVideo:
    label: str
    method: str
    source: Path
    target: Path


PREFERRED_REAL = ("578.mp4", "015.mp4")
PREFERRED_FAKE = {
    "DeepFakeDetection": "DeepFakeDetection_03_07__walking_down_indoor_hall_disgust__PWXXULHR.mp4",
    "Deepfakes": "Deepfakes_169_227.mp4",
    "FaceSwap": "FaceSwap_321_288.mp4",
    "NeuralTextures": "NeuralTextures_715_721.mp4",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="CAVE 영상 데모 세트 준비")
    parser.add_argument("--output", default="test_data/demo_videos")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = ROOT / args.output
    if args.overwrite and output.exists():
        shutil.rmtree(output)

    items = _collect_items(output)
    for item in items:
        _link(item.source, item.target)
    _write_manifest(output / "manifest.csv", items)
    _write_readme(output / "README.md", items)

    print(f"demo video set: {output}")
    for label in ("real", "deepfake"):
        count = sum(1 for item in items if item.label == label)
        print(f"  {label}: {count}")


def _collect_items(output: Path) -> list[DemoVideo]:
    items: list[DemoVideo] = []
    real_root = ROOT / "test_data/ffpp_c23/eval/real"
    fake_root = ROOT / "test_data/ffpp_c23/eval/deepfake"

    for filename in PREFERRED_REAL:
        source = _preferred_video(real_root, filename)
        if source is None:
            continue
        target = output / "real" / f"ffpp_real_{source.stem}.mp4"
        items.append(DemoVideo("real", "Original", source, target))
        break

    for method, filename in PREFERRED_FAKE.items():
        source = _preferred_video(fake_root, filename) or _first_video(fake_root, prefix=f"{method}_")
        if source is None:
            continue
        target = output / "deepfake" / f"{method.lower()}_{source.stem}.mp4"
        items.append(DemoVideo("deepfake", method, source, target))

    if not items:
        raise SystemExit("데모 세트에 넣을 영상이 없습니다. FFPP 경로를 확인하세요.")
    return items


def _preferred_video(root: Path, filename: str) -> Path | None:
    path = root / filename
    if path.exists() and path.suffix.lower() in VIDEO_EXTENSIONS:
        return path
    return None


def _first_video(root: Path, prefix: str = "") -> Path | None:
    if not root.exists():
        return None
    for path in sorted(root.iterdir()):
        if path.name.startswith(prefix) and path.suffix.lower() in VIDEO_EXTENSIONS:
            return path
    return None


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source.resolve())


def _write_manifest(path: Path, items: list[DemoVideo]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["label", "method", "target_path", "source_path"])
        writer.writeheader()
        for item in items:
            writer.writerow({
                "label": item.label,
                "method": item.method,
                "target_path": str(item.target.relative_to(ROOT)),
                "source_path": str(item.source.relative_to(ROOT)),
            })


def _write_readme(path: Path, items: list[DemoVideo]) -> None:
    lines = [
        "# CAVE 영상 데모 세트",
        "",
        "웹 데모와 영상 Cross-layer Audit 검증에 쓰는 작은 FFPP C23 샘플 묶음입니다.",
        "원본 영상은 복사하지 않고 symlink로 연결합니다.",
        "",
        "## 구성",
        "",
        "- `real/`: FFPP Original 기반 진본 영상",
        "- `deepfake/`: FFPP 조작 방식별 딥페이크 영상",
        "",
        "## 파일 목록",
        "",
    ]
    for item in items:
        lines.append(f"- `{item.target.relative_to(ROOT)}`: {item.label}, {item.method}")
    lines.extend([
        "",
        "## 검증",
        "",
        "```bash",
        "python scripts/evaluate_video_audit.py --real-dir test_data/demo_videos/real --fake-dir test_data/demo_videos/deepfake --max-per-label 0 --fake-per-method 0",
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
