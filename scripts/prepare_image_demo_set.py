"""이미지 웹 데모용 샘플 세트를 symlink로 구성.

실행:
  python scripts/prepare_image_demo_set.py --overwrite
"""
from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
METHODS = ("EFS", "FAM", "FR", "FS")


@dataclass
class DemoItem:
    label: str
    method: str
    source: Path
    target: Path


def main() -> None:
    parser = argparse.ArgumentParser(description="CAVE 이미지 데모 세트 준비")
    parser.add_argument("--output", default="test_data/demo_images")
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

    print(f"demo image set: {output}")
    for label in ("real", "fake", "ai_generated"):
        count = sum(1 for item in items if item.label == label)
        print(f"  {label}: {count}")


def _collect_items(output: Path) -> list[DemoItem]:
    items: list[DemoItem] = []

    real_root = ROOT / "test_data/redface/eval/real"
    real = _preferred_image(real_root, "000918.jpg") or _first_image(real_root)
    if real is not None:
        items.append(DemoItem("real", "Original", real, output / "real" / "redface_real.jpg"))

    fake_root = ROOT / "test_data/redface/eval/fake"
    for method in METHODS:
        fake = _first_image(fake_root, prefix=f"{method}_")
        if fake is not None:
            name = f"redface_{method.lower()}_fake.jpg"
            items.append(DemoItem("fake", method, fake, output / "fake" / name))

    generated_root = ROOT / "test_data/ai_generated"
    for source_name, target_name in (
        ("img_001_firefly.jpg", "firefly_img_001.jpg"),
        ("img_002_firefly.jpg", "firefly_img_002.jpg"),
    ):
        source = generated_root / source_name
        if source.exists():
            items.append(DemoItem("ai_generated", "Firefly", source, output / "ai_generated" / target_name))

    if not items:
        raise SystemExit("데모 세트에 넣을 이미지가 없습니다. RedFace/ai_generated 경로를 확인하세요.")
    return items


def _first_image(root: Path, prefix: str = "") -> Path | None:
    if not root.exists():
        return None
    for path in sorted(root.iterdir()):
        if path.name.startswith(prefix) and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path
    return None


def _preferred_image(root: Path, filename: str) -> Path | None:
    path = root / filename
    if path.exists() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return path
    return None


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source.resolve())


def _write_manifest(path: Path, items: list[DemoItem]) -> None:
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


def _write_readme(path: Path, items: list[DemoItem]) -> None:
    lines = [
        "# CAVE 이미지 데모 세트",
        "",
        "웹 데모와 Cross-layer Audit 검증에 쓰는 작은 이미지 샘플 묶음입니다.",
        "원본 데이터는 복사하지 않고 symlink로 연결합니다.",
        "",
        "## 구성",
        "",
        "- `real/`: RedFace Original 기반 진본 얼굴 이미지",
        "- `fake/`: RedFace EFS/FAM/FR/FS 조작 방식별 이미지",
        "- `ai_generated/`: Firefly 생성 이미지",
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
        "python scripts/evaluate_image_audit.py --input-dir test_data/demo_images",
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
