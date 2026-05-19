"""Hugging Face Tiny-GenImage를 CAVE GenImage split 구조로 저장.

대용량 원본 GenImage 전체를 받지 않고, streaming으로 필요한 샘플만 내려받아
다음 구조로 저장한다.

test_data/genimage/
  calibration/{real,ai}
  eval/{real,ai}
  holdout/{real,ai}

기본 설정은 발표/로컬 데모용으로 가볍게 잡았다. 더 많이 쓰고 싶으면
per-generator 값을 올리면 된다.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


DEFAULT_DATASET_ID = "TheKernel01/Tiny-GenImage"
DEFAULT_GENERATORS = "Midjourney,SD15,GLIDE,Wukong,VQDM"
GENERATOR_LABELS = {
    0: "Real",
    1: "ADM",
    2: "BigGAN",
    3: "GLIDE",
    4: "Midjourney",
    5: "SD14",
    6: "SD15",
    7: "VQDM",
    8: "Wukong",
}
TINY_AVAILABLE_GENERATORS = {"ADM", "BigGAN", "GLIDE", "Midjourney", "SD15", "VQDM", "Wukong"}
LABEL_NAMES = {
    0: "real",
    1: "ai",
}


@dataclass
class Row:
    split: str
    label: str
    generator: str
    source_path: str
    target_path: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny-GenImage -> CAVE split 다운로드")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--output", default="test_data/genimage")
    parser.add_argument(
        "--generators",
        default=DEFAULT_GENERATORS,
        help="쉼표 구분 generator명. all이면 Real을 제외한 전체 generator 사용.",
    )
    parser.add_argument("--calibration-per-generator", type=int, default=120)
    parser.add_argument("--eval-per-generator", type=int, default=40)
    parser.add_argument("--holdout-per-generator", type=int, default=40)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-scan-rows", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if args.overwrite and output.exists():
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"출력 폴더가 이미 있습니다. 덮어쓰려면 --overwrite를 사용하세요: {output}")

    selected_generators = _parse_generators(args.generators)
    rows: list[Row] = []
    rows.extend(_collect_split(
        dataset_id=args.dataset_id,
        hf_split="train",
        output=output,
        split_targets={"calibration": args.calibration_per_generator},
        selected_generators=selected_generators,
        jpeg_quality=args.jpeg_quality,
        max_scan_rows=args.max_scan_rows,
    ))
    rows.extend(_collect_split(
        dataset_id=args.dataset_id,
        hf_split="validation",
        output=output,
        split_targets={
            "eval": args.eval_per_generator,
            "holdout": args.holdout_per_generator,
        },
        selected_generators=selected_generators,
        jpeg_quality=args.jpeg_quality,
        max_scan_rows=args.max_scan_rows,
    ))

    _write_manifest(output / "manifest.csv", rows)
    _print_summary(rows)


def _collect_split(
    dataset_id: str,
    hf_split: str,
    output: Path,
    split_targets: dict[str, int],
    selected_generators: set[str],
    jpeg_quality: int,
    max_scan_rows: int,
) -> list[Row]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "datasets 패키지가 필요합니다. `pip install datasets` 또는 "
            "`pip install -r requirements.txt` 후 다시 실행하세요."
        ) from exc

    dataset = load_dataset(dataset_id, split=hf_split, streaming=True)
    ai_counts = {
        split: {generator: 0 for generator in selected_generators}
        for split in split_targets
    }
    real_targets = {
        split: len(selected_generators) * target
        for split, target in split_targets.items()
    }
    real_counts = {split: 0 for split in split_targets}
    rows: list[Row] = []

    for index, item in enumerate(dataset):
        if max_scan_rows > 0 and index >= max_scan_rows:
            break
        label = _label_name(item.get("label"))
        generator = _generator_name(item.get("generator"))

        assigned_split = None
        if label == "ai":
            if generator not in selected_generators:
                continue
            for split, target in split_targets.items():
                if ai_counts[split][generator] < target:
                    assigned_split = split
                    ai_counts[split][generator] += 1
                    break
        elif label == "real":
            generator = "Real"
            for split, target in real_targets.items():
                if real_counts[split] < target:
                    assigned_split = split
                    real_counts[split] += 1
                    break

        if assigned_split is None:
            if _targets_done(ai_counts, real_counts, split_targets, real_targets):
                break
            continue

        target = _target_path(output, assigned_split, label, generator, index)
        _save_image(item["image"], target, jpeg_quality)
        rows.append(Row(
            split=assigned_split,
            label=label,
            generator=generator,
            source_path=f"{dataset_id}:{hf_split}:{index}",
            target_path=str(target),
        ))

        if len(rows) % 250 == 0:
            print(f"[{hf_split}] saved {len(rows)} images...")
        if _targets_done(ai_counts, real_counts, split_targets, real_targets):
            break

    _warn_missing(hf_split, ai_counts, real_counts, split_targets, real_targets)
    return rows


def _parse_generators(value: str) -> set[str]:
    if value.strip().lower() == "all":
        return set(TINY_AVAILABLE_GENERATORS)
    return {_safe_name(item.strip()) for item in value.split(",") if item.strip()}


def _label_name(value) -> str:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in ("real", "nature", "0"):
            return "real"
        if lowered in ("fake", "ai", "generated", "1"):
            return "ai"
    try:
        return LABEL_NAMES.get(int(value), "unknown")
    except Exception:
        return "unknown"


def _generator_name(value) -> str:
    if isinstance(value, str):
        return _safe_name(value)
    try:
        return _safe_name(GENERATOR_LABELS.get(int(value), str(value)))
    except Exception:
        return "unknown"


def _target_path(output: Path, split: str, label: str, generator: str, index: int) -> Path:
    name = f"{_safe_name(generator)}__tiny_genimage_{index:07d}.jpg"
    return output / split / label / name


def _save_image(image: Image.Image, target: Path, jpeg_quality: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image.convert("RGB").save(target, format="JPEG", quality=jpeg_quality, optimize=True)


def _targets_done(
    ai_counts: dict[str, dict[str, int]],
    real_counts: dict[str, int],
    split_targets: dict[str, int],
    real_targets: dict[str, int],
) -> bool:
    for split, target in split_targets.items():
        if real_counts[split] < real_targets[split]:
            return False
        if any(count < target for count in ai_counts[split].values()):
            return False
    return True


def _warn_missing(
    hf_split: str,
    ai_counts: dict[str, dict[str, int]],
    real_counts: dict[str, int],
    split_targets: dict[str, int],
    real_targets: dict[str, int],
) -> None:
    for split, target in split_targets.items():
        if real_counts[split] < real_targets[split]:
            print(f"[warn] {hf_split}/{split} real {real_counts[split]}/{real_targets[split]}")
        for generator, count in sorted(ai_counts[split].items()):
            if count < target:
                print(f"[warn] {hf_split}/{split} {generator} {count}/{target}")


def _write_manifest(path: Path, rows: Iterable[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["split", "label", "generator", "source_path", "target_path"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _print_summary(rows: list[Row]) -> None:
    counts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = (row.split, row.label, row.generator)
        counts[key] = counts.get(key, 0) + 1

    print("Tiny-GenImage summary")
    for split in ("calibration", "eval", "holdout"):
        split_total = sum(value for key, value in counts.items() if key[0] == split)
        print(f"  {split}: {split_total}")
        for key, value in sorted(counts.items()):
            if key[0] == split:
                print(f"    {key[1]},{key[2]}: {value}")


def _safe_name(name: str) -> str:
    return (
        name.strip()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "")
    )


if __name__ == "__main__":
    main()
