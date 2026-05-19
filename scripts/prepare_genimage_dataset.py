"""GenImage 원본 폴더를 CAVE split 구조로 정리.

지원하는 일반 구조:

GenImage/
  Midjourney/
    train/{ai,nature}
    val/{ai,nature}
  Stable Diffusion V1.4/
    train/{ai,nature}
    val/{ai,nature}

출력:

test_data/genimage/
  calibration/{real,ai}
  eval/{real,ai}
  holdout/{real,ai}

실행 예:
  python scripts/prepare_genimage_dataset.py --source GenImage --overwrite
  python scripts/prepare_genimage_dataset.py --generators "Midjourney,Stable Diffusion V1.4,GLIDE"
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LABEL_DIRS = {
    "ai": ("ai", "fake", "generated", "synthetic"),
    "real": ("nature", "real", "0_real", "imagenet"),
}


@dataclass
class Row:
    split: str
    label: str
    generator: str
    source_path: str
    target_path: str


def main() -> None:
    parser = argparse.ArgumentParser(description="GenImage -> CAVE split 정리")
    parser.add_argument("--source", default="GenImage", help="GenImage 원본 폴더")
    parser.add_argument("--output", default="test_data/genimage", help="출력 폴더")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--generators", default="", help="쉼표로 구분한 generator 폴더명. 비우면 자동 탐색")
    parser.add_argument("--max-per-generator-split", type=int, default=1500)
    parser.add_argument("--eval-ratio-from-val", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if not source.exists():
        raise SystemExit(f"GenImage 원본 폴더를 찾을 수 없습니다: {source}")

    if args.overwrite and output.exists():
        shutil.rmtree(output)

    rng = random.Random(args.seed)
    generators = _selected_generators(source, args.generators)
    if not generators:
        raise SystemExit(f"GenImage generator 폴더를 찾지 못했습니다: {source}")

    rows: list[Row] = []
    for generator_dir in generators:
        rows.extend(_prepare_generator(
            generator_dir=generator_dir,
            output=output,
            mode=args.mode,
            rng=rng,
            max_per_split=args.max_per_generator_split,
            eval_ratio_from_val=args.eval_ratio_from_val,
        ))

    _write_manifest(output / "manifest.csv", rows)
    _print_summary(rows)


def _selected_generators(source: Path, requested: str) -> list[Path]:
    if requested.strip():
        result = []
        for name in requested.split(","):
            path = source / name.strip()
            if path.exists():
                result.append(path)
            else:
                print(f"[warn] generator 폴더 없음: {path}")
        return result

    generators = []
    for path in sorted(source.iterdir()):
        if not path.is_dir():
            continue
        if (path / "train").exists() or (path / "val").exists() or (path / "test").exists():
            generators.append(path)
    return generators


def _prepare_generator(
    generator_dir: Path,
    output: Path,
    mode: str,
    rng: random.Random,
    max_per_split: int,
    eval_ratio_from_val: float,
) -> list[Row]:
    generator = _safe_name(generator_dir.name)
    rows: list[Row] = []

    train_ai = _image_files(_find_label_dir(generator_dir / "train", "ai"))
    train_real = _image_files(_find_label_dir(generator_dir / "train", "real"))
    val_ai = _image_files(_find_label_dir(generator_dir / "val", "ai"))
    val_real = _image_files(_find_label_dir(generator_dir / "val", "real"))

    if not train_ai and not val_ai:
        print(f"[warn] AI 이미지 없음: {generator_dir}")
        return rows
    if not train_real and not val_real:
        print(f"[warn] real/nature 이미지 없음: {generator_dir}")
        return rows

    rows.extend(_materialize_split(
        images=_sample(train_real, max_per_split, rng),
        split="calibration",
        label="real",
        generator=generator,
        output=output,
        mode=mode,
    ))
    rows.extend(_materialize_split(
        images=_sample(train_ai, max_per_split, rng),
        split="calibration",
        label="ai",
        generator=generator,
        output=output,
        mode=mode,
    ))

    val_real_eval, val_real_holdout = _split_val(val_real, eval_ratio_from_val, rng)
    val_ai_eval, val_ai_holdout = _split_val(val_ai, eval_ratio_from_val, rng)
    rows.extend(_materialize_split(_sample(val_real_eval, max_per_split, rng), "eval", "real", generator, output, mode))
    rows.extend(_materialize_split(_sample(val_ai_eval, max_per_split, rng), "eval", "ai", generator, output, mode))
    rows.extend(_materialize_split(_sample(val_real_holdout, max_per_split, rng), "holdout", "real", generator, output, mode))
    rows.extend(_materialize_split(_sample(val_ai_holdout, max_per_split, rng), "holdout", "ai", generator, output, mode))
    return rows


def _find_label_dir(split_dir: Path, label: str) -> Path:
    for candidate in LABEL_DIRS[label]:
        path = split_dir / candidate
        if path.exists():
            return path
    return split_dir / LABEL_DIRS[label][0]


def _materialize_split(
    images: list[Path],
    split: str,
    label: str,
    generator: str,
    output: Path,
    mode: str,
) -> list[Row]:
    rows: list[Row] = []
    for path in images:
        target = output / split / label / f"{generator}__{path.name}"
        _materialize(path, target, mode)
        rows.append(Row(split, label, generator, str(path), str(target)))
    return rows


def _image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _sample(items: list[Path], max_count: int, rng: random.Random) -> list[Path]:
    items = list(items)
    rng.shuffle(items)
    if max_count > 0:
        return items[:max_count]
    return items


def _split_val(items: list[Path], eval_ratio: float, rng: random.Random) -> tuple[list[Path], list[Path]]:
    items = list(items)
    rng.shuffle(items)
    eval_count = int(len(items) * max(0.0, min(eval_ratio, 1.0)))
    return items[:eval_count], items[eval_count:]


def _materialize(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(source, target)
        return
    target.symlink_to(source.resolve())


def _write_manifest(path: Path, rows: list[Row]) -> None:
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

    print("GenImage summary")
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
