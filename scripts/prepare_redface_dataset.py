"""RedFace 원본 폴더를 CAVE 평가/보정용 split으로 정리.

기본 입력:
  RedFace/
    Original/
    EFS/{train,valid,test}/
    FAM/{train,valid,test}/
    FR/{train,valid,test,videos}/
    FS/{train,valid,test}/

실행:
  python scripts/prepare_redface_dataset.py
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
METHODS = ("EFS", "FAM", "FR", "FS")
FAKE_SPLIT_MAP = {
    "calibration": "train",
    "eval": "valid",
    "holdout": "test",
}


@dataclass
class Row:
    split: str
    label: str
    method: str
    source_path: str
    target_path: str


def main() -> None:
    parser = argparse.ArgumentParser(description="RedFace -> CAVE split 정리")
    parser.add_argument("--source", default="RedFace", help="RedFace 원본 폴더")
    parser.add_argument("--output", default="test_data/redface", help="이미지 split 출력 폴더")
    parser.add_argument("--video-output", default="test_data/redface_video", help="FR 영상 출력 폴더")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-train-ratio", type=float, default=0.60)
    parser.add_argument("--real-eval-ratio", type=float, default=0.20)
    parser.add_argument("--max-real", type=int, default=0, help="0이면 전체 Original 사용")
    parser.add_argument("--include-videos", action="store_true", default=True)
    parser.add_argument("--no-videos", action="store_false", dest="include_videos")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    source = Path(args.source)
    output = Path(args.output)
    video_output = Path(args.video_output)

    if not source.exists():
        raise SystemExit(f"RedFace 폴더를 찾을 수 없습니다: {source}")

    if args.overwrite:
        _remove_if_exists(output)
        _remove_if_exists(video_output)

    rows = _prepare_image_splits(
        source=source,
        output=output,
        mode=args.mode,
        rng=rng,
        real_train_ratio=args.real_train_ratio,
        real_eval_ratio=args.real_eval_ratio,
        max_real=args.max_real,
    )
    _write_manifest(output / "manifest.csv", rows)
    _print_summary("images", rows)

    if args.include_videos:
        video_rows = _prepare_video_splits(
            source=source,
            output=video_output,
            mode=args.mode,
            rng=rng,
            real_train_ratio=args.real_train_ratio,
            real_eval_ratio=args.real_eval_ratio,
        )
        _write_manifest(video_output / "manifest.csv", video_rows)
        _print_summary("videos", video_rows)
        if video_rows:
            print("note: RedFace FR videos are fake/deepfake only; add separate real videos for binary video evaluation.")


def _prepare_image_splits(
    source: Path,
    output: Path,
    mode: str,
    rng: random.Random,
    real_train_ratio: float,
    real_eval_ratio: float,
    max_real: int,
) -> list[Row]:
    original_dir = source / "Original"
    real_images = _image_files(original_dir)
    rng.shuffle(real_images)
    if max_real > 0:
        real_images = real_images[:max_real]

    real_by_split = _split_items(real_images, real_train_ratio, real_eval_ratio)
    rows: list[Row] = []

    for split, images in real_by_split.items():
        for path in images:
            target = output / split / "real" / path.name
            _materialize(path, target, mode)
            rows.append(_row(split, "real", "Original", path, target))

    for split, fake_split in FAKE_SPLIT_MAP.items():
        target_count = len(real_by_split[split])
        fake_images = _sample_fake_images(source, fake_split, target_count, rng)
        for method, path in fake_images:
            target = output / split / "fake" / f"{method}_{path.name}"
            _materialize(path, target, mode)
            rows.append(_row(split, "fake", method, path, target))

    return rows


def _prepare_video_splits(
    source: Path,
    output: Path,
    mode: str,
    rng: random.Random,
    real_train_ratio: float,
    real_eval_ratio: float,
) -> list[Row]:
    videos = sorted((source / "FR" / "videos").glob("*.mp4"))
    rng.shuffle(videos)
    video_by_split = _split_items(videos, real_train_ratio, real_eval_ratio)
    rows: list[Row] = []

    for split, items in video_by_split.items():
        for path in items:
            target = output / split / "deepfake" / f"FR_{path.name}"
            _materialize(path, target, mode)
            rows.append(_row(split, "deepfake", "FR_video", path, target))
    return rows


def _sample_fake_images(
    source: Path,
    fake_split: str,
    target_count: int,
    rng: random.Random,
) -> list[tuple[str, Path]]:
    pools: dict[str, list[Path]] = {}
    for method in METHODS:
        files = _image_files(source / method / fake_split)
        rng.shuffle(files)
        pools[method] = files

    selected: list[tuple[str, Path]] = []
    per_method = target_count // len(METHODS)
    remainder = target_count % len(METHODS)

    for index, method in enumerate(METHODS):
        quota = per_method + (1 if index < remainder else 0)
        take = min(quota, len(pools[method]))
        selected.extend((method, path) for path in pools[method][:take])

    if len(selected) < target_count:
        used = {path for _, path in selected}
        leftovers = [
            (method, path)
            for method, files in pools.items()
            for path in files
            if path not in used
        ]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: target_count - len(selected)])

    rng.shuffle(selected)
    return selected[:target_count]


def _image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _split_items(
    items: list[Path],
    train_ratio: float,
    eval_ratio: float,
) -> dict[str, list[Path]]:
    train_count = int(len(items) * train_ratio)
    eval_count = int(len(items) * eval_ratio)
    return {
        "calibration": items[:train_count],
        "eval": items[train_count: train_count + eval_count],
        "holdout": items[train_count + eval_count:],
    }


def _materialize(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(source, target)
        return
    target.symlink_to(source.resolve())


def _remove_if_exists(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def _row(split: str, label: str, method: str, source: Path, target: Path) -> Row:
    return Row(
        split=split,
        label=label,
        method=method,
        source_path=str(source),
        target_path=str(target),
    )


def _write_manifest(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["split", "label", "method", "source_path", "target_path"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _print_summary(name: str, rows: list[Row]) -> None:
    print(f"\n{name} summary")
    counts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = (row.split, row.label, row.method)
        counts[key] = counts.get(key, 0) + 1

    for split in ("calibration", "eval", "holdout"):
        split_total = sum(value for key, value in counts.items() if key[0] == split)
        print(f"  {split}: {split_total}")
        for label in sorted({key[1] for key in counts if key[0] == split}):
            label_total = sum(value for key, value in counts.items() if key[0] == split and key[1] == label)
            methods = [
                f"{method}:{counts[(split, label, method)]}"
                for method in sorted({key[2] for key in counts if key[0] == split and key[1] == label})
            ]
            print(f"    {label}: {label_total} ({', '.join(methods)})")


if __name__ == "__main__":
    main()
