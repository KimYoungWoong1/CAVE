"""FaceForensics++ C23 폴더를 CAVE 영상 calibration/eval/holdout split으로 정리."""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


FAKE_METHODS = ("Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures")
EXTRA_FAKE_METHODS = ("DeepFakeDetection",)


@dataclass
class Row:
    split: str
    label: str
    method: str
    source_path: str
    target_path: str


def main() -> None:
    parser = argparse.ArgumentParser(description="FaceForensics++ C23 -> CAVE split 정리")
    parser.add_argument("--source", default="FaceForensics++_C23")
    parser.add_argument("--output", default="test_data/ffpp_c23")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--eval-ratio", type=float, default=0.20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if not source.exists():
        raise SystemExit(f"FaceForensics++ 폴더를 찾을 수 없습니다: {source}")

    if args.overwrite:
        _remove_if_exists(output)

    rng = random.Random(args.seed)
    id_to_split = _make_original_id_split(source, rng, args.train_ratio, args.eval_ratio)
    rows = _materialize_real(source, output, args.mode, id_to_split)
    rows.extend(_materialize_numeric_fakes(source, output, args.mode, id_to_split))
    rows.extend(_materialize_extra_fakes(source, output, args.mode, rng, args.train_ratio, args.eval_ratio))

    _write_manifest(output / "manifest.csv", rows)
    _print_summary(rows)


def _make_original_id_split(
    source: Path,
    rng: random.Random,
    train_ratio: float,
    eval_ratio: float,
) -> dict[str, str]:
    originals = sorted((source / "original").glob("*.mp4"))
    ids = [path.stem for path in originals]
    rng.shuffle(ids)
    train_count = int(len(ids) * train_ratio)
    eval_count = int(len(ids) * eval_ratio)
    mapping = {}
    for video_id in ids[:train_count]:
        mapping[video_id] = "calibration"
    for video_id in ids[train_count: train_count + eval_count]:
        mapping[video_id] = "eval"
    for video_id in ids[train_count + eval_count:]:
        mapping[video_id] = "holdout"
    return mapping


def _materialize_real(source: Path, output: Path, mode: str, id_to_split: dict[str, str]) -> list[Row]:
    rows: list[Row] = []
    for path in sorted((source / "original").glob("*.mp4")):
        split = id_to_split.get(path.stem)
        if split is None:
            continue
        target = output / split / "real" / path.name
        _materialize(path, target, mode)
        rows.append(_row(split, "real", "original", path, target))
    return rows


def _materialize_numeric_fakes(source: Path, output: Path, mode: str, id_to_split: dict[str, str]) -> list[Row]:
    rows: list[Row] = []
    for method in FAKE_METHODS:
        for path in sorted((source / method).glob("*.mp4")):
            source_id = path.stem.split("_", 1)[0]
            split = id_to_split.get(source_id)
            if split is None:
                continue
            target = output / split / "deepfake" / f"{method}_{path.name}"
            _materialize(path, target, mode)
            rows.append(_row(split, "deepfake", method, path, target))
    return rows


def _materialize_extra_fakes(
    source: Path,
    output: Path,
    mode: str,
    rng: random.Random,
    train_ratio: float,
    eval_ratio: float,
) -> list[Row]:
    rows: list[Row] = []
    for method in EXTRA_FAKE_METHODS:
        files = sorted((source / method).glob("*.mp4"))
        rng.shuffle(files)
        split_map = _split_items(files, train_ratio, eval_ratio)
        for split, paths in split_map.items():
            for path in paths:
                target = output / split / "deepfake" / f"{method}_{path.name}"
                _materialize(path, target, mode)
                rows.append(_row(split, "deepfake", method, path, target))
    return rows


def _split_items(items: list[Path], train_ratio: float, eval_ratio: float) -> dict[str, list[Path]]:
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


def _print_summary(rows: list[Row]) -> None:
    counts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = (row.split, row.label, row.method)
        counts[key] = counts.get(key, 0) + 1

    print("FaceForensics++ C23 summary")
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
