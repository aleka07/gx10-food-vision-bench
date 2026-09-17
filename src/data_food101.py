"""Materialise a balanced Food-101 subset from the Hugging Face Hub.

Dataset: ``food101`` (Bossard, Guillaumin, Van Gool, ECCV 2014), CC BY 4.0,
101 dish classes, 101 000 real-world photographs.

Split mapping matters and is stated explicitly here: the Hub repository stores
the official 75 750-image training set as ``data/train-*`` and the official
25 250-image test set as ``data/validation-*``. We therefore materialise a
training subset from ``train`` shards and the evaluation set from
``validation`` shards, and label the evaluation directory ``test``.

Class names are read from the parquet schema metadata (the authoritative copy
shipped with the data), not from the dataset card.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from common import DATA, json_dump, set_seed

DATASET_ID = "food101"
REVISION = "main"
TRAIN_SHARD_TMPL = "data/train-{i:05d}-of-00008.parquet"
EVAL_SHARD_TMPL = "data/validation-{i:05d}-of-00003.parquet"


def _download(patterns: list[str]) -> list[Path]:
    from huggingface_hub import hf_hub_download

    return [
        Path(
            hf_hub_download(
                repo_id=DATASET_ID, repo_type="dataset", filename=p, revision=REVISION
            )
        )
        for p in patterns
    ]


def _class_names(parquet_path: Path) -> list[str]:
    import pyarrow.parquet as pq

    meta = pq.read_schema(parquet_path).metadata or {}
    payload = meta.get(b"huggingface")
    if not payload:
        raise RuntimeError("parquet schema carries no Hugging Face feature metadata")
    info = json.loads(payload.decode("utf-8"))["info"]["features"]["label"]
    names = list(info["names"])
    if len(names) != 101:
        raise RuntimeError(f"expected 101 class names, got {len(names)}")
    return names


def _save(image_field, target: Path) -> bool:
    payload = image_field.get("bytes") if isinstance(image_field, dict) else None
    if payload is None:
        return False
    try:
        from PIL import Image

        with Image.open(io.BytesIO(payload)) as im:
            im.convert("RGB").save(target, format="JPEG", quality=95)
    except Exception:
        return False
    return True


def _harvest(split: str, shard_paths: list[Path], class_names: list[str], cap: int) -> dict:
    """Copy up to `cap` images per class out of the given shards."""
    import pyarrow.parquet as pq

    counts = {c: 0 for c in class_names}
    root = DATA / "food101" / split
    for parquet_path in shard_paths:
        table = pq.read_table(parquet_path, columns=["image", "label"])
        images = table.column("image").to_pylist()
        labels = table.column("label").to_pylist()
        for image_field, label in zip(images, labels):
            cls = class_names[int(label)]
            if counts[cls] >= cap:
                continue
            target_dir = root / cls
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{counts[cls]:04d}.jpg"
            if target.exists() or _save(image_field, target):
                counts[cls] += 1
        if all(counts[c] >= cap for c in class_names):
            break
    return counts


def materialise(
    per_class: int, eval_per_class: int = 25, train_shards: int = 8, eval_shards: int = 3
) -> dict:
    set_seed(42)
    train_paths = _download([TRAIN_SHARD_TMPL.format(i=i) for i in range(train_shards)])
    eval_paths = _download([EVAL_SHARD_TMPL.format(i=i) for i in range(eval_shards)])
    class_names = _class_names(train_paths[0])

    train_counts = _harvest("train", train_paths, class_names, per_class)
    eval_counts = _harvest("test", eval_paths, class_names, eval_per_class)

    manifest = {
        "dataset": DATASET_ID,
        "revision": REVISION,
        "license": "CC BY 4.0",
        "source_url": f"https://huggingface.co/datasets/{DATASET_ID}",
        "split_mapping": {
            "train": "Hub data/train-* = official Food-101 training set (75 750 images)",
            "test": "Hub data/validation-* = official Food-101 test set (25 250 images)",
        },
        "train_shards_used": train_shards,
        "eval_shards_used": eval_shards,
        "per_class_train_cap": per_class,
        "per_class_eval_cap": eval_per_class,
        "classes": class_names,
        "train_counts": train_counts,
        "train_total": sum(train_counts.values()),
        "test_counts": eval_counts,
        "test_total": sum(eval_counts.values()),
        "classes_below_train_cap": sorted(c for c in class_names if train_counts[c] < per_class),
        "classes_below_eval_cap": sorted(c for c in class_names if eval_counts[c] < eval_per_class),
    }
    json_dump(manifest, DATA / "food101" / "manifest.json")
    return manifest


def load_split(split: str, root: Path | None = None) -> tuple[list[Path], list[int], list[str]]:
    """Return (image paths, integer labels, class names) for a materialised split."""
    root = root or (DATA / "food101")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    classes = manifest["classes"]
    paths, labels = [], []
    for index, cls in enumerate(classes):
        for image_path in sorted((root / split / cls).glob("*.jpg")):
            paths.append(image_path)
            labels.append(index)
    return paths, labels, classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=50, help="training images per class")
    parser.add_argument("--eval-per-class", type=int, default=25)
    parser.add_argument("--shards", type=int, default=8, help="train shards to scan (class-ordered)")
    parser.add_argument("--eval-shards", type=int, default=3)
    args = parser.parse_args()
    manifest = materialise(args.per_class, args.eval_per_class, args.shards, args.eval_shards)
    summary = {k: v for k, v in manifest.items() if k not in ("classes", "train_counts", "test_counts")}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
