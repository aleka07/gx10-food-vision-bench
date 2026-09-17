"""One-shot driver for the whole measurement: data -> features -> probe -> bench.

The stages write every intermediate artefact under ../data and ../results, so a
failed or interrupted run can be resumed with `--stage`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import BACKBONES, DATA, RESULTS, host_info, json_dump, set_seed
from data_food101 import load_split, materialise
from features import extract_features
from probe import sweep


def stage_data(per_class: int, shards: int, eval_per_class: int = 25, eval_shards: int = 1) -> dict:
    manifest = materialise(per_class, eval_per_class, shards, eval_shards)
    print(
        f"dataset: {manifest['train_total']} train images "
        f"({per_class}/class cap), {manifest['test_total']} evaluation images"
    )
    return manifest


def stage_features(backbones: list[str], per_class: int, precision: str) -> dict:
    stats = {}
    for backbone in backbones:
        out = RESULTS / f"features_{backbone}.npz"
        if out.exists():
            print(f"features for {backbone} already present, skipping")
            continue
        train_paths, train_y, _ = load_split("train")
        test_paths, test_y, _ = load_split("test")
        train_x, train_y, s_train = extract_features(
            backbone, train_paths, train_y, device="cuda", precision=precision
        )
        test_x, test_y, s_test = extract_features(
            backbone, test_paths, test_y, device="cuda", precision=precision
        )
        np.savez_compressed(
            out, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y
        )
        stats[backbone] = {"train": s_train, "test": s_test, "file": str(out)}
        print(f"{backbone}: {s_train['images_per_second']} img/s feature extraction")
    json_dump(stats, RESULTS / "feature_stats.json")
    return stats


def stage_probe(backbones: list[str], per_class: list[int]) -> dict:
    out_dir = RESULTS / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for backbone in backbones:
        features = RESULTS / f"features_{backbone}.npz"
        if not features.exists():
            raise SystemExit(f"missing {features}; run --stage features first")
        result = sweep(features, per_class)
        json_dump(result, out_dir / f"{backbone}_probe.json")
        all_results[backbone] = result
        print(
            backbone,
            [(p["per_class"], p["top1"]) for p in result["points"]],
            flush=True,
        )
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        nargs="+",
        default=["data", "features", "probe"],
        choices=["data", "features", "probe", "bench", "figures", "all"],
    )
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--eval-per-class", type=int, default=25)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--eval-shards", type=int, default=3)
    parser.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    parser.add_argument(
        "--probe-per-class",
        type=int,
        nargs="+",
        default=[5, 10, 25, 50],
        help="training-set sizes per class swept by the linear probe",
    )
    parser.add_argument("--fp16", default="fp16")
    args = parser.parse_args()

    stages = args.stage
    if "all" in stages:
        stages = ["data", "features", "probe", "bench", "figures"]

    set_seed(42)
    json_dump(host_info(), RESULTS / "stand_info.json")

    if "data" in stages:
        stage_data(args.per_class, args.shards, args.eval_per_class, args.eval_shards)
    if "features" in stages:
        stage_features(args.backbones, args.per_class, args.fp16)
    if "probe" in stages:
        stage_probe(args.backbones, args.probe_per_class)
    if "bench" in stages:
        from bench import main as bench_main

        import sys

        sys.argv = ["bench.py"]
        bench_main()

    print("done")


if __name__ == "__main__":
    main()
