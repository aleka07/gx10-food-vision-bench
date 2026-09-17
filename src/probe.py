"""Linear probe on frozen features (ridge regression, closed form).

Frozen-feature probing is the cheapest way to adapt a generic backbone to a new
domain, and it is what runs in practice when a production line adds a new
product class: extract features once, fit a linear head on the CPU in seconds.
We fit the probe with ridge regression on one-hot targets and read the class
from the arg-max of the scores, following the standard probing protocol.

The training-size sweep (`--per-class 5 10 25 50`) re-fits the same probe on
nested subsets of the same features, so the curve isolates training-set size and
nothing else.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from common import RESULTS, json_dump, json_load, set_seed


def _one_hot(labels: np.ndarray, classes: int) -> np.ndarray:
    out = np.zeros((labels.shape[0], classes), dtype=np.float32)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def fit_probe(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, ridge: float = 1e-3
) -> dict:
    """Closed-form ridge probe; returns top-1 accuracy on the test split."""
    classes = int(max(train_y.max(), 0)) + 1
    x = train_x.astype(np.float64)
    y = _one_hot(train_y, classes).astype(np.float64)
    # Bias column keeps the closed form exact without a separate intercept pass.
    x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    gram = x.T @ x + ridge * np.eye(x.shape[1], dtype=np.float64)
    weights = np.linalg.solve(gram, x.T @ y)
    test = np.concatenate([test_x.astype(np.float64), np.ones((test_x.shape[0], 1))], axis=1)
    scores = test @ weights
    pred = scores.argmax(axis=1)
    return {"classes": classes, "predictions": pred, "max_abs_weight": float(np.abs(weights).max())}


def top1(pred: np.ndarray, truth: np.ndarray) -> float:
    return float((pred == truth).mean())


def subset_per_class(y: np.ndarray, per_class: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    index = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        index.append(idx[:per_class])
    return np.concatenate(index)


def sweep(features_path, per_class_list: list[int], seed: int = 42) -> dict:
    payload = np.load(features_path)
    train_x, train_y = payload["train_x"], payload["train_y"]
    test_x, test_y = payload["test_x"], payload["test_y"]
    results = {"feature_file": str(features_path), "seed": seed, "points": []}
    for per_class in per_class_list:
        idx = subset_per_class(train_y, per_class, seed)
        set_seed(seed)
        fitted = fit_probe(train_x[idx], train_y[idx], test_x)
        results["points"].append(
            {
                "per_class": per_class,
                "train_images": int(idx.shape[0]),
                "top1": round(top1(fitted["predictions"], test_y), 4),
                "test_images": int(test_y.shape[0]),
            }
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-class", type=int, nargs="+", default=[5, 10, 25, 50])
    parser.add_argument("--seed", type=int, default=42, help="seed for the training-subset draw")
    args = parser.parse_args()
    result = sweep(args.features, args.per_class, args.seed)
    json_dump(result, args.out)
    print(json.dumps(result, indent=2, ensure_ascii=False))
