"""Download every Food-101 parquet shard with Xet disabled.

The Hub's Xet read-token endpoint returned 404 for this dataset from the stand,
so plain HTTPS LFS downloads are used instead. Shards are class-ordered, hence
all 8 training and all 3 validation shards are required to cover 101 classes.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download

NAMES = [f"data/train-{i:05d}-of-00008.parquet" for i in range(8)] + [
    f"data/validation-{i:05d}-of-00003.parquet" for i in range(3)
]


def fetch(name: str) -> str:
    path = hf_hub_download("food101", name, repo_type="dataset")
    print(f"ok {name}", flush=True)
    return path


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, NAMES))
    print("DOWNLOAD_DONE", flush=True)
