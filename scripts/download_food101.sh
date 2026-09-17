#!/bin/bash
# Download every Food-101 parquet shard (train + validation) from the Hub.
# Shards are class-ordered, so all shards are needed to cover 101 classes.
set -x
cd "$HOME/paper-gx10-vision" || exit 1
.venv/bin/hf download food101 --repo-type dataset --include 'data/*' --max-workers 8
echo "DOWNLOAD_DONE rc=$?"
du -sh "$HOME/.cache/huggingface"
