#!/usr/bin/env bash
# Activate the Ascend environment and Python virtualenv before invoking this script.
set -euo pipefail

model_path=${1:?Usage: serve.sh MODEL_PATH [PORT] [DEVICE]}
api_port=${2:-8003}
export ASCEND_RT_VISIBLE_DEVICES=${3:-0}

exec python -m vllm.entrypoints.cli.main serve "$model_path" \
    --served-model-name qwen3-vl-scene \
    --host 127.0.0.1 --port "$api_port" \
    --dtype float16 --tensor-parallel-size 1 \
    --max-model-len 8192 --max-num-batched-tokens 1024 --max-num-seqs 2 \
    --gpu-memory-utilization 0.65 \
    --limit-mm-per-prompt '{"image":8,"video":0}' \
    --mm-processor-kwargs '{"max_pixels":262144}' \
    --enforce-eager --no-async-scheduling --disable-custom-all-reduce
