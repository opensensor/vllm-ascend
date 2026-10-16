#!/usr/bin/env bash
set -eo pipefail
experiment=/srv/ai/storage/artifacts/qfn-single-card-refresh-20261016/collective-experiment
source "$experiment/bundle/environment.sh"
export PYTHONPATH="$experiment/tooling:$experiment/bundle/runtime:$experiment/bundle/upstream:$PYTHONPATH"
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
unset TORCH_DEVICE_BACKEND_AUTOLOAD
exec /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python "$@"
