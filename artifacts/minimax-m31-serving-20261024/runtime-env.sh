#!/usr/bin/env bash
# Preserve CANN's Python modules while removing the unrelated source overlay.
source /srv/ai/bin/ascend-env.sh
export PYTHONPATH="${PYTHONPATH#/srv/ai/src/vllm-ascend:}"
export PATH=/srv/ai/venvs/minimax-m31/bin:$PATH
export ASCEND_CUSTOM_OPP_PATH=/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/qfn-runtime/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
export VLLM_USE_V2_MODEL_RUNNER=0
