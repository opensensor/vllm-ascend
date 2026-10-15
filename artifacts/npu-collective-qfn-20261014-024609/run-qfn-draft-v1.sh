#!/usr/bin/env bash
set -eo pipefail
source /srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-hardware-env.sh
experiment_root=/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609
runtime_root=${experiment_root}/qfn-draft-partition-runtime-v1
coherent_opp=/srv/ai/src/qwen38-coherent-opp-20261001-r2/vendors/qwen38_coherent_transformer
packaged_opp=${runtime_root}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
retained_opp=/srv/ai/src/native-int4-w4a8.KiuhBN/opp-retained-good-20260928
export PYTHONPATH="${experiment_root}/optimization-wave-source-v1:${runtime_root}:/srv/ai/src/vllm-opensensor:${PYTHONPATH:-}"
export ASCEND_CUSTOM_OPP_PATH="${coherent_opp}:${packaged_opp}:${retained_opp}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export LD_LIBRARY_PATH="${coherent_opp}/op_api/lib:${packaged_opp}/op_api/lib:${retained_opp}/op_api/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_ASCEND_KV_CACHE_FRACTION=0.80
export VLLM_ASCEND_LOG_REQUEST_TIMINGS=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export TASK_QUEUE_ENABLE=1
export OMP_NUM_THREADS=1
cd "${experiment_root}/optimization-wave-source-v1"
exec /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python -u -m tools.sidecard.qfn_collective_profile "$@"
