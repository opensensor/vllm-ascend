#!/usr/bin/env bash
set -eo pipefail
source /srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-hardware-env.sh
runtime=/srv/ai/src/qwen-six-chip-test-f25e70c09-20261009
output=$runtime/gdn-unified-v5-opp/vendors/qwen_gdn_unified_v5_transformer
stride=$runtime/gdn-stride-opp/vendors/qwen_gdn_stride_v1_transformer
coherent=/srv/ai/src/qwen38-prefill-batch256-coherent-opp-20261005/vendors/qwen38_batch256_coherent_transformer
retained=/srv/ai/src/native-int4-w4a8.KiuhBN/opp-retained-good-20260928
packaged=$runtime/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export PYTHONPATH="$runtime:/srv/ai/src/vllm-opensensor:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0
export ASCEND_CUSTOM_OPP_PATH="$output:$stride:$coherent:$packaged:$retained"
export LD_LIBRARY_PATH="$output/op_api/lib:$stride/op_api/lib:$coherent/op_api/lib:$packaged/op_api/lib:$retained/op_api/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
cd "$runtime"
exec taskset -c 32-39 /srv/ai/venvs/fork028/bin/python "$@"
