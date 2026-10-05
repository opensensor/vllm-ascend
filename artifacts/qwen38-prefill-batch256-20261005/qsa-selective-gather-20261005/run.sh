#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail
source /srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-hardware-env.sh
set -u
runtime=/srv/ai/src/qwen38-prefill-batch256-runtime-20261005
experiment=/srv/ai/src/qsa-selective-gather-20261005
candidate="$experiment/opp/vendors/qsa_selective_gather_310p_transformer"
coherent=/srv/ai/src/qwen38-prefill-batch256-coherent-opp-20261005/vendors/qwen38_batch256_coherent_transformer
packaged="$runtime/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
retained=/srv/ai/src/native-int4-w4a8.KiuhBN/opp-retained-good-20260928
export ASCEND_CUSTOM_OPP_PATH="$candidate:$coherent:$packaged:$retained"
export LD_LIBRARY_PATH="$candidate/op_api/lib:$coherent/op_api/lib:$packaged/op_api/lib:$retained/op_api/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$runtime:/srv/ai/src/vllm-opensensor:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0
export TASK_QUEUE_ENABLE=1
cd "$experiment"
if [[ "${1:-benchmark}" == regression ]]; then
  timeout 180 /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python -m pytest --noconftest -q test_named_gather.py \
    | tee named-regression.log
else
  timeout 240 /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python benchmark_named_gather.py \
    --binding "$experiment/binding-build/qsa_selective_probe.so" --output "$experiment/result.json" \
    | tee benchmark.log
fi
