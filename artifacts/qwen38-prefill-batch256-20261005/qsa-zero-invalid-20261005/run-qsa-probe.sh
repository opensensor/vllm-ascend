#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail

variant=${1:?baseline or zero}
active_groups=${2:?active group count}
label=${3:?unique result label}
action=${4:-benchmark}
case "$variant" in
  baseline|zero) ;;
  *) echo "variant must be baseline or zero" >&2; exit 2 ;;
esac
case "$action" in
  benchmark|regression) ;;
  *) echo "action must be benchmark or regression" >&2; exit 2 ;;
esac
[[ "$active_groups" =~ ^[0-9]+$ ]] || { echo "active group count must be numeric" >&2; exit 2; }

runtime=/srv/ai/src/qwen38-prefill-batch256-runtime-20261005
candidate=/srv/ai/src/qsa-zero-invalid-20261005/opp-v4/vendors/qsa_zero_invalid_310p_transformer
coherent=/srv/ai/src/qwen38-prefill-batch256-coherent-opp-20261005/vendors/qwen38_batch256_coherent_transformer
packaged="$runtime/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
retained=/srv/ai/src/native-int4-w4a8.KiuhBN/opp-retained-good-20260928
result_dir="$runtime/results/qsa-visible-table-20261005"
source /srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-hardware-env.sh
set -u
export PYTHONPATH="$runtime:/srv/ai/src/vllm-opensensor:${PYTHONPATH:-}"
if [[ "$variant" == zero ]]; then
  export ASCEND_CUSTOM_OPP_PATH="$candidate:$coherent:$packaged:$retained"
  export LD_LIBRARY_PATH="$candidate/op_api/lib:$coherent/op_api/lib:$packaged/op_api/lib:$retained/op_api/lib:${LD_LIBRARY_PATH:-}"
else
  export ASCEND_CUSTOM_OPP_PATH="$coherent:$packaged:$retained"
  export LD_LIBRARY_PATH="$coherent/op_api/lib:$packaged/op_api/lib:$retained/op_api/lib:${LD_LIBRARY_PATH:-}"
fi
export ASCEND_RT_VISIBLE_DEVICES=0
export TASK_QUEUE_ENABLE=1
cd "$runtime"
if [[ "$action" == regression ]]; then
  timeout 180 /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python -m pytest --noconftest -q \
    results/qsa-visible-table-20261005/test_qsa_gather_value_nz_310.py \
    | tee "$result_dir/$label.log"
else
  timeout 180 /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python \
    results/qsa-visible-table-20261005/benchmark_qsa_table_slice_310.py \
    --visible-tokens 2560 --cache-blocks 16800 --active-groups "$active_groups" --repeats 24 \
    | tee "$result_dir/$label.jsonl"
fi
