#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -eo pipefail

variant=${1:?usage: run_gate.sh baseline|candidate bench|test|test_extra}
phase=${2:?usage: run_gate.sh baseline|candidate bench|test|test_extra}
gate_dir=/srv/ai/src/qwen38-route-lookup-gate-20261003
runtime=/srv/ai/src/qwen38-head-unified-runtime-20261001
python=/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python
packaged="$runtime/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
retained=/srv/ai/src/native-int4-w4a8.KiuhBN/opp-retained-good-20260928
case "$variant" in
  baseline) opp=/srv/ai/src/qwen38-coherent-opp-20261001-r2/vendors/qwen38_coherent_transformer ;;
  candidate) opp=/srv/ai/src/qwen38-route-lookup-opp-20261003/vendors/custom_transformer ;;
  *) echo "unknown variant: $variant" >&2; exit 2 ;;
esac

source /srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-hardware-env.sh
export PYTHONPATH="$runtime:/srv/ai/src/vllm-opensensor:${PYTHONPATH:-}"
export ASCEND_CUSTOM_OPP_PATH="$opp:$packaged:$retained${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$opp/op_api/lib:$packaged/op_api/lib:$retained/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ASCEND_RT_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
cd "$runtime"

case "$phase" in
  bench)
    output=${3:-$gate_dir/$variant.jsonl}
    exec "$python" "$gate_dir/sweep_route_lookup.py" \
      --variant "$variant" --output "$output" \
      --tokens 3 6 12 --iterations 12 --repeats 5 --graph-unroll 4
    ;;
  test)
    exec "$python" -m pytest --noconftest -q "$gate_dir/test_qwen_w4_native_schedule_310.py" \
      -k 'native_routed_topk10_large_expert_bank_graph_replay or native_routed_decode_matches_sorted_reference or native_routed_changing_input_and_ids_graph or native_routed_reuses_packed_token_across_experts or model_c1_gate_up_partitioned_schedule_graph_replay or model_c1_weight_pipeline_and_full_tile_fallback_graph_replay'
    ;;
  test_extra)
    exec "$python" -m pytest --noconftest -q \
      "$gate_dir/test_qwen_w4_native_schedule_310.py" \
      "$gate_dir/test_qwen_w4_down_reduce_310.py" \
      -k 'native_routed_reused_activation_graph_replay or pack_accepts_full_2k_prefill_route_capacity or down_reduce'
    ;;
  *) echo "unknown phase: $phase" >&2; exit 2 ;;
esac
