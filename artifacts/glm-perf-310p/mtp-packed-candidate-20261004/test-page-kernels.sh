#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# First device smoke for the selective packed-W3 checkpoint, on a free TP4 host.
set -eo pipefail

source_root=${1:?pass the staged vllm-ascend source root}
checkpoint=${2:?pass the relinked selective-W3 checkpoint directory}
max_model_len=${3:-32768}
kv_fraction=${4:-0.70}
opp_root=${5:-$source_root/opp-w3-native}
max_num_seqs=${6:-4}
execution_mode=${7:-graph}
max_batched_tokens=${8:-640}
prefill_route_count_mode=${9:-compare}
profile_root=${10:-}
qsa_opp_root=${11:-/srv/ai/src/glm-qsa-padded-20260930/opp}
prefix_cache_mode=${12:-off}
sse_heartbeat_mode=${13:-off}
fused_route_combine_mode=${14:-off}
empty_peer_rows_mode=${15:-off}
grouped_max_routes=${16:-}
mtp_tokens=${17:-1}
case "$mtp_tokens" in
  0) speculative_args=() ;;
  1|2) speculative_args=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$mtp_tokens,\"draft_load_config\":{\"load_format\":\"glm_w2_filtered\"}}") ;;
  *) echo "MTP candidate supports 0, 1, or 2 draft tokens" >&2; exit 1 ;;
esac

if [[ ! "$max_batched_tokens" =~ ^[1-9][0-9]*$ ]] || (( max_batched_tokens < 64 || max_batched_tokens > 1280 || max_batched_tokens % 64 )); then
  echo "max_batched_tokens must be a multiple of 64 from 64 through 1280" >&2
  exit 1
fi

case "$prefill_route_count_mode" in
  compare)
    hf_overrides='{"architectures":["Glm5NextW2ForCausalLM"],"ascend_glm_nz_packed_codes":true,"ascend_glm_mhc_batched_round":true,"ascend_glm_mhc_fp16_state":true,"ascend_glm_mhc_ai_core_round":true,"ascend_glm_kda_nz_grouped":true}'
    ;;
  histogram)
    hf_overrides='{"architectures":["Glm5NextW2ForCausalLM"],"ascend_glm_nz_packed_codes":true,"ascend_glm_mhc_batched_round":true,"ascend_glm_mhc_fp16_state":true,"ascend_glm_mhc_ai_core_round":true,"ascend_glm_kda_nz_grouped":true,"ascend_glm_prefill_route_histogram":true}'
    ;;
  *)
    echo "prefill route count mode must be compare or histogram: $prefill_route_count_mode" >&2
    exit 1
    ;;
esac

if [[ -n "$grouped_max_routes" ]]; then
  if [[ ! "$grouped_max_routes" =~ ^[1-9][0-9]*$ ]] || (( grouped_max_routes > 32768 )); then
    echo "grouped_max_routes must be an integer from 1 through 32768" >&2
    exit 1
  fi
  # The selected experimental OPP must support the same route capacity.
  hf_overrides="${hf_overrides%?},\"ascend_glm_grouped_max_routes\":$grouped_max_routes}"
fi

case "$execution_mode" in
  eager)
    execution_args=(--enforce-eager)
    ;;
  graph)
    if [[ "$max_num_seqs" != 4 || "$mtp_tokens" != 1 ]]; then
      echo "graph candidate requires MTP1 and max_num_seqs=4 for captures 2 and 8" >&2
      exit 1
    fi
    hf_overrides="${hf_overrides%?},\"ascend_glm_mtp_full_graph\":true}"
    execution_args=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[2,8]}')
    ;;
  *)
    echo "execution mode must be eager or graph: $execution_mode" >&2
    exit 1
    ;;
esac

case "$prefix_cache_mode" in
  off)
    prefix_cache_args=(--no-enable-prefix-caching)
    ;;
  on)
    prefix_cache_args=(--enable-prefix-caching)
    ;;
  *)
    echo "prefix cache mode must be off or on: $prefix_cache_mode" >&2
    exit 1
    ;;
esac

case "$sse_heartbeat_mode" in
  off)
    heartbeat_args=()
    ;;
  on)
    heartbeat_args=(--middleware vllm_ascend._310p.sse_heartbeat.SSEHeartbeatMiddleware)
    ;;
  *)
    echo "SSE heartbeat mode must be off or on: $sse_heartbeat_mode" >&2
    exit 1
    ;;
esac

case "$fused_route_combine_mode" in
  off) ;;
  on)
    hf_overrides="${hf_overrides%?},\"ascend_glm_fused_route_combine\":true}"
    ;;
  *)
    echo "fused route combine mode must be off or on: $fused_route_combine_mode" >&2
    exit 1
    ;;
esac

case "$empty_peer_rows_mode" in
  off) ;;
  on)
    if [[ "$fused_route_combine_mode" != on ]]; then
      echo "empty peer rows requires fused route combine" >&2
      exit 1
    fi
    hf_overrides="${hf_overrides%?},\"ascend_glm_empty_peer_rows\":true}"
    ;;
  *)
    echo "empty peer rows mode must be off or on: $empty_peer_rows_mode" >&2
    exit 1
    ;;
esac

source /srv/ai/bin/ascend-env.sh
set -u
if [[ "$execution_mode" == graph ]]; then
  export VLLM_USE_BREAKABLE_CUDAGRAPH=1
fi

round_vendor=/srv/ai/src/glm-bf16-round-nan-20261003/opp-nan/vendors/custom_transformer
w3_vendor="$opp_root/vendors/custom_transformer"
if [[ ! -d "$w3_vendor/op_api/lib" ]]; then
  echo "native W3 OPP package is missing: $w3_vendor" >&2
  exit 1
fi
split_vendor=/srv/ai/src/kda-persistent-scores-opp/vendors/custom_transformer
grouped_vendor=/srv/ai/src/glm-w2-ktile256-20261003/opp-ktile256/vendors/custom_transformer
sinkhorn_vendor=/srv/ai/src/glm-sinkhorn-20260930/opp-sinkhorn/vendors/custom_transformer
mla_write_vendor=/srv/ai/src/glm-sinkhorn-20260930/opp-mla-write/vendors/custom_transformer
qsa_vendor="$qsa_opp_root/vendors/custom_transformer"
if [[ ! -d "$qsa_vendor/op_api/lib" ]]; then
  echo "QSA OPP package is missing: $qsa_vendor" >&2
  exit 1
fi
mla_vendor=/srv/ai/src/glm-mla-opp/vendors/custom_transformer
base_vendor=/srv/ai/src/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer

state_vendor=/srv/ai/src/glm-l1-wide-build-20261004/opp-mtp-pages-20261005/packages/vendors/custom_transformer
if [[ ! -d "$state_vendor/op_api/lib" ]]; then
  echo "MTP recurrent-state OPP package is missing: $state_vendor" >&2
  exit 1
fi

export PYTHONPATH="$source_root:${PYTHONPATH:-}"
export ASCEND_CUSTOM_OPP_PATH="$state_vendor:$w3_vendor:$round_vendor:$qsa_vendor:$sinkhorn_vendor:$mla_write_vendor:$split_vendor:$grouped_vendor:$mla_vendor:$base_vendor"
export LD_LIBRARY_PATH="$state_vendor/op_api/lib:$w3_vendor/op_api/lib:$round_vendor/op_api/lib:$qsa_vendor/op_api/lib:$sinkhorn_vendor/op_api/lib:$mla_write_vendor/op_api/lib:$split_vendor/op_api/lib:$grouped_vendor/op_api/lib:$mla_vendor/op_api/lib:$base_vendor/op_api/lib:${LD_LIBRARY_PATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export SOC_VERSION=ascend310p1
export TASK_QUEUE_ENABLE=1
export OMP_NUM_THREADS=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export VLLM_ASCEND_310P_ENABLE_MLA=1
export VLLM_ASCEND_310P_GLM_HOST_KV=0
export VLLM_ASCEND_KV_CACHE_FRACTION="$kv_fraction"
export VLLM_ASCEND_LOG_REQUEST_TIMINGS=1

profiler_args=()
if [[ -n "$profile_root" ]]; then
  mkdir -p "$profile_root"
  profiler_config=$(printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":false,"torch_profiler_with_memory":false,"ignore_frontend":true,"delay_iterations":0,"max_iterations":2}' "$profile_root")
  profiler_args=(--profiler-config "$profiler_config")
fi

cd "$source_root"
exec /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python -m pytest -q -s /home/matteius/experiments/glm-w3-20261004/mtp-pages-09/test_glm_mtp_state_graph_310.py -k 'convolution or independent_cpu_recurrence or native_state_table'
