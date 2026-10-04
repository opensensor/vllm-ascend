#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# First device smoke for the selective packed-W3 checkpoint, on a free TP4 host.
set -eo pipefail

source_root=${1:?pass the staged vllm-ascend source root}
checkpoint=${2:?pass the relinked selective-W3 checkpoint directory}
max_model_len=${3:-32768}
kv_fraction=${4:-0.70}
opp_root=${5:-$source_root/opp-w3-native}
max_num_seqs=${6:-1}
execution_mode=${7:-eager}
max_batched_tokens=${8:-640}
prefill_route_count_mode=${9:-compare}
profile_root=${10:-}
qsa_opp_root=${11:-/srv/ai/src/glm-qsa-padded-20260930/opp}

if [[ ! "$max_batched_tokens" =~ ^[1-9][0-9]*$ ]] || (( max_batched_tokens < 64 || max_batched_tokens > 1280 || max_batched_tokens % 64 )); then
  echo "max_batched_tokens must be a multiple of 64 from 64 through 1280" >&2
  exit 1
fi

case "$prefill_route_count_mode" in
  compare)
    hf_overrides='{"architectures":["Glm5NextW2ForCausalLM"],"ascend_glm_nz_packed_codes":true,"ascend_glm_mhc_batched_round":true,"ascend_glm_mhc_ai_core_round":true,"ascend_glm_kda_nz_grouped":true}'
    ;;
  histogram)
    hf_overrides='{"architectures":["Glm5NextW2ForCausalLM"],"ascend_glm_nz_packed_codes":true,"ascend_glm_mhc_batched_round":true,"ascend_glm_mhc_ai_core_round":true,"ascend_glm_kda_nz_grouped":true,"ascend_glm_prefill_route_histogram":true}'
    ;;
  *)
    echo "prefill route count mode must be compare or histogram: $prefill_route_count_mode" >&2
    exit 1
    ;;
esac

case "$execution_mode" in
  eager)
    execution_args=(--enforce-eager)
    ;;
  graph)
    if [[ ! "$max_num_seqs" =~ ^[0-9]+$ ]] || (( max_num_seqs < 4 )); then
      echo "graph mode requires max_num_seqs >= 4 for capture sizes 1 and 4" >&2
      exit 1
    fi
    execution_args=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,4]}')
    ;;
  *)
    echo "execution mode must be eager or graph: $execution_mode" >&2
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

export PYTHONPATH="$source_root:${PYTHONPATH:-}"
export ASCEND_CUSTOM_OPP_PATH="$w3_vendor:$round_vendor:$qsa_vendor:$sinkhorn_vendor:$mla_write_vendor:$split_vendor:$grouped_vendor:$mla_vendor:$base_vendor"
export LD_LIBRARY_PATH="$w3_vendor/op_api/lib:$round_vendor/op_api/lib:$qsa_vendor/op_api/lib:$sinkhorn_vendor/op_api/lib:$mla_write_vendor/op_api/lib:$split_vendor/op_api/lib:$grouped_vendor/op_api/lib:$mla_vendor/op_api/lib:$base_vendor/op_api/lib:${LD_LIBRARY_PATH:-}"
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
exec /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python -m vllm.entrypoints.cli.main serve "$checkpoint" \
  --served-model-name glm53-flash-selective-w3 \
  --host 0.0.0.0 --port 8001 \
  --dtype float16 --tensor-parallel-size 4 \
  --max-model-len "$max_model_len" --max-num-seqs "$max_num_seqs" --max-num-batched-tokens "$max_batched_tokens" \
  --enable-chunked-prefill --no-enable-prefix-caching \
  --gpu-memory-utilization 0.965 \
  --load-format glm_w2_filtered \
  "${execution_args[@]}" --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser poolside_v1 \
  --reasoning-parser glm47 \
  --hf-overrides "$hf_overrides" \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --enable-logging-iteration-details \
  --disable-custom-all-reduce \
  "${profiler_args[@]}"
