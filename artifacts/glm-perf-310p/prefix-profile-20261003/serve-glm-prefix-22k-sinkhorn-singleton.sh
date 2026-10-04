#!/usr/bin/env bash
set -eo pipefail

source /srv/ai/bin/ascend-env.sh
set -u

source_root=/srv/ai/src/glm-sinkhorn-singleton-20261003
checkpoint=/srv/ai/models/GLM-5.3-Flash-W4through32-noclip-310p
profile_root=/home/matteius/experiments/glm-gate-a-20261002/profile-prefix-22k-sinkhorn-singleton-20261003
split_vendor=/srv/ai/src/kda-persistent-scores-opp/vendors/custom_transformer
grouped_vendor=/srv/ai/src/glm-w2-rint-scale-pair-20261003/opp-combined/vendors/custom_transformer
sinkhorn_vendor=/srv/ai/src/glm-sinkhorn-20260930/opp-sinkhorn/vendors/custom_transformer
mla_write_vendor=/srv/ai/src/glm-sinkhorn-20260930/opp-mla-write/vendors/custom_transformer
qsa_vendor=/srv/ai/src/glm-qsa-padded-20260930/opp/vendors/custom_transformer
mla_vendor=/srv/ai/src/glm-mla-opp/vendors/custom_transformer
base_vendor=/srv/ai/src/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer

mkdir -p "$profile_root"
export PYTHONPATH="$source_root:${PYTHONPATH:-}"
export ASCEND_CUSTOM_OPP_PATH="$qsa_vendor:$sinkhorn_vendor:$mla_write_vendor:$split_vendor:$grouped_vendor:$mla_vendor:$base_vendor"
export LD_LIBRARY_PATH="$qsa_vendor/op_api/lib:$sinkhorn_vendor/op_api/lib:$mla_write_vendor/op_api/lib:$split_vendor/op_api/lib:$grouped_vendor/op_api/lib:$mla_vendor/op_api/lib:$base_vendor/op_api/lib:${LD_LIBRARY_PATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export SOC_VERSION=ascend310p1
export TASK_QUEUE_ENABLE=1
export OMP_NUM_THREADS=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export VLLM_ASCEND_310P_ENABLE_MLA=1
export VLLM_ASCEND_310P_GLM_HOST_KV=0
export VLLM_ASCEND_LOG_REQUEST_TIMINGS=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1

profiler_config=$(printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":false,"torch_profiler_with_memory":false,"ignore_frontend":true,"delay_iterations":0,"max_iterations":32}' "$profile_root")
cd "$source_root"
exec /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python -m vllm.entrypoints.cli.main serve "$checkpoint" \
  --served-model-name glm53-flash-ascend-graph \
  --host 0.0.0.0 --port 8001 \
  --dtype float16 --tensor-parallel-size 4 \
  --max-model-len 22528 --max-num-seqs 4 --max-num-batched-tokens 640 \
  --enable-chunked-prefill --enable-prefix-caching \
  --gpu-memory-utilization 0.965 \
  --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser poolside_v1 \
  --reasoning-parser glm47 \
  --hf-overrides '{"architectures":["Glm5NextW2ForCausalLM"],"ascend_glm_nz_packed_codes":true,"ascend_glm_mhc_batched_round":true,"ascend_glm_fused_sinkhorn":true}' \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --enable-logging-iteration-details \
  --disable-custom-all-reduce \
  --profiler-config "$profiler_config" \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,4]}'
