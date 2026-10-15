#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Candidate: preserve public identity while enabling authenticated local reload.
set -eo pipefail
experiment_root=/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609
runtime_root=${experiment_root}/qfn-pinned-position-runtime
tools_root=${experiment_root}/resident-source-v1
run_root=
show=0
while (( $# )); do
  case "$1" in
    --runtime) runtime_root=$2; shift 2 ;;
    --tools-root) tools_root=$2; shift 2 ;;
    --run-root) run_root=$2; shift 2 ;;
    --show) show=1; shift ;;
    *) echo "Usage: $0 --run-root NEW [--runtime PATH] [--tools-root PATH] [--show]" >&2; exit 2 ;;
  esac
done
[[ "$run_root" == /* ]] || { echo 'A new absolute --run-root is required' >&2; exit 2; }
python_bin=/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python
additional_config=$("$python_bin" - "$experiment_root/qfn-custom-unobserved-config-v2.json" "$run_root/owners" <<'PY'
import json
import sys
config = json.load(open(sys.argv[1]))
if config['execution_mode'] != 'graph' or config['graph_replay_tracking'] is not False:
    raise SystemExit('unobserved custom graph config required')
config['output_root'] = sys.argv[2]
print(json.dumps({'npu_collective': config}, separators=(',', ':')))
PY
)
hf_overrides='{"text_config":{"ascend_expert_quantization":{"backend":"cube_310_int4_a8","activation_quantization":"int8_per_group","bits":4,"format":"qwen4exp_w4a16_group_v1","group_size":128,"lm_head_execution":"w8a8_dynamic","ple_projection_execution":"w8a8_dynamic","offset_dtype":"int8","packing":"signed_int4_low_nibble_first_in_axis","scale_dtype":"float16","shared_expert_execution":"tp_sharded","symmetric":false}}}'
serve_cmd=(
  "$python_bin" -u -m tools.sidecard.qfn_resident_server
  --control-token-file "$run_root/control.token"
  --close-receipt "$run_root/collective-close.json"
  -- serve /srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i
  --served-model-name qwen38-flash-next --host 0.0.0.0 --port 8001
  --dtype float16 --tensor-parallel-size 4 --no-async-scheduling
  --disable-custom-all-reduce --max-model-len 262144
  --max-num-batched-tokens 2048 --max-num-seqs 4
  --gpu-memory-utilization 0.965 --kv-cache-memory 88673894400
  --language-model-only --enable-expert-parallel --enable-ep-weight-filter
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
  --mamba-cache-mode align --enable-chunked-prefill --enable-prefix-caching
  --enable-prompt-tokens-details --seed 0
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[3,6]}'
  --hf-overrides "$hf_overrides" --limit-mm-per-prompt '{"image":0,"video":0}'
  --worker-extension-cls tools.sidecard.qfn_resident_worker.QfnResidentWorkerExtension
  --additional-config "$additional_config"
)
if (( show )); then
  printf '%q ' "${serve_cmd[@]}"
  printf '\n'
  exit 0
fi
[[ ! -e "$run_root" ]] || { echo 'Run root must be new; owners are never reused' >&2; exit 1; }
[[ -f "$tools_root/tools/sidecard/qfn_resident_worker.py" ]] || { echo 'Resident worker overlay is missing' >&2; exit 1; }
[[ -f "$tools_root/tools/sidecard/qfn_resident_server.py" ]] || { echo 'Resident API overlay is missing' >&2; exit 1; }
umask 077
mkdir -m 700 "$run_root"
# Reuse the qualified cold environment and coherent package precedence.
# shellcheck disable=SC1091
source /srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-hardware-env.sh
coherent_opp=/srv/ai/src/qwen38-coherent-opp-20261001-r2/vendors/qwen38_coherent_transformer
packaged_opp=${runtime_root}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
retained_opp=/srv/ai/src/native-int4-w4a8.KiuhBN/opp-retained-good-20260928
export PYTHONPATH="${tools_root}:${runtime_root}:/srv/ai/src/vllm-opensensor:${PYTHONPATH:-}"
export ASCEND_CUSTOM_OPP_PATH="${coherent_opp}:${packaged_opp}:${retained_opp}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export LD_LIBRARY_PATH="${coherent_opp}/op_api/lib:${packaged_opp}/op_api/lib:${retained_opp}/op_api/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_ASCEND_KV_CACHE_FRACTION=0.80
export VLLM_ASCEND_LOG_REQUEST_TIMINGS=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export VLLM_SERVER_DEV_MODE=1
export TASK_QUEUE_ENABLE=1
export OMP_NUM_THREADS=1
cd "$tools_root"
exec "${serve_cmd[@]}" >>"$run_root/server.log" 2>&1
