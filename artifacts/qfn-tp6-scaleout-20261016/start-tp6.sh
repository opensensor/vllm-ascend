#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Six-NPU QFN profile: bounded prefix state, native MTP2, and qualified collectives.
set -eo pipefail
experiment_root=/srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016
runtime_root=${experiment_root}/runtime
tools_root=${experiment_root}/runtime
run_root=
show=0
dma_reserve_args=()
while (( $# )); do
  case "$1" in
    --runtime) runtime_root=$2; shift 2 ;;
    --tools-root) tools_root=$2; shift 2 ;;
    --run-root) run_root=$2; shift 2 ;;
    --show) show=1; shift ;;
    --dma-reserve-device) dma_reserve_args+=(--dma-reserve-device "$2"); shift 2 ;;
    *) echo "Usage: $0 --run-root NEW [--runtime PATH] [--tools-root PATH] [--show] [--dma-reserve-device ID ...]" >&2; exit 2 ;;
  esac
done
[[ "$run_root" == /* ]] || { echo 'A new absolute --run-root is required' >&2; exit 2; }
python_bin=/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python
additional_config=$("$python_bin" - "$experiment_root/collective-serving.json" "$run_root/owners" <<'PY'
import json
import sys
config = json.load(open(sys.argv[1]))
if config['execution_mode'] != 'graph' or config['graph_replay_tracking'] is not False:
    raise SystemExit('unobserved custom graph config required')
config['output_root'] = sys.argv[2]
print(json.dumps({'npu_collective': config, 'qwen_prefill_pacing': {'target_step_ms': 200, 'min_tokens': 128, 'initial_tokens': 128, 'max_tokens': 640, 'alignment': 128, 'smoothing': 0.25, 'decode_only_steps': 8}}, separators=(',', ':')))
PY
)

serve_cmd=(
  "$python_bin" -u -m tools.sidecard.qfn_resident_server
  --engine-control --expected-ranks 6
  "${dma_reserve_args[@]}"
  --control-token-file "$run_root/control.token"
  --close-receipt "$run_root/collective-close.json"
  -- serve /srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-TP6-MTP-20261010
  --served-model-name qwen38-flash-next --host 0.0.0.0 --port 8001
  --dtype float16 --tensor-parallel-size 6 --no-async-scheduling
  --disable-custom-all-reduce --max-model-len 262144
  --max-num-batched-tokens 2048 --max-num-seqs 6
  --gpu-memory-utilization 0.955
  --scheduler-cls vllm_ascend.core.qwen_prefill_scheduler.QwenPrefillPacedScheduler
  --language-model-only --mm-encoder-tp-mode data --enable-expert-parallel --enable-ep-weight-filter
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
  --mamba-cache-mode align --enable-chunked-prefill --enable-prefix-caching
  --enable-prompt-tokens-details --seed 0
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[3,6,12,18]}'
  --limit-mm-per-prompt '{"image":0,"video":0}'
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
coherent_opp=/srv/ai/src/qwen38-prefill-batch256-coherent-opp-20261005/vendors/qwen38_batch256_coherent_transformer
packaged_opp=${runtime_root}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
retained_opp=/srv/ai/src/native-int4-w4a8.KiuhBN/opp-retained-good-20260928
stride_opp=${runtime_root}/gdn-unified-v5-opp/vendors/qwen_gdn_unified_v5_transformer
for required in "$coherent_opp" "$packaged_opp" "$retained_opp" "$stride_opp"; do
  [[ -d "$required" ]] || { echo "Required operator package missing: $required" >&2; exit 1; }
done
export PYTHONPATH="${tools_root}:${runtime_root}:/srv/ai/src/vllm-opensensor:${PYTHONPATH:-}"
stride_opp=${runtime_root}/gdn-unified-v5-opp/vendors/qwen_gdn_unified_v5_transformer
export ASCEND_CUSTOM_OPP_PATH="${stride_opp}:${coherent_opp}:${packaged_opp}:${retained_opp}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export LD_LIBRARY_PATH="${stride_opp}/op_api/lib:${coherent_opp}/op_api/lib:${packaged_opp}/op_api/lib:${retained_opp}/op_api/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5
# Restore the retained bounded DMA polling budget without a driver reload.
if [[ $(cat /sys/module/drv_pcie_host/parameters/dma_query_spin_us) != 5 ]]; then
  sudo -n sh -c 'printf 5 > /sys/module/drv_pcie_host/parameters/dma_query_spin_us'
fi
export VLLM_ASCEND_KV_CACHE_FRACTION=0.95
export VLLM_ASCEND_LOG_REQUEST_TIMINGS=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export VLLM_SERVER_DEV_MODE=1
export TASK_QUEUE_ENABLE=1
export OMP_NUM_THREADS=1
cd "$tools_root"
exec "${serve_cmd[@]}" >>"$run_root/server.log" 2>&1
