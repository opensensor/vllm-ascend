#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Faster image-enabled serving profile. Port 8001, three slots, MTP2,
# graphs [3,9], 2560-token scheduler batch and bounded checkpoints.
# Deferred launch only: 94C thermal hold, resume at 85C, emergency cutoff 96C.
# Apply native HC through the existing select-hc.py after loading real weights.
set -eo pipefail

# This snapshot is the full source set used by the retained coherency and GPQA
# gates. Override it only with another runtime that passes --check-runtime and
# an end-to-end thinking-enabled coherency probe.
RUNTIME_ROOT=${QWEN38_PLUGIN_ROOT:-/srv/ai/src/qwen38-prefix-bounded-runtime-20261008}
MODEL=${QWEN38_MODEL_ROOT:-/srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i}
PYTHON_BIN=${QWEN38_PYTHON_BIN:-/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python}
HARDWARE_ENV=${QWEN38_HARDWARE_ENV:-/srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-hardware-env.sh}
COHERENT_OPP=${QWEN38_COHERENT_OPP:-/srv/ai/src/qwen38-prefill-batch256-coherent-opp-20261005/vendors/qwen38_batch256_coherent_transformer}
RETAINED_OPP=${QWEN38_RETAINED_OPP:-/srv/ai/src/native-int4-w4a8.KiuhBN/opp-retained-good-20260928}
PACKAGED_OPP=${RUNTIME_ROOT}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
AFFINITY_HELPER=${QWEN38_AFFINITY_HELPER:-/srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-kilo-affinity-r8.py}

# Profile KV allocation on startup; preserve the qualified compact state pool.
PORT=${PORT:-8001}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen38-prefix-bounded-validation}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-3}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2560}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.965}
KV_CACHE_FRACTION=${KV_CACHE_FRACTION:-0.70}
TEMP_LIMIT=${TEMP_LIMIT:-96}

LOG=${LOG:-$HOME/logs/serve_qwen38_pass4_unified_${PORT}.log}
WATCHDOG_LOG=${WATCHDOG_LOG:-$HOME/logs/watchdog_qwen38_native_int4_mtp_graph.log}
AFFINITY_LOG=${AFFINITY_LOG:-$HOME/logs/affinity_qwen38_native_int4_mtp_graph.log}

usage() {
  echo "Usage: PORT=8001 $0 [--show|--check-runtime|--capture-routes]" >&2
}

if (( $# > 1 )) || { (( $# == 1 )) && [[ "$1" != --show && "$1" != --check-runtime && "$1" != --capture-routes ]]; }; then
  usage
  exit 2
fi

check_runtime_coherence() {
  "$PYTHON_BIN" - "$RUNTIME_ROOT" <<'PY'
import ast
import sys
from pathlib import Path

runtime_root = Path(sys.argv[1])
model_path = runtime_root / "vllm_ascend/models/qwen4_exp/model.py"
mtp_path = runtime_root / "vllm_ascend/models/qwen4_exp/mtp.py"
w4_moe_path = runtime_root / "vllm_ascend/models/qwen4_exp/w4_moe.py"
w4a8_path = runtime_root / "vllm_ascend/models/qwen4_exp/w4a8_int4.py"
for path in (model_path, mtp_path, w4_moe_path, w4a8_path):
    if not path.is_file():
        raise SystemExit(f"Qwen runtime coherence check: missing {path}")

model_tree = ast.parse(model_path.read_text(), filename=str(model_path))
formatter = next(
    (
        node
        for node in ast.walk(model_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_format_eager_linear_weights_npu"
    ),
    None,
)
formatter_args = [] if formatter is None else [arg.arg for arg in formatter.args.args]
if "extra_projection_types" not in formatter_args:
    raise SystemExit(
        "Qwen runtime coherence check: "
        f"{model_path} has a stale _format_eager_linear_weights_npu signature; "
        f"it is incompatible with {mtp_path}"
    )

mtp_tree = ast.parse(mtp_path.read_text(), filename=str(mtp_path))
uses_extended_formatter = any(
    isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "_format_eager_linear_weights_npu"
    and len(node.args) >= 2
    for node in ast.walk(mtp_tree)
)
if not uses_extended_formatter:
    raise SystemExit(
        "Qwen runtime coherence check: "
        f"{mtp_path} does not use the qualified eager projection formatter"
    )

w4_moe_tree = ast.parse(w4_moe_path.read_text(), filename=str(w4_moe_path))
required_w4_names = {
    alias.name
    for node in ast.walk(model_tree)
    if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == "w4_moe"
    for alias in node.names
}
available_w4_names = {
    node.name
    for node in w4_moe_tree.body
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
}
for node in w4_moe_tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        available_w4_names.update(alias.asname or alias.name for alias in node.names)
    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        available_w4_names.update(
            target.id for target in targets if isinstance(target, ast.Name)
        )
missing_w4_names = sorted(required_w4_names - available_w4_names)
if missing_w4_names:
    raise SystemExit(
        "Qwen runtime coherence check: "
        f"{model_path} imports {missing_w4_names} missing from {w4_moe_path}"
    )

w4a8_tree = ast.parse(w4a8_path.read_text(), filename=str(w4a8_path))
pack_function = next(
    (
        node
        for node in w4a8_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "pack_native_weight"
    ),
    None,
)
uses_parallel_pack = pack_function is not None and any(
    isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and isinstance(node.func.value, ast.Name)
    and node.func.value.id == "torch"
    and node.func.attr == "set_num_threads"
    for node in ast.walk(pack_function)
)
if not uses_parallel_pack:
    raise SystemExit(
        "Qwen runtime coherence check: "
        f"{w4a8_path} lost the qualified parallel native-weight packer"
    )

print(f"Qwen runtime coherence check passed: {runtime_root}")
PY
}

check_import_provenance() {
  "$PYTHON_BIN" - "$RUNTIME_ROOT" <<'PY'
import importlib.util
import sys
from pathlib import Path

runtime_root = Path(sys.argv[1]).resolve()
spec = importlib.util.find_spec("vllm_ascend")
if spec is None or spec.origin is None:
    raise SystemExit("Qwen runtime provenance check: cannot resolve vllm_ascend")
origin = Path(spec.origin).resolve()
try:
    origin.relative_to(runtime_root)
except ValueError:
    raise SystemExit(
        "Qwen runtime provenance check: "
        f"selected {origin}, expected a module under {runtime_root}"
    ) from None
print(f"Qwen runtime provenance check passed: {origin}")
PY
}
if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "PORT must be an integer from 1 to 65535" >&2
  exit 2
fi
if [[ ! "$NUM_SPEC_TOKENS" =~ ^[1-3]$ ]]; then
  echo "NUM_SPEC_TOKENS must be from 1 to 3" >&2
  exit 2
fi
if [[ ! "$MAX_NUM_SEQS" =~ ^[1-4]$ ]]; then
  echo "MAX_NUM_SEQS must be from 1 to 4" >&2
  exit 2
fi
for numeric_value in "$MAX_MODEL_LEN" "$MAX_NUM_BATCHED_TOKENS" "$TEMP_LIMIT"; do
  if [[ ! "$numeric_value" =~ ^[0-9]+$ ]]; then
    echo "Context, batching, and temperature settings must be integers" >&2
    exit 2
  fi
done

hf_overrides=$(cat <<'JSON'
{
  "text_config": {
    "ascend_expert_quantization": {
      "backend": "cube_310_int4_a8",
      "activation_quantization": "int8_per_group",
      "grouped_activation": "cann_builtin_fp16",
      "grouped_finalize": "cann_v2",
      "grouped_prefill_chunk_tokens": 2560,
      "bits": 4,
      "format": "qwen4exp_w4a16_group_v1",
      "group_size": 128,
      "lm_head_execution": "w8a8_dynamic",
      "ple_projection_execution": "w8a8_dynamic",
      "offset_dtype": "int8",
      "packing": "signed_int4_low_nibble_first_in_axis",
      "scale_dtype": "float16",
      "shared_expert_execution": "tp_sharded",
      "symmetric": false
    }
  }
}
JSON
)

speculative_config=$(printf '{"method":"mtp","num_speculative_tokens":%d}' "$NUM_SPEC_TOKENS")
decode_query_len=$((NUM_SPEC_TOKENS + 1))
# TP graph capture on 310P has a two-size event-id budget. Capture C1 and
# the maximum scheduled concurrency; intermediate batches pad to that size.
# With three slots and MTP2 this yields [3,9].
if (( MAX_NUM_SEQS == 1 )); then
  capture_sizes=$(printf '[%d]' "$decode_query_len")
else
  capture_sizes=$(printf '[%d,%d]' "$decode_query_len" "$((MAX_NUM_SEQS * decode_query_len))")
fi
compilation_config=$(printf \
  '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":%s}' \
  "$capture_sizes")

# Keep the default safetensors iterator: --enable-ep-weight-filter passes local
# expert IDs into it so each rank skips non-local expert tensors before disk I/O.
# The generic multi-thread iterator does not preserve that filter. The initial
# dense shards are slow on a cold page cache; expert shards skip rapidly after
# the checkpoint boundary (currently around shard 48).
serve_cmd=(
  "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 0.0.0.0 --port "$PORT"
  --worker-extension-cls tools.qwen4exp.prefix_npu_validation.QwenPrefixValidationExtension
  --middleware tools.glm_perf.resident_middleware.ResidentControlMiddleware
  --scheduler-cls vllm_ascend.core.prefix_mamba_scheduler.PrefixMambaBoundedScheduler
  --dtype float16
  --tensor-parallel-size 4
  --no-async-scheduling
  --disable-custom-all-reduce
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --enable-expert-parallel
  --enable-ep-weight-filter
  --enable-auto-tool-choice
  --tool-call-parser qwen3_xml
  --reasoning-parser qwen3
  --mamba-cache-mode align
  --enable-chunked-prefill
  --enable-prefix-caching
  --enable-prompt-tokens-details
  --speculative-config "$speculative_config"
  --compilation-config "$compilation_config"
  --hf-overrides "$hf_overrides"
  --limit-mm-per-prompt '{"image":1,"video":0}'
  --mm-processor-kwargs '{"min_pixels":65536,"max_pixels":1048576}'
)

# Diagnostic only: route capture adds device memory and host transfers.
if [[ "${1:-}" == --capture-routes ]]; then
  serve_cmd+=(--enable-return-routed-experts)
fi

if [[ "${1:-}" == --show ]]; then
  printf '%q ' "${serve_cmd[@]}"
  printf '\n'
  exit 0
fi

if [[ "${1:-}" == --check-runtime ]]; then
  check_runtime_coherence
  exit 0
fi

for required_dir in "$MODEL" "$RUNTIME_ROOT" "$COHERENT_OPP" "$RETAINED_OPP" "$PACKAGED_OPP"; do
  [[ -d "$required_dir" ]] || { echo "Required directory is missing: $required_dir" >&2; exit 1; }
done
[[ -x "$PYTHON_BIN" ]] || { echo "Python is missing or not executable: $PYTHON_BIN" >&2; exit 1; }
[[ -r "$HARDWARE_ENV" ]] || { echo "Hardware environment is missing: $HARDWARE_ENV" >&2; exit 1; }
[[ -r "$AFFINITY_HELPER" ]] || { echo "Affinity helper is missing: $AFFINITY_HELPER" >&2; exit 1; }

check_runtime_coherence

# Source the isolated 310P environment, then put the retained runtime and
# custom operators first. TASK_QUEUE_ENABLE=2 failed during graph capture.
# The environment path is intentionally configurable.
# shellcheck disable=SC1090
source "$HARDWARE_ENV"
export PYTHONPATH="${RUNTIME_ROOT}:/srv/ai/src/vllm-opensensor:${PYTHONPATH:-}"
# Keep each critical W4 and recurrent operator's host API and kernels in one
# package. Mixing the FP32 recurrent kernels with the retained FP16 host API
# made graph capture validate the state tensor against the wrong dtype. The
# r2 package also raises the W4 matmul and activation-pack tilers' grouped-
# prefill route capacity to 20,480 rows, which covers a 2,048-token chunk with
# eight routed experts. The older 5,120-row package passed decode capture but
# failed cold prefill. The
# embedded package must precede the retained fallback because its QSA host API
# matches the current logical_kv_heads ABI; the retained QSA API predates that
# argument. Plugin bootstrap prepends PACKAGED_OPP only when it is absent, so
# listing it explicitly also prevents it from shadowing COHERENT_OPP.
export ASCEND_CUSTOM_OPP_PATH="${COHERENT_OPP}:${PACKAGED_OPP}:${RETAINED_OPP}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export LD_LIBRARY_PATH="${COHERENT_OPP}/op_api/lib:${PACKAGED_OPP}/op_api/lib:${RETAINED_OPP}/op_api/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_ASCEND_KV_CACHE_FRACTION="$KV_CACHE_FRACTION"
export VLLM_ASCEND_LOG_REQUEST_TIMINGS=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export TASK_QUEUE_ENABLE=1
export OMP_NUM_THREADS=1

cd "$RUNTIME_ROOT"
check_import_provenance
[[ -r "$RUNTIME_ROOT/tools/qwen4exp/thermal_controller.py" ]] || { echo "Thermal controller is missing" >&2; exit 1; }
mkdir -p "$(dirname "$LOG")" "$(dirname "$WATCHDOG_LOG")"
touch "$LOG"

server_pid=$$
thermal_log="$(dirname "$WATCHDOG_LOG")/thermal-control-${PORT}.jsonl"
# Freeze active requests and queue new arrivals without resetting state.
# The controller exits with this API process and never launches a server.
"$PYTHON_BIN" -m tools.qwen4exp.thermal_controller \
  --base-url "http://127.0.0.1:${PORT}" \
  --api-pid "$server_pid" \
  --high-c 94 --low-c 85 --expected-devices 4 \
  --poll-s 1 >>"$thermal_log" 2>&1 &
(
  set +e
  while kill -0 "$server_pid" 2>/dev/null; do
    max_temp=$(timeout 10 npu-smi info 2>/dev/null | awk '/310P3/{for(i=1;i<=NF;i++) if($i=="NA") print $(i+1)}' | sort -n | tail -1)
    if [[ "$max_temp" =~ ^[0-9]+$ ]] && (( max_temp >= TEMP_LIMIT )); then
      printf '%s WATCHDOG: %sC >= %sC; SIGTERM server %s\n' "$(date)" "$max_temp" "$TEMP_LIMIT" "$server_pid"
      kill -TERM "$server_pid" 2>/dev/null
      break
    fi
    sleep 10
  done
) >>"$WATCHDOG_LOG" 2>&1 &

# The retained measurements isolate each TP worker tree on six physical CPU
# cores. Apply the same mapping after EngineCore creates all four workers;
# threads created later inherit their worker's mask.
(
  set +e
  for _ in $(seq 1 1800); do
    kill -0 "$server_pid" 2>/dev/null || exit 0
    engine_pid=$(pgrep -P "$server_pid" -f 'VLLM::EngineCore' | head -n 1)
    if [[ -n "$engine_pid" ]]; then
      worker_pids=()
      for rank in 0 1 2 3; do
        worker_pid=$(pgrep -P "$engine_pid" -f "VLLM::Worker_TP${rank}_EP${rank}" | head -n 1)
        [[ -n "$worker_pid" ]] || break
        worker_pids+=("$worker_pid")
      done
      if (( ${#worker_pids[@]} == 4 )); then
        "$PYTHON_BIN" "$AFFINITY_HELPER" \
          --api-pid "$server_pid" \
          --engine-pid "$engine_pid" \
          --workers "${worker_pids[@]}"
        exit $?
      fi
    fi
    sleep 1
  done
  echo "Timed out waiting for four TP workers; affinity was not applied" >&2
  exit 1
) >>"$AFFINITY_LOG" 2>&1 &

echo "Starting $SERVED_MODEL_NAME on port $PORT"
echo "Native INT4 W4A8; TP4; context $MAX_MODEL_LEN; MTP k=$NUM_SPEC_TOKENS; max sequences $MAX_NUM_SEQS"
echo "Server log: $LOG"
echo "Follow startup: tail -f $LOG"
echo "Thermal hold: 94C; resume at 85C; control log: $thermal_log"
echo "Thermal cutoff: ${TEMP_LIMIT}C (watchdog log: $WATCHDOG_LOG)"
echo "Worker affinity log: $AFFINITY_LOG"
exec "${serve_cmd[@]}" >>"$LOG" 2>&1
