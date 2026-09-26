#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

usage() {
    echo "Usage: bash $0 --model /path/to/downloaded/model [--host ADDRESS] [--port PORT] [--max-model-len TOKENS] [--dry-run]"
    echo "Activate your Ascend Python environment and source the CANN/ATB set_env.sh files first."
    echo "Fixed serving profile: TP4, MTP k=1, decode graphs, one sequence, memory utilization .965."
}

model_path=""
listen_host="0.0.0.0"
listen_port=8001
context_tokens=160000
dry_run=0
while (($#)); do
    case "$1" in
        --model|--host|--port|--max-model-len)
            if (($# < 2)); then
                usage >&2
                exit 2
            fi
            case "$1" in
                --model) model_path="$2" ;;
                --host) listen_host="$2" ;;
                --port) listen_port="$2" ;;
                --max-model-len) context_tokens="$2" ;;
            esac
            shift 2
            ;;
        --dry-run) dry_run=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -f "$model_path/config.json" || ! -f "$model_path/model.safetensors.index.json" ]]; then
    echo "--model must be a downloaded checkpoint directory with config.json and model.safetensors.index.json." >&2
    echo "The PLE host tables require local checkpoint shards; an HF repository ID is not sufficient." >&2
    exit 2
fi
if [[ ! "$listen_port" =~ ^[1-9][0-9]{0,4}$ ]] || ((listen_port > 65535)); then
    echo "--port must be in 1..65535." >&2
    exit 2
fi
if [[ ! "$context_tokens" =~ ^[1-9][0-9]{0,5}$ ]] || ((context_tokens > 160000)); then
    echo "--max-model-len must be in 1..160000 for this single-sequence profile." >&2
    exit 2
fi

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export SOC_VERSION=ascend310p1
export VLLM_ASCEND_ENABLE_310P=1
export VLLM_ASCEND_KV_CACHE_FRACTION=0.88
export VLLM_ASCEND_LOG_REQUEST_TIMINGS=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export TASK_QUEUE_ENABLE=1
export OMP_NUM_THREADS=1
export PYTHONFAULTHANDLER=1

serve_command=(
    vllm serve "$model_path"
    --served-model-name qwen38-flash-next-w8a8
    --host "$listen_host" --port "$listen_port"
    --dtype float16 --quantization ascend --tensor-parallel-size 4
    --max-model-len "$context_tokens" --max-num-batched-tokens 2048 --max-num-seqs 1
    --gpu-memory-utilization 0.965 --disable-custom-all-reduce
    --enable-prefix-caching --enable-prompt-tokens-details --mamba-cache-mode align
    --enable-chunked-prefill --language-model-only
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
    --no-async-scheduling
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --limit-mm-per-prompt '{"image":0,"video":0}'
)
if ((dry_run)); then
    printf '%q ' "${serve_command[@]}"
    printf '\n'
    exit 0
fi
if ! command -v vllm >/dev/null; then
    echo "vllm is not on PATH; activate the environment containing the pinned core and rebuilt plugin." >&2
    exit 2
fi
# Keep this in the foreground: Ctrl-C and service-manager signals reach vLLM.
# No existing processes are killed and no service is restarted by this script.
exec "${serve_command[@]}"
