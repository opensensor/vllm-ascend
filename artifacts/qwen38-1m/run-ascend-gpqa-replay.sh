#!/usr/bin/env bash

set -euo pipefail

# Exact quality-evaluation replay of the September 23, 2026 llama.cpp run.
# Run this from the RTX workstation, where the pinned AISBench checkout and
# GPQA dataset are already installed. The model server may run on another host.

AIS_BENCH_BIN="${AIS_BENCH_BIN:-ais_bench}"
SERVER_HOST="${SERVER_HOST:-}"
SERVER_PORT="${SERVER_PORT:-8001}"
MODEL_NAME="${MODEL_NAME:-qwen38-flash-next-w8a8}"
WORK_ROOT="${WORK_ROOT:-./benchmark-results/qwen38-quality}"
RUN_NAME="${RUN_NAME:-gpqa-diamond-deterministic-ascend-w8a8}"
NUM_PROMPTS="${NUM_PROMPTS:-}"

GENERATION_KWARGS='{"ignore_eos":false,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0,"seed":1024,"temperature":0.0,"top_k":20,"top_p":1.0}'

if [[ -z "${SERVER_HOST}" ]]; then
  echo "Set SERVER_HOST to the Ascend server address." >&2
  exit 1
fi

if ! command -v "${AIS_BENCH_BIN}" >/dev/null 2>&1 && [[ ! -x "${AIS_BENCH_BIN}" ]]; then
  echo "AISBench executable not found: ${AIS_BENCH_BIN}" >&2
  exit 1
fi

PROMPT_LIMIT_ARGS=()
if [[ -n "${NUM_PROMPTS}" ]]; then
  if [[ ! "${NUM_PROMPTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_PROMPTS must be a positive integer: ${NUM_PROMPTS}" >&2
    exit 1
  fi
  PROMPT_LIMIT_ARGS=(--num-prompts "${NUM_PROMPTS}")
fi

echo "Checking OpenAI-compatible endpoint at ${SERVER_HOST}:${SERVER_PORT}..."
curl --fail --silent --show-error \
  "http://${SERVER_HOST}:${SERVER_PORT}/v1/models" >/dev/null

echo "Starting exact GPQA Diamond replay against ${MODEL_NAME}."
if [[ -n "${NUM_PROMPTS}" ]]; then
  echo "Limiting this run to the first ${NUM_PROMPTS} dataset prompts."
fi
echo "Results will be written below ${WORK_ROOT}/${RUN_NAME}."

exec "${AIS_BENCH_BIN}" \
  --models vllm_api_general_chat \
  --datasets gpqa_gen_0_shot_cot_chat_prompt \
  --mode all \
  --work-dir "${WORK_ROOT}/${RUN_NAME}" \
  --host-ip "${SERVER_HOST}" \
  --host-port "${SERVER_PORT}" \
  --model-name "${MODEL_NAME}" \
  --batch-size 3 \
  --max-out-len 8192 \
  --request-rate 0 \
  --retry 2 \
  --num-warmups 1 \
  "${PROMPT_LIMIT_ARGS[@]}" \
  --max-num-workers 1 \
  --max-workers-per-gpu 1 \
  --dump-eval-details \
  --dump-extract-rate \
  --no-trust-remote-code \
  --generation-kwargs "${GENERATION_KWARGS}"
