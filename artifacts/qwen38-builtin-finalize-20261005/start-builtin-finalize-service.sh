#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

export QWEN38_PLUGIN_ROOT=/srv/ai/src/qwen38-builtin-finalize-runtime-20261005
export QWEN38_COHERENT_OPP=/srv/ai/src/qwen38-prefill-swiglu-opp-20261004/vendors/qwen38_swiglu_prefill_transformer
export PORT=8001
export SERVED_MODEL_NAME=qwen38-w4-builtin-finalize-candidate
export LOG=/srv/ai/src/qwen38-prefill-swiglu-src-20261004/results/builtin-finalize-20261005/serve.log
export WATCHDOG_LOG=/srv/ai/src/qwen38-prefill-swiglu-src-20261004/results/builtin-finalize-20261005/watchdog.log
export AFFINITY_LOG=/srv/ai/src/qwen38-prefill-swiglu-src-20261004/results/builtin-finalize-20261005/affinity.log

exec bash "$QWEN38_PLUGIN_ROOT/examples/start_qwen38_flash_next_w4_310p.sh" "$@"
