#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

export QWEN38_PLUGIN_ROOT=/srv/ai/src/qwen38-prefill-swiglu-runtime-20261004
export QWEN38_COHERENT_OPP=/srv/ai/src/qwen38-prefill-swiglu-opp-20261004/vendors/qwen38_swiglu_prefill_transformer
export PORT=8001
export SERVED_MODEL_NAME=qwen38-w4-builtin-swiglu-candidate
export LOG=/srv/ai/src/qwen38-prefill-swiglu-src-20261004/results/serve-builtin-v2-20261004.log
export WATCHDOG_LOG=/srv/ai/src/qwen38-prefill-swiglu-src-20261004/results/watchdog-builtin-v2-20261004.log
export AFFINITY_LOG=/srv/ai/src/qwen38-prefill-swiglu-src-20261004/results/affinity-builtin-v2-20261004.log

exec bash "$QWEN38_PLUGIN_ROOT/examples/start_qwen38_flash_next_w4_310p.sh"
