#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

export PORT=8001
export LOG=/home/matteius/logs/serve_qwen38_named_prefill_20261005_8001.log
export WATCHDOG_LOG=/home/matteius/logs/watchdog_qwen38_named_prefill_20261005.log
export AFFINITY_LOG=/home/matteius/logs/affinity_qwen38_named_prefill_20261005.log

exec /srv/ai/src/qwen38-prefill-batch256-runtime-20261005/examples/start_qwen38_flash_next_w4_310p_batch2560_profile_prefill.sh
