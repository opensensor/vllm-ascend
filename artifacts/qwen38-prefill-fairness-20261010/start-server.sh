#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail
profile_root=/srv/ai/src/qwen-prefill-fairness-20261010
export QWEN38_PLUGIN_ROOT=/srv/ai/src/qwen-performance-paced-tp6-20261010
export GPU_MEM_UTIL=0.955
export KV_CACHE_FRACTION=0.95
export MAX_MODEL_LEN=262144
export MAX_NUM_SEQS=6
export QWEN38_AFFINITY_HELPER="$profile_root/affinity-mtp-tp6.py"
export LOG="$profile_root/server-8001.log"
export WATCHDOG_LOG="$profile_root/watchdog.log"
export AFFINITY_LOG="$profile_root/affinity.log"
exec bash "$profile_root/start-paced-tp6.sh" "$@"
