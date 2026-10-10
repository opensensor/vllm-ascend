#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail
profile_root=/srv/ai/src/qwen-prefill-fairness-20261010
export QWEN38_PLUGIN_ROOT=/srv/ai/src/qwen-quality-repair-tp6-v3-20261010
export GPU_MEM_UTIL=0.955
export KV_CACHE_FRACTION=0.95
export MAX_MODEL_LEN=262144
export MAX_NUM_SEQS=6
export QWEN38_AFFINITY_HELPER="$profile_root/affinity-mtp-tp6.py"
export LOG="/srv/ai/src/qwen-quality-mtp1-20261010/server-8001.log"
export WATCHDOG_LOG="/srv/ai/src/qwen-quality-mtp1-20261010/watchdog.log"
export AFFINITY_LOG="/srv/ai/src/qwen-quality-mtp1-20261010/affinity.log"
exec bash "/srv/ai/src/qwen-quality-mtp1-20261010/start-paced-tp6.sh" "$@"
