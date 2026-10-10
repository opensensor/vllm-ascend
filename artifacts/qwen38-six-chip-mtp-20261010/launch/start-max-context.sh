#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Restart the tested six-chip service with its maximum-context profile.
set -eo pipefail

profile_root=/srv/ai/src/qwen-max-context-20261010
export GPU_MEM_UTIL=0.955
export KV_CACHE_FRACTION=0.95
export MAX_MODEL_LEN=262144
export MAX_NUM_SEQS=6
export QWEN38_AFFINITY_HELPER="$profile_root/affinity-mtp-tp6.py"
export LOG="$profile_root/server-8001.log"
export WATCHDOG_LOG="$profile_root/watchdog.log"
export AFFINITY_LOG="$profile_root/affinity.log"

exec bash /srv/ai/src/qwen-performance-evidence-20261010/start-mtp-tp6-v2.sh "$@"
