#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail
profile_root=/srv/ai/src/qwen-prefill-fairness-20261010
export QWEN38_PLUGIN_ROOT=/srv/ai/src/qwen-quality-repair-tp6-v10-20261010
export GPU_MEM_UTIL=0.955
export KV_CACHE_FRACTION=0.95
export MAX_MODEL_LEN=262144
export MAX_NUM_SEQS=6
export QWEN38_AFFINITY_HELPER="$profile_root/affinity-mtp-tp6.py"
export LOG="/srv/ai/src/qwen-quality-fast-correct-tp6-20261010/server-8001.log"
export WATCHDOG_LOG="/srv/ai/src/qwen-quality-fast-correct-tp6-20261010/watchdog.log"
export AFFINITY_LOG="/srv/ai/src/qwen-quality-fast-correct-tp6-20261010/affinity.log"
if (( $# == 0 )) || [[ "$1" == --capture-routes ]]; then
  export PYTHONPATH="$QWEN38_PLUGIN_ROOT:/srv/ai/src/vllm-opensensor${PYTHONPATH:+:$PYTHONPATH}"
  /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python \
    /srv/ai/src/qwen-quality-fast-correct-tp6-20261010/bootstrap_profile.py \
    --profile-root /srv/ai/src/qwen-quality-fast-correct-tp6-20261010 \
    --api-pid "$$" >>/srv/ai/src/qwen-quality-fast-correct-tp6-20261010/bootstrap.log 2>&1 &
fi
exec bash "/srv/ai/src/qwen-quality-fast-correct-tp6-20261010/start-paced-tp6.sh" "$@"
