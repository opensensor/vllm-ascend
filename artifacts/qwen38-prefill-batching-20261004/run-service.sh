#!/usr/bin/env bash
set -eo pipefail

arm=${1:?baseline or batch1536 required}
case "$arm" in
  baseline)
    runtime=/srv/ai/src/qwen38-cann-finalize-gate-20261004
    opp=/srv/ai/src/qwen38-coherent-opp-20261001-r2/vendors/qwen38_coherent_transformer
    ;;
  batch1536)
    runtime=/srv/ai/src/qwen38-batch1536-runtime-20261004
    opp=/srv/ai/src/qwen38-batch-switch-128-opp-20261004/vendors/qwen38_coherent_transformer
    ;;
  *)
    echo "unknown arm: $arm" >&2
    exit 2
    ;;
esac

results=/srv/ai/src/qwen38-cann-finalize-gate-20261004/prefill-batching-20261004
export QWEN38_PLUGIN_ROOT="$runtime"
export QWEN38_COHERENT_OPP="$opp"
export SERVED_MODEL_NAME="qwen38-w4-batch-$arm"
export PORT=8001
export LOG="$results/server-$arm.log"
export WATCHDOG_LOG="$results/watchdog-$arm.log"
export AFFINITY_LOG="$results/affinity-$arm.log"
exec bash "$runtime/examples/start_qwen38_flash_next_w4_310p.sh"
