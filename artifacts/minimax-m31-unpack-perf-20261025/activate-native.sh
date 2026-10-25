#!/usr/bin/env bash
# Restore the qualified native dispatch after worker startup, without rereading weights.
# shellcheck disable=SC1091
source /workspace/minimax-m31-env.sh
set -euo pipefail
cd /workspace
for attempt in $(seq 1 120); do
    if curl --max-time 2 -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
        break
    fi
    if [[ "$attempt" == 120 ]]; then
        echo 'MiniMax did not become healthy before native activation' >&2
        exit 1
    fi
    sleep 5
done
python -m tools.minimax_m3.resident load-native \
    --manifest /vllm-workspace/minimax-native-v9/manifest.json \
    --output /vllm-workspace/minimax-native-startup-load.json
python -m tools.minimax_m3.resident switch --native-name minimax_w4_v9 \
    --output /vllm-workspace/minimax-native-startup-switch.json
