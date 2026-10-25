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
    --manifest /vllm-workspace/minimax-native-v12/manifest.json \
    --output /vllm-workspace/minimax-native-startup-load.json
python -m tools.minimax_m3.resident switch --native-name minimax_w4_v12 \
    --output /vllm-workspace/minimax-native-startup-switch.json
python -m tools.minimax_m3.resident_int8 enable \
    --manifest /vllm-workspace/minimax-native-int8-v15/manifest.json \
    --baseline minimax_w4_v12 \
    --output /vllm-workspace/minimax-native-startup-int8.json
python -m tools.minimax_m3.resident load-native \
    --manifest /vllm-workspace/minimax-activation-v2/manifest.json \
    --output /vllm-workspace/minimax-native-startup-activation-load.json
python -m tools.minimax_m3.resident switch --native-name minimax_w4_int8_v15 \
    --activation-name minimax_activation_v2 --partial-graphs \
    --module-file /vllm-workspace/vllm-ascend/vllm_ascend/_310p/ops/minimax_m3.py \
    --output /vllm-workspace/minimax-native-startup-activation-switch.json
# Snapshot the complete generation for the drained dense conversion transaction.
python - <<'SOURCE'
from pathlib import Path
from tools.minimax_m3.resident import candidate_source
root = Path('/vllm-workspace/vllm-ascend')
source = candidate_source(
    (root / 'vllm_ascend/_310p/ops/minimax_m3.py').read_text(),
    'minimax_w4_int8_v15', 'minimax_activation_v2',
    (root / 'tools/minimax_m3/partial_graph.py').read_text(),
)
Path('/workspace/m31-startup-serving-source.py').write_text(source)
SOURCE
python -m tools.minimax_m3.resident_dense_int8 enable \
    --serving-source /workspace/m31-startup-serving-source.py \
    --output /vllm-workspace/minimax-dense-int8-startup.json

# Set operator determinism after releasing captures, then restore the full generation.
python -m tools.minimax_m3.resident_determinism enable \
    --serving-source /workspace/m31-startup-serving-source.py \
    --output /vllm-workspace/minimax-determinism-startup.json
