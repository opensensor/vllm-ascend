#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run on the NPU host after the preceding server has fully stopped.
set -euo pipefail

variant=${1:?baseline, overlap, or batch1280}
run_label=${2:-$variant}
max_batched_tokens=640
grouped_max_routes=
kv_fraction=0.70
if [[ ! "$run_label" =~ ^[a-z0-9-]+$ ]]; then
  echo "run label must contain lowercase letters, digits, or hyphens" >&2
  exit 1
fi
case "$variant" in
  baseline) opp_root=/srv/ai/src/build-only-glm-w3-nz-csrc-20261004/opp-w3-nz-candidate ;;
  overlap) opp_root=/srv/ai/src/glm-l1-wide-build-20261004/opp-l1-overlap-20261004 ;;
  batch1280|batch1280-kv94|batch1280-kv100)
    opp_root=/srv/ai/src/glm-l1-wide-build-20261004/opp-l1-w3-batch2560-20261004
    max_batched_tokens=1280
    grouped_max_routes=20480
    if [[ "$variant" == batch1280-kv94 ]]; then
      kv_fraction=0.94
    elif [[ "$variant" == batch1280-kv100 ]]; then
      kv_fraction=1.0
    fi
    ;;
  *) echo "unknown variant: $variant" >&2; exit 1 ;;
esac

experiment_root=/home/matteius/experiments/glm-w3-20261004
source_root=/srv/ai/src/glm-selective-w3-nz-test-20261004
checkpoint=/srv/ai/models/GLM-5.3-Flash-selective-W3-310p
qsa_opp=/srv/ai/src/build-only-glm-w3-nz-csrc-20261004/opp-qsa-cube512-candidate
launcher="$experiment_root/serve-l1-overlap-controlled-20261004.sh"
if [[ -n "$grouped_max_routes" ]]; then
  launcher="$experiment_root/serve-l1-expert-batching-controlled-20261004.sh"
fi
log="$experiment_root/server-l1-overlap-$run_label-20261004.log"
pid_file="$experiment_root/server-l1-overlap-$run_label-20261004.pid"
if [[ -n "$(ss -ltnH '( sport = :8001 )')" ]]; then
  echo "port 8001 already has a listener" >&2
  exit 1
fi
if [[ -e "$log" ]]; then
  echo "refusing to overwrite $log" >&2
  exit 1
fi
nohup setsid bash "$launcher" "$source_root" "$checkpoint" \
  196608 "$kv_fraction" "$opp_root" 4 graph "$max_batched_tokens" histogram "" "$qsa_opp" on off off off "$grouped_max_routes" \
  > "$log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$pid_file"
printf 'pid=%s log=%s\n' "$!" "$log"
