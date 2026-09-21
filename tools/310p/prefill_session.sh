#!/bin/bash
# One box window, every prefill measurement, no babysitting.
#
# The box is shared with the GLM bring-up and gets reclaimed without warning,
# so this writes each result the moment it lands and always leaves the box in a
# known state. Run it and walk away:
#
#     setsid nohup bash ~/prefill_session.sh > ~/logs/prefill_session.log 2>&1 &
#     tail -f ~/prefill_session_results.txt
#
# To stop early: `kill -TERM` the script (never -9 -- see the NPU note below).
# Whatever finished is already in the results file, and STAGE 5 still runs.
set +u

RESULTS=$HOME/prefill_session_results.txt
say() { echo "== $* ==" | tee -a "$RESULTS"; }

# The 310P chips wedge if a process is killed mid-NPU-operation and only a
# reboot clears it. Every stop below is SIGTERM followed by a wait.
stop_vllm() {
  for p in $(pgrep -f "v[l]lm serve"); do kill -TERM "$p" 2>/dev/null; done
  for _ in $(seq 1 40); do pgrep -f "v[l]lm serve" >/dev/null || return 0; sleep 3; done
  echo "WARNING: vllm still up after 120s" | tee -a "$RESULTS"
}
trap 'say "interrupted, restoring production"; stop_vllm; bash "$HOME/start_server.sh"; exit 130' TERM INT

: > "$RESULTS"
say "started $(date -Is)"

source /srv/ai/bin/ascend-env.sh >/dev/null 2>&1
source /srv/ai/venvs/fork028/bin/activate
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 SOC_VERSION=ascend310p1 TASK_QUEUE_ENABLE=2

stop_vllm

# STAGE 1 -- per-chip budget. Slow: CANN compiles the uncached fp32 5D shapes
# in the WY/UT sections on first call, which alone took ~10 min. It streams to
# its own file as each section lands.
say "STAGE 1 per-chip budget"
python3 "$HOME/prefill_budget.py" 2>&1 | grep -vE "^(INFO|WARNING|Warning|\[W)" | tee -a "$RESULTS"

# STAGE 2 -- the 128 all-reduces a step does, at the real message size.
say "STAGE 2 collectives (TP4)"
torchrun --nproc_per_node=4 "$HOME/prefill_budget.py" --collective 2>&1 \
  | grep -vE "^(INFO|WARNING|Warning|\[W)" | tee -a "$RESULTS"

# STAGE 3/4 -- served prefill, blocked WY inverse vs the row substitution it
# replaced. Same server config either way, so the difference is the transform
# and nothing else. This is the attribution that 6b4a24b03 still lacks.
for mode in 1 0; do
  label=$([ "$mode" = 1 ] && echo "blocked inverse (default)" || echo "row substitution (legacy)")
  say "STAGE $([ "$mode" = 1 ] && echo 3 || echo 4) served prefill -- $label"
  stop_vllm
  VLLM_ASCEND_GDN_UT_BLOCKED=$mode bash "$HOME/start_server.sh" >/dev/null 2>&1
  up=""
  for i in $(seq 1 170); do
    curl -sf -m 5 http://127.0.0.1:8080/v1/models >/dev/null 2>&1 && { up="yes"; echo "  serving after ~$((i*5))s" | tee -a "$RESULTS"; break; }
    sleep 5
  done
  if [ -z "$up" ]; then
    echo "  SERVER DID NOT COME UP -- see ~/logs/serve_27b.log" | tee -a "$RESULTS"
    continue
  fi
  python3 "$HOME/prefill_probe.py" 2>&1 | tee -a "$RESULTS"
done

# STAGE 4b -- the grouped WY gram, which is opt-in because it leans on a 6D
# broadcast matmul that is exact on CPU but unverified on Ascend. If it faults
# or regresses here, it simply stays off; nothing in production depends on it.
say "STAGE 4b served prefill -- grouped WY gram (opt-in)"
stop_vllm
VLLM_ASCEND_GDN_WY_GROUPED_GRAM=1 bash "$HOME/start_server.sh" >/dev/null 2>&1
up=""
for i in $(seq 1 170); do
  curl -sf -m 5 http://127.0.0.1:8080/v1/models >/dev/null 2>&1 && { up="yes"; echo "  serving after ~$((i*5))s" | tee -a "$RESULTS"; break; }
  sleep 5
done
if [ -n "$up" ]; then
  python3 "$HOME/prefill_probe.py" 2>&1 | tee -a "$RESULTS"
else
  echo "  DID NOT COME UP -- grouped gram likely unsupported; see ~/logs/serve_27b.log" | tee -a "$RESULTS"
  grep -iE "error|not support|Traceback" "$HOME/logs/serve_27b.log" | tail -5 | tee -a "$RESULTS"
fi

# STAGE 5 -- hand the box back the way it was found.
say "STAGE 5 restoring production defaults"
stop_vllm
bash "$HOME/start_server.sh" 2>&1 | tail -3 | tee -a "$RESULTS"
say "done $(date -Is)"
