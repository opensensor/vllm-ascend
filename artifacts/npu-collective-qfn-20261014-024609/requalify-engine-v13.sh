#!/usr/bin/env bash
# Restore the qualified worker owners after the approved engine-control restart.
set -eo pipefail
experiment_root=/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609
source /srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-hardware-env.sh
export PYTHONPATH="${experiment_root}/resident-source-v1:${experiment_root}/qfn-pinned-position-runtime:/srv/ai/src/vllm-opensensor:${PYTHONPATH:-}"
cd "$experiment_root/resident-source-v1"
python_bin=/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python
run_gate() {
  "$python_bin" "$experiment_root/qfn-recovery-v13.py" "$@"
}
run_gate install
run_gate load --kind w4 --rows 650
run_gate schedule --kind w4 --rows 650 --projection gate_up_proj
run_gate schedule --kind w4 --rows 650 --projection down_proj
run_gate validate --kind w4 --rows 650
run_gate schedule --kind w4 --rows 20480 --projection gate_up_proj
run_gate schedule --kind w4 --rows 20480 --projection down_proj
run_gate validate --kind w4 --rows 20480
run_gate load --kind hc --rows 68
for rows in 68 640 2048; do
  run_gate validate --kind hc --rows "$rows"
done
run_gate load --kind gather --rows 1
for rows in 1 3 64; do
  run_gate validate --kind gather --rows "$rows"
done
run_gate load --kind geometry --rows 3
run_gate validate --kind geometry --rows 3
run_gate restore
"$python_bin" "$experiment_root/qsa-resident-validation.py" \
  --run-root "$experiment_root/resident-live-v13" \
  --output "$experiment_root/resident-live-v13/requalification/model-validation.json" \
  --parallel-requests 4
