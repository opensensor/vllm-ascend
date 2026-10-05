#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Isolated 310P operator gate; does not launch or change the model server.
set -eo pipefail

output=${1:?usage: run_qsa_position_metadata_gate.sh /path/to/new/output.jsonl}
snapshot=/srv/ai/src/qsa-position-metadata-probe-20261004
python=/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python
probe="$snapshot/opp/vendors/qsa_position_metadata_probe_transformer"
embedded="$snapshot/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export PYTHONPATH="$snapshot:/srv/ai/src/vllm-opensensor:${PYTHONPATH:-}"
export ASCEND_CUSTOM_OPP_PATH="$probe:$embedded${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$probe/op_api/lib:$embedded/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ASCEND_RT_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

cd "$snapshot"
exec "$python" tools/qwen4exp/benchmark_qsa_position_metadata_aicpu_310.py --output "$output"
