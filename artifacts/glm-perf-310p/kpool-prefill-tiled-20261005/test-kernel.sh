#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Future hardware gate. Do not run while the public server owns the devices.
set -eo pipefail
source /srv/ai/bin/ascend-env.sh
set -u
runtime_root=${1:?source tree containing the staged Python and hardware tests}
experiment_root=${2:?experiment directory containing the compiled binding}
package_root=${3:?isolated prefill OPP package root}
export ASCEND_RT_VISIBLE_DEVICES=${4:-0}
export ASCEND_CUSTOM_OPP_PATH="$package_root/packages/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH:-}"
export PYTHONPATH="$runtime_root:${PYTHONPATH:-}"
cd "$runtime_root"
test_file=tests/e2e/nightly/310p/single_node/ops/test_glm_kpool_prefill_score_310.py
/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python - "$experiment_root" "$test_file" <<'PY'
import sys
from pathlib import Path
import torch
import torch_npu
import pytest
from vllm_ascend.utils import enable_custom_op
enable_custom_op()
torch.ops.load_library(str(Path(sys.argv[1]) / "glm_kpool_prefill_score_candidate.so"))
raise SystemExit(pytest.main(["--noconftest", "-q", "-x", sys.argv[2]]))
PY
/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python -m tools.glm_perf.benchmark_kpool_prefill_310 \
  --extension "$experiment_root/glm_kpool_prefill_score_candidate.so" \
  --fixtures "$test_file" --output "$experiment_root/selector-results.json"
