#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Uses one NPU. Run only after hardware is released for this experiment.
set -eo pipefail
source /srv/ai/bin/ascend-env.sh
set -u
runtime_root=${1:?runtime source root}
output_root=${2:?experiment directory containing tests, benchmark, and binding}
physical_device=${3:-1}
export ASCEND_RT_VISIBLE_DEVICES="$physical_device"
export ASCEND_CUSTOM_OPP_PATH="/srv/ai/src/glm-l1-wide-build-20261004/opp-kpool-live-score-20261005/packages/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH:-}"
export PYTHONPATH="$runtime_root:${PYTHONPATH:-}"
cd "$runtime_root"
/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python - "$output_root" <<'PY'
import sys
from pathlib import Path
import torch
import torch_npu
import pytest
from vllm_ascend.utils import enable_custom_op
root = Path(sys.argv[1])
enable_custom_op()
torch.ops.load_library(str(root / "glm_kpool_score_candidate.so"))
torch.npu.set_device(0)
torch_npu.npu.set_compile_mode(jit_compile=False)
raise SystemExit(pytest.main(["--noconftest", "-q", "-x", str(root / "test_glm_kpool_score_310.py")]))
PY
