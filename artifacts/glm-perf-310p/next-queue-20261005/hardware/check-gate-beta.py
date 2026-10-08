# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401 - register the NPU backend

from tools.glm_perf.kda_gate_beta_native import GateBeta

root = Path(__file__).resolve().parent
build = root / "gate-beta-build"
torch.npu.set_device(0)
torch.ops.load_library(str(build / "glm_kda_prepare_bridge_v1.so"))
spec = importlib.util.spec_from_file_location("validate_gate", root / "staged/validate-kda-gate-beta.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    op = GateBeta(str(build / "kda-gate-beta-v1.bin"), heads=16, lower_bound=-5.0, device=torch.device("npu:0"))
    result = module.validate(op)
except Exception as exc:
    result = {"passed": False, "error": repr(exc)}
(root / "gate-beta-parity.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
