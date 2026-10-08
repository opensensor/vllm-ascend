# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path
import torch
import torch_npu
import pytest
from tools.glm_perf.direct_prefill import DirectPrefillScore
root = Path(__file__).resolve().parent
torch.npu.set_device(0)
torch_npu.npu.set_compile_mode(jit_compile=False)
# Initialize a context and allocate a live tensor BEFORE loading the library.
resident = torch.arange(16, device="npu", dtype=torch.float32)
address = resident.data_ptr()
torch.npu.synchronize()
torch.ops.load_library(str(root / "glm_native_bridge_v1.so"))
torch.ops._C_ascend.npu_glm_kpool_prefill_score_310 = DirectPrefillScore(str(root / "prefill-v1.bin"))
status = pytest.main(["--noconftest", "-q", "-x", str(root / "test_glm_kpool_prefill_score_310.py"), *sys.argv[1:]])
assert resident.data_ptr() == address
torch.testing.assert_close(resident.cpu(), torch.arange(16, dtype=torch.float32))
print({"resident_address_unchanged": True, "status": status}, flush=True)
raise SystemExit(status)
