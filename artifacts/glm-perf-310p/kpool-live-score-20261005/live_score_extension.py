# SPDX-License-Identifier: Apache-2.0
"""Load isolated candidate bindings before graph capture."""
from pathlib import Path
import torch
from tools.glm_perf.resident_worker import ResidentWorkerExtension
from vllm_ascend.utils import enable_custom_op

enable_custom_op()
if not hasattr(torch.ops._C_ascend, "npu_glm_mhc_post_310"):
    torch.ops.load_library("/home/matteius/experiments/glm-native-mhc-post-20261005/glm_native_mhc_post_candidate.so")
if not hasattr(torch.ops._C_ascend, "npu_glm_kpool_score_310"):
    torch.ops.load_library(str(Path(__file__).with_name("glm_kpool_score_candidate.so")))

class GlmScoreResidentExtension(ResidentWorkerExtension):
    pass
