# SPDX-License-Identifier: Apache-2.0
"""Load the separately named candidate once; expose existing resident controls."""

from pathlib import Path

import torch

from tools.glm_perf.resident_worker import ResidentWorkerExtension
from vllm_ascend.utils import enable_custom_op

enable_custom_op()
if not hasattr(torch.ops._C_ascend, "npu_glm_mhc_post_310"):
    torch.ops.load_library(str(Path(__file__).with_name("glm_native_mhc_post_candidate.so")))


class NativeMhcResidentExtension(ResidentWorkerExtension):
    pass
