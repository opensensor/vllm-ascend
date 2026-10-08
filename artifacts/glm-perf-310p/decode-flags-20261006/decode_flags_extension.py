# SPDX-License-Identifier: Apache-2.0
"""Startup loading of existing native adapters for the config flag study."""

from pathlib import Path

import torch
from live_score_extension import GlmScoreResidentExtension

if not hasattr(torch.ops._C_ascend, "npu_w2_swiglu_310"):
    assert not hasattr(torch.ops._C_ascend, "npu_w2_route_combine_310")
    torch.ops.load_library(str(Path(__file__).with_name("glm_decode_flags.so")))


class DecodeFlagsExtension(GlmScoreResidentExtension):
    def decode_flags_status(self):
        banks = []
        for role, wrapper in zip(("target", "draft"), self._resident_wrappers()):
            for name, module in wrapper.runnable.named_modules():
                bank = getattr(module, "w2_experts", None)
                if bank is not None:
                    banks.append(
                        {
                            "role": role,
                            "module": name,
                            "decode_swiglu": bank.decode_swiglu,
                            "decode_combine": bank.decode_combine,
                            "offload_to_cpu": bank.offload_to_cpu,
                        }
                    )
        return {"rank": torch.distributed.get_rank(), "banks": banks}
