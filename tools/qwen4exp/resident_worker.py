# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in Qwen worker extension using the existing resident experiment control.

Import only through --worker-extension-cls on a diagnostic server. Unlike the
GLM metadata audit extension, this does not copy attention metadata to the host.
The shared controller supplies request draining, restoration and recapture.
"""

from tools.glm_perf.resident_worker import ResidentWorkerExtension
from vllm_ascend.compilation.breakable_aclgraph import BreakableACLGraphWrapper


class QwenResidentExtension(ResidentWorkerExtension):
    def resident_reset(self):
        # The controller drains requests and resets scheduler caches first.
        # The base reset verifies pending execution and clears request tensors;
        # tier metadata also owns checkpoints outside runner.kv_caches.
        super().resident_reset()
        for tier in getattr(self.model_runner, "_prefix_mamba_tiers", {}).values():
            tier.reset()
        return self.resident_status()

    def resident_status(self):
        result = super().resident_status()
        result["prefix_mamba"] = {
            str(group_id): tier.cache_status()
            for group_id, tier in getattr(self.model_runner, "_prefix_mamba_tiers", {}).items()
        }
        return result


def install_direct_dispatch():
    original_call = BreakableACLGraphWrapper.__call__

    def call(self, *args, **kwargs):
        if self.__dict__.get("_resident_direct", False):
            return self.runnable(*args, **kwargs)
        return original_call(self, *args, **kwargs)

    BreakableACLGraphWrapper.__call__ = call


install_direct_dispatch()
