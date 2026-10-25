# SPDX-License-Identifier: Apache-2.0
"""Reserve lazy M3 communication resources before an explicit large KV pool."""

import torch

from vllm_ascend._310p.ops.minimax_m3 import W4_LAYOUT


def warmup_minimax_before_cache(worker):
    text = getattr(getattr(worker, "model_config", None), "hf_text_config", None)
    if getattr(text, "ascend_minimax_m3_w4", None) != W4_LAYOUT:
        return False
    # Weight conversion/loading can leave large free blocks in PyTorch's pool.
    # HCCL allocates outside that pool, so release them before its first call.
    torch.npu.synchronize()
    torch.npu.empty_cache()
    # The custom pipeline transport already warms its channels at construction.
    # The normal profile initializes TP reductions and real projection workspaces.
    # Otherwise the explicit-budget fast path delays these allocations until the
    # first request, when 310P's huge-page allocator can no longer satisfy HCCL.
    worker.model_runner.profile_run()
    torch.npu.synchronize()
    torch.npu.empty_cache()
    return True
