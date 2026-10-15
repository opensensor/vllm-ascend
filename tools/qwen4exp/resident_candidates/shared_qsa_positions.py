# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Share a causal-position copy across QSA layers within one target forward.

Opt-in resident experiment; requires the model's position-cache dispatch seam.
MRoPE coordinates never substitute for logical causal positions. Captured
callbacks refresh the shared output on every replay and check that all users
still receive the same scheduler metadata object. No cross-step cache exists.
"""

import torch
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.forward_context import get_forward_context

from vllm_ascend.models.qwen4_exp.model import _host_qsa_logical_positions, _QSAAttention


class SharedPositions:
    def __init__(self, positions: torch.Tensor, prefix: str):
        self.positions = positions
        self.prefixes = {prefix}


def step_logical_positions(metadata, num_tokens, device, rope_positions, layer_prefix, cache):
    if rope_positions.ndim == 1 or cache is None or layer_prefix is None:
        return _QSAAttention._logical_query_positions(metadata, num_tokens, device, rope_positions, layer_prefix)
    # Share only identical metadata, not merely equal capture-time lengths.
    # Different KV groups can have equal lengths at capture and diverge later.
    key = (id(metadata), num_tokens, device)
    shared = cache.get(key)
    if shared is not None:
        shared.prefixes.add(layer_prefix)
        return shared.positions
    capture = BreakableCUDAGraphCapture.current()
    if capture is None or not capture._capturing:
        positions = _host_qsa_logical_positions(metadata, num_tokens).to(device=device, non_blocking=True)
        shared = SharedPositions(positions, layer_prefix)
    else:
        from vllm_ascend.utils import weak_ref_tensor

        positions = torch.empty(num_tokens, dtype=torch.int64, device=device)
        shared = SharedPositions(positions, layer_prefix)
        weak_positions = weak_ref_tensor(positions)
        # The callback must retain only the weak graph tensor, not the owning
        # tensor. Keep the mutable prefix set independently for replay guards.
        prefixes = shared.prefixes

        def refresh():
            metadata_by_layer = get_forward_context().attn_metadata
            current = metadata_by_layer[layer_prefix]
            if any(metadata_by_layer[prefix] is not current for prefix in prefixes):
                raise RuntimeError("shared QSA causal positions require identical replay metadata across layers")
            causal = _host_qsa_logical_positions(current, num_tokens)
            if causal.numel() != num_tokens:
                raise RuntimeError("shared QSA causal positions do not cover the captured token count")
            weak_positions.copy_(causal.to(device=device, non_blocking=True))

        capture.add_eager(refresh)
    cache[key] = shared
    return shared.positions


def replacements():
    return {"vllm_ascend.models.qwen4_exp.model:_step_qsa_logical_positions": step_logical_positions}
