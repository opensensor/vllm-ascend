# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend Qwen4Exp QSA (query-sparse attention) stub (plan T1.2).

Full QSA attention + Ascend FlashAttention wiring lands in T1.4/T4.x. This stub
fixes the class surface and pins the QSA main / KV-cache dtypes through the
authoritative policy (no dtype literals here). No Triton/CUDA kernel module is
imported on the 310P path.
"""

from __future__ import annotations

import torch
from torch import nn

from .dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY, Qwen4ExpDtypePolicy


class AscendQwen4ExpQSAAttention(nn.Module):
    """Query-sparse attention (skeleton).

    Runs in ``policy.qsa_main_dtype`` with a ``policy.kv_cache_dtype`` KV cache
    and ``policy.attention_accumulation_dtype`` score accumulation.
    """

    def __init__(
        self,
        *,
        config: object,
        layer_idx: int,
        dtype_policy: Qwen4ExpDtypePolicy = ASCEND_QWEN4EXP_DTYPE_POLICY,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype_policy = dtype_policy
        self.qsa_dtype = dtype_policy.cast_site("qsa")
        self.kv_cache_dtype = dtype_policy.cast_site("kv_cache")
        self.accumulation_dtype = dtype_policy.cast_site("attention_accumulation")
        # TODO(T1.4/T4.x): build the QSA projections, the Ascend FlashAttention
        # backend, and the indexer wiring reading dtypes from ``dtype_policy``.

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError("TODO(T1.4/T4.x): AscendQwen4ExpQSAAttention.forward is implemented in a later task.")


__all__ = ["AscendQwen4ExpQSAAttention"]
