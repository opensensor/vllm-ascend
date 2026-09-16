# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend Qwen4Exp PLE (parallel layer embedding) layer stub (plan T1.2).

Full PLE short-conv + grouped-norm logic lands in T1.3. This stub fixes the
class surface and pins every dtype through the authoritative policy so no dtype
literal is spelled here.
"""

from __future__ import annotations

import torch
from torch import nn

from .dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY, Qwen4ExpDtypePolicy


class AscendQwen4ExpPLELayer(nn.Module):
    """PLE injection layer (skeleton).

    The projection runs in ``policy.ple_projection_dtype``; the grouped RMSNorm
    accumulates in ``policy.ple_norm_accumulation_dtype`` before casting back.
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
        self.projection_dtype = dtype_policy.cast_site("ple_projection")
        self.norm_accumulation_dtype = dtype_policy.cast_site("ple_norm_accumulation")
        # TODO(T1.3): build the short-conv state, grouped RMSNorm, and PLE
        # projection reading dtypes from ``dtype_policy``.

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError("TODO(T1.3): AscendQwen4ExpPLELayer.forward is implemented in a later task.")


__all__ = ["AscendQwen4ExpPLELayer"]
