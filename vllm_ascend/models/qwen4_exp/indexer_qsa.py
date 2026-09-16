# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend Qwen4Exp QSA indexer stub (plan T1.2).

Full sparse top-k indexer logic lands in T1.5/T4.x. This stub fixes the class
surface and pins the indexer dtype through the authoritative policy (no dtype
literals here).
"""

from __future__ import annotations

import torch
from torch import nn

from .dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY, Qwen4ExpDtypePolicy


class AscendQwen4ExpQSAIndexer(nn.Module):
    """QSA sparse index producer (skeleton).

    Runs the compressed indexer in ``policy.qsa_indexer_dtype``.
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
        self.indexer_dtype = dtype_policy.cast_site("qsa_indexer")
        # TODO(T1.5/T4.x): build the indexer projections + top-k selection
        # reading dtypes from ``dtype_policy``.

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError("TODO(T1.5/T4.x): AscendQwen4ExpQSAIndexer.forward is implemented in a later task.")


__all__ = ["AscendQwen4ExpQSAIndexer"]
