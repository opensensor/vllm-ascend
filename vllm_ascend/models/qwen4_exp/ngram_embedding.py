# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend Qwen4Exp n-gram embedding stub (plan T1.2).

Full n-gram / PLE vocab-parallel embedding logic lands in T1.3. This stub fixes
the class surface and pins the embedding dtype through the authoritative policy
(no dtype literals here).
"""

from __future__ import annotations

import torch
from torch import nn

from .dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY, Qwen4ExpDtypePolicy


class AscendQwen4ExpNGramEmbedding(nn.Module):
    """N-gram embedding (skeleton).

    Materializes n-gram embeddings in ``policy.ngram_embedding_dtype``.
    """

    def __init__(
        self,
        *,
        config: object,
        dtype_policy: Qwen4ExpDtypePolicy = ASCEND_QWEN4EXP_DTYPE_POLICY,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.dtype_policy = dtype_policy
        self.embedding_dtype = dtype_policy.cast_site("ngram_embedding")
        # TODO(T1.3): build the n-gram hashing + PLE vocab-parallel embedding
        # reading dtypes from ``dtype_policy``.

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError("TODO(T1.3): AscendQwen4ExpNGramEmbedding.forward is implemented in a later task.")


__all__ = ["AscendQwen4ExpNGramEmbedding"]
