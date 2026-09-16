# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend Qwen4Exp multi-token-prediction (MTP) stub (plan T1.2).

The ``Qwen4ExpMTP`` architecture is REGISTERED by T1.2 so checkpoints resolve,
but MTP is NOT wired here -- the speculative-decode head is built in S2. This
file provides the registerable class surface only; every dtype flows from the
authoritative policy (no dtype literals here).
"""

from __future__ import annotations

import torch
from torch import nn
from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsPP

from .dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY, Qwen4ExpDtypePolicy

try:  # pragma: no cover - typing-only import guard
    from vllm.config import VllmConfig
except Exception:  # pragma: no cover
    VllmConfig = object  # type: ignore[assignment, misc]


class AscendQwen4ExpMTP(nn.Module, SupportsPP, MixtureOfExperts):
    """Qwen4Exp MTP draft head (registered, not wired).

    TODO(S2): build the MTP predictor + shared backbone reuse. Construction is
    intentionally deferred so the arch name resolves without pulling MTP logic.
    """

    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.dtype_policy: Qwen4ExpDtypePolicy = Qwen4ExpDtypePolicy.from_vllm_config(vllm_config)
        raise NotImplementedError("TODO(S2): AscendQwen4ExpMTP is registered by T1.2 but wired in S2.")

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError("TODO(S2): AscendQwen4ExpMTP.forward")


_DEFAULT_DTYPE_POLICY = ASCEND_QWEN4EXP_DTYPE_POLICY

__all__ = ["AscendQwen4ExpMTP"]
