# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend 310P Qwen4Exp (Qwen3.8-Flash-Next) model package (plan T1.2).

Importable in isolation: the authoritative dtype policy has no heavy deps, and
model/MTP classes only pull core vLLM layers (no Triton/CUDA kernel modules on
the 310P path). Registration happens in ``vllm_ascend/models/__init__.py`` via
string entry points, so importing this package is optional at register time.
"""

from .dtype_policy import (
    ASCEND_QWEN4EXP_DTYPE_POLICY,
    REQUIRED_CAST_SITES,
    Qwen4ExpDtypePolicy,
)
from .model import (
    AscendQwen4ExpForCausalLM,
    AscendQwen4ExpForConditionalGeneration,
    AscendQwen4ExpModel,
)

__all__ = [
    "ASCEND_QWEN4EXP_DTYPE_POLICY",
    "REQUIRED_CAST_SITES",
    "AscendQwen4ExpForCausalLM",
    "AscendQwen4ExpForConditionalGeneration",
    "AscendQwen4ExpModel",
    "Qwen4ExpDtypePolicy",
]
