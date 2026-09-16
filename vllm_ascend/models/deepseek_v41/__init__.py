# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend 310P DeepSeek V4.1 (552B at 2-bit experts) model package (plan E1.x).

Host-side math for the W2 (2-bit) routed-expert path. Importable in isolation:
the runtime W2->INT8 active-expert unpack and grouped QDQ MoE forward (E1.2) pull
only the canonical E1.1 packed-W2 format and the shared per-token INT8
activation quantizer -- no Triton/CUDA kernel modules on the 310P path. The E1.3
device bridge wraps :func:`unpack_active_experts` /
:func:`w2_active_moe_forward` onto the NPU INT8 grouped matmul.
"""

from .dtype_policy import (
    ASCEND_DEEPSEEKV41_DTYPE_POLICY,
    REQUIRED_CAST_SITES,
    DeepseekV41DtypePolicy,
)
from .w2_unpack import (
    ActiveExpertWeights,
    route_topk_w2,
    swiglu_gate_up,
    unpack_active_experts,
    w2_active_moe_forward,
    w2_group_qdq_linear,
)

# NOTE: the E2.1 model classes (AscendDeepseekV41ForCausalLM / ...ForConditional
# Generation / DeepSeekV41MTP) live in :mod:`.model` and are intentionally NOT
# imported here. Their shipped ``deepseek_v4`` base is resolved lazily so that
# importing this package stays Triton-free on the 310P path; import them from
# ``vllm_ascend.models.deepseek_v41.model`` (or via the ModelRegistry arch name).

__all__ = [
    # E2.1 dtype policy (authoritative)
    "ASCEND_DEEPSEEKV41_DTYPE_POLICY",
    "REQUIRED_CAST_SITES",
    "DeepseekV41DtypePolicy",
    # E1.2 W2 host-math kernel
    "ActiveExpertWeights",
    "route_topk_w2",
    "swiglu_gate_up",
    "unpack_active_experts",
    "w2_active_moe_forward",
    "w2_group_qdq_linear",
]
