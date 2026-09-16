# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend 310P GLM-5.3-Flash W2 (288 experts at 2-bit) model package (plan G3+).

Host-side math for the W2 (2-bit) routed-expert path of GLM-5.3-Flash
(``glm5_next``). Importable in isolation: this package init eagerly exposes only
the authoritative dtype policy (Triton-free, cheap) and lazily forwards the W2
routed-expert host math -- reusing the DeepSeek E1.2/E1.3 W2->INT8 unpack +
grouped-QDQ MoE forward (the plan mandates GLM reuse that kernel; only routed
experts are quantized, GLM has no Engram and no separate sparse indexer format).

The G3 model classes (``AscendGlm5NextW2ForCausalLM`` /
``...ForConditionalGeneration`` / ``Glm5NextW2MTP``) live in :mod:`.model` and
are intentionally NOT imported here. Their shipped ``glm5next`` base is resolved
lazily so that importing this package stays Triton-free on the 310P path (the
shipped ``glm5next.model`` pulls ``FusedMoEFactory`` and the Triton KDA op);
import them from ``vllm_ascend.models.glm5next_w2.model`` or via the
ModelRegistry arch name.
"""

from .dtype_policy import (
    ASCEND_GLM5NEXT_W2_DTYPE_POLICY,
    REQUIRED_CAST_SITES,
    Glm5NextW2DtypePolicy,
)

# The W2 routed-expert host-math kernel is REUSED from the DeepSeek E1.2/E1.3
# work (``deepseek_v41.w2_unpack``; itself Triton-free). It is forwarded lazily
# via PEP 562 so this package init stays light and the (future) GLM-specific W2
# MoE forward (G6) can transparently supersede it here without a call-site
# change. G6 may replace this forwarding with a ``glm5next_w2`` local module.
_W2_FORWARD_EXPORTS = (
    "ActiveExpertWeights",
    "route_topk_w2",
    "swiglu_gate_up",
    "unpack_active_experts",
    "w2_active_moe_forward",
    "w2_group_qdq_linear",
)

# G5 DSA (deepseek_sparse_attention) 310P path. Lazily forwarded from the local
# ``.dsa`` module (Triton-free: it reuses the deepseek_v41 indexer *selection*
# and adds GLM's NoPE MLA core). Kept lazy so this package init stays light --
# ``.dsa`` pulls torch + the deepseek_v41 indexer only on first access.
_DSA_EXPORTS = (
    "AscendGlm5NextW2DSA",
    "Glm5NextW2DsaIndexer",
    "DsaIndexerResult",
    "DsaSelection",
)


# G4 KDA (Kimi Delta Attention) linear-attention 310P path. Lazily forwarded
# from the local ``.kda`` module (Triton-free eager gated-delta recurrence,
# validated against the tools/glm_w2 parity oracle). Kept lazy so this package
# init stays light -- ``.kda`` pulls torch (+ optional guarded torch_npu) only
# on first access.
_KDA_EXPORTS = (
    "Glm5NextW2KDA",
)


# G6 routed-expert W2 MoE (host router -> E1.3 W2 method -> eager combine) +
# multi-head hyper-connection ops. Lazily forwarded from the local ``.moe``
# module (Triton-free: reuses the DeepSeek E1.2/E1.3 W2 kernel + combine). Kept
# lazy so importing this package init pulls neither ``.moe`` nor the E1.3 method
# (which needs torch_npu) until first access.
_MOE_EXPORTS = (
    "Glm5NextW2MoE",
    "glm_route_topk",
    "eager_moe_combine",
    "hc_pre",
    "hc_post",
    "hc_expand",
    "hc_contract",
)

# G6 weight mapping: name -> W2 destination classification, expert fusion, and
# coverage validation for the streamed loader. Lazily forwarded from ``.weight_mapping``.
_WEIGHT_MAPPING_EXPORTS = (
    "classify_tensor",
    "map_expert_tensor",
    "validate_weight_map",
    "moe_block_ids",
    "estimate_per_chip_bytes",
    "iter_artifact_tensor_metas",
)


def __getattr__(name: str):  # PEP 562 module-level lazy attribute
    if name in _W2_FORWARD_EXPORTS:
        from vllm_ascend.models import deepseek_v41 as _dsv41

        return getattr(_dsv41, name)
    if name in _DSA_EXPORTS:
        from . import dsa as _dsa

        return getattr(_dsa, name)
    if name in _KDA_EXPORTS:
        from . import kda as _kda

        return getattr(_kda, name)
    if name in _MOE_EXPORTS:
        from . import moe as _moe

        return getattr(_moe, name)
    if name in _WEIGHT_MAPPING_EXPORTS:
        from . import weight_mapping as _wm

        return getattr(_wm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        [
            *globals().keys(),
            *_W2_FORWARD_EXPORTS,
            *_DSA_EXPORTS,
            *_KDA_EXPORTS,
            *_MOE_EXPORTS,
            *_WEIGHT_MAPPING_EXPORTS,
        ]
    )


__all__ = [
    # G3 dtype policy (authoritative)
    "ASCEND_GLM5NEXT_W2_DTYPE_POLICY",
    "REQUIRED_CAST_SITES",
    "Glm5NextW2DtypePolicy",
    # W2 host-math kernel (reused from DeepSeek E1.2/E1.3; lazy)
    *_W2_FORWARD_EXPORTS,
    # G5 DSA sparse-attention path (lazy; local .dsa module)
    *_DSA_EXPORTS,
    # G4 KDA linear-attention path (lazy; local .kda module)
    *_KDA_EXPORTS,
    # G6 routed W2 MoE + hyper-connection (lazy; local .moe module)
    *_MOE_EXPORTS,
    # G6 weight mapping (lazy; local .weight_mapping module)
    *_WEIGHT_MAPPING_EXPORTS,
]
