# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Authoritative dtype policy for the Ascend 310P DeepSeek V4.1 path (E2.1).

THIS FILE IS THE SINGLE SOURCE OF TRUTH for every dtype decision in the Ascend
DeepSeek V4.1 (552B, 2-bit / W2 experts) 310P implementation. Every downstream
task (E2.2 KV specs, E2.3 Engram host table, E3.1 MLA, E3.2 indexer, E3.3 W2
MoE, E3.4 weight load, E4.x assembly/MTP) MUST read dtypes from
:data:`ASCEND_DEEPSEEKV41_DTYPE_POLICY` (or a policy derived from a
``VllmConfig`` via :meth:`DeepseekV41DtypePolicy.from_vllm_config`) rather than
spelling ``torch.float16`` / ``torch.int8`` / ``torch.uint8`` literals locally.
The package unit test enforces that layer modules route through the policy.

Pinned assumptions (mirrors ``qwen4_exp/dtype_policy.py``)
----------------------------------------------------------
* Main compute dtype is ``float16`` (the 310P has no native bfloat16 matmul
  path; the NVIDIA/AMD DeepSeek forks run bfloat16 / FP8, we do NOT).
* Accumulation / reduction sites (router/softmax, attention scores, RMSNorm,
  MoE combine, logits) run in ``float32`` for numerical stability, then cast
  back to ``float16``.
* Routed experts are **2-bit (W2)**: weights are packed into ``uint8`` code
  banks (E1.1/E1.2 format) with ``float16`` per-group dequant scales, and the
  per-token activation is quantized to ``int8`` (W2A8-dynamic, E1.3). Expert
  matmul accumulates in ``float32``.
* The two Engram n-gram tables (``engram_layer_ids=[1,14]``) are ~W4: rows are
  stored packed in ``uint8`` and dequantized to ``float16`` on gather; the
  n-gram post-mix accumulates in ``float32``.
* MLA / indexer / dense / shared-expert / LM-head all ride the ``float16`` main
  dtype; the latent-KV and indexer caches ride ``float16`` as well.

CANN VERIFICATION NOTE (hardware follow-up)
-------------------------------------------
The float16-main + fp32-accumulation split and the W2A8 / ~W4-Engram storage
dtypes below are the *documented working assumption* for the 310P host lane.
They MUST be verified against the pinned CANN toolkit on real 310P silicon
(INT8 grouped matmul for W2, the ~W4 Engram gather, MLA, and the indexer
top-k). If a kernel only supports a narrower dtype set, override the affected
field here -- do NOT patch call sites.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# The one and only place dtype literals are allowed to live.
_MAIN = torch.float16
_ACCUM = torch.float32
# W2 routed experts: 2-bit codes packed into uint8 banks, INT8 dynamic act.
_W2_PACKED = torch.uint8
_INT8_ACT = torch.int8
# ~W4 Engram n-gram tables: 4-bit rows packed into uint8, dequant to fp16.
_W4_PACKED = torch.uint8


@dataclass(frozen=True)
class DeepseekV41DtypePolicy:
    """Immutable table pinning every DeepSeek V4.1 dtype + cast site for 310P.

    Fields are ``torch.dtype``. Read them; never hardcode dtype literals in
    layer modules.
    """

    # --- top-level compute / accumulation ---------------------------------
    main_dtype: torch.dtype = _MAIN
    accumulation_dtype: torch.dtype = _ACCUM

    # --- embeddings / head / logits ---------------------------------------
    embedding_dtype: torch.dtype = _MAIN
    lm_head_dtype: torch.dtype = _MAIN
    logits_dtype: torch.dtype = _ACCUM

    # --- MoE: router fp32, shared-expert fp16, routed experts W2A8 ---------
    router_dtype: torch.dtype = _ACCUM
    shared_expert_dtype: torch.dtype = _MAIN
    # Routed-expert weight is a packed 2-bit (W2) code bank stored as uint8.
    expert_weight_dtype: torch.dtype = _W2_PACKED
    # Per-token dynamic activation quantization is INT8 (W2A8-dynamic, E1.3).
    expert_activation_dtype: torch.dtype = _INT8_ACT
    # Per-group dequant scales and the fp16 output of the expert matmul.
    expert_scale_dtype: torch.dtype = _MAIN
    expert_dtype: torch.dtype = _MAIN
    # Grouped INT8 matmul accumulates in fp32 before the fp16 combine.
    expert_accumulation_dtype: torch.dtype = _ACCUM

    # --- Engram (~W4 host n-gram tables, layers 1/14) ---------------------
    # Rows are stored packed 4-bit in uint8 and dequantized on gather.
    engram_table_dtype: torch.dtype = _W4_PACKED
    engram_dtype: torch.dtype = _MAIN
    engram_accumulation_dtype: torch.dtype = _ACCUM

    # --- MLA / dense attention --------------------------------------------
    mla_dtype: torch.dtype = _MAIN
    mla_accumulation_dtype: torch.dtype = _ACCUM
    attention_dtype: torch.dtype = _MAIN
    attention_accumulation_dtype: torch.dtype = _ACCUM

    # --- sparse-attention indexer / compressor ----------------------------
    indexer_dtype: torch.dtype = _MAIN
    indexer_accumulation_dtype: torch.dtype = _ACCUM

    # --- dense MLP (non-MoE decoder layers) -------------------------------
    dense_dtype: torch.dtype = _MAIN

    # --- caches -----------------------------------------------------------
    kv_cache_dtype: torch.dtype = _MAIN
    indexer_cache_dtype: torch.dtype = _MAIN

    def cast_site(self, name: str) -> torch.dtype:
        """Resolve a named cast site to its pinned dtype.

        ``name`` is a key of :data:`REQUIRED_CAST_SITES`. This indirection lets
        call sites write ``policy.cast_site("expert")`` and keeps the mapping
        auditable in one place.
        """
        try:
            attr = REQUIRED_CAST_SITES[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown DeepseekV41 cast site {name!r}; known sites: {sorted(REQUIRED_CAST_SITES)}"
            ) from exc
        return getattr(self, attr)

    def as_dict(self) -> dict[str, torch.dtype]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def for_310p(cls) -> DeepseekV41DtypePolicy:
        """Return the pinned 310P policy (float16 main + fp32 accumulation)."""
        return cls()

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig | None) -> DeepseekV41DtypePolicy:
        """Derive a policy from a ``VllmConfig``.

        The 310P policy is authoritative and does not read the model dtype (we
        force float16). We only honour an explicit ``torch.dtype`` KV-cache
        override via ``cache_config.cache_dtype`` when present; everything else
        is pinned. Downstream tasks that need config-driven dtype should extend
        this hook rather than branching at call sites.
        """
        policy = cls.for_310p()
        if vllm_config is None:
            return policy
        cache_config = getattr(vllm_config, "cache_config", None)
        cache_dtype = getattr(cache_config, "cache_dtype", None)
        if isinstance(cache_dtype, torch.dtype) and cache_dtype is not policy.kv_cache_dtype:
            # Respect an explicit torch.dtype override only.
            return cls(**{**policy.as_dict(), "kv_cache_dtype": cache_dtype})
        return policy


# Named cast sites -> policy field. These are the mandatory dtype cast points
# every layer must route through the policy (no bare literals at call sites).
REQUIRED_CAST_SITES: dict[str, str] = {
    # top-level
    "main": "main_dtype",
    "accumulation": "accumulation_dtype",
    # embeddings / head
    "embedding": "embedding_dtype",
    "lm_head": "lm_head_dtype",
    "logits": "logits_dtype",
    # MoE
    "router": "router_dtype",
    "shared_expert": "shared_expert_dtype",
    "expert": "expert_dtype",
    "expert_weight": "expert_weight_dtype",
    "expert_activation": "expert_activation_dtype",
    "expert_scale": "expert_scale_dtype",
    "expert_accumulation": "expert_accumulation_dtype",
    # Engram (~W4)
    "engram_table": "engram_table_dtype",
    "engram": "engram_dtype",
    "engram_accumulation": "engram_accumulation_dtype",
    # MLA / dense attention
    "mla": "mla_dtype",
    "mla_accumulation": "mla_accumulation_dtype",
    "attention": "attention_dtype",
    "attention_accumulation": "attention_accumulation_dtype",
    # indexer / compressor
    "indexer": "indexer_dtype",
    "indexer_accumulation": "indexer_accumulation_dtype",
    # dense MLP
    "dense": "dense_dtype",
    # caches
    "kv_cache": "kv_cache_dtype",
    "indexer_cache": "indexer_cache_dtype",
}

# The process-wide authoritative policy every module imports.
ASCEND_DEEPSEEKV41_DTYPE_POLICY = DeepseekV41DtypePolicy.for_310p()

__all__ = [
    "ASCEND_DEEPSEEKV41_DTYPE_POLICY",
    "REQUIRED_CAST_SITES",
    "DeepseekV41DtypePolicy",
]
