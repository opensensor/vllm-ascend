# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Authoritative dtype policy for the Ascend 310P Qwen4Exp path (PRD §5.3 / R4).

THIS FILE IS THE SINGLE SOURCE OF TRUTH for every dtype decision in the Ascend
Qwen4Exp implementation. Every downstream task (T1.3 PLE, T1.4 QSA, T1.5
indexer, T3.1 fused-expert load, T4.x attention, T5.x MoE, T6.1 sampler) and
the T0.5 / T8.3 byte-math harnesses MUST read dtypes from
:data:`ASCEND_QWEN4EXP_DTYPE_POLICY` (or a policy derived from a
``VllmConfig``) rather than spelling ``torch.float16`` / ``torch.bfloat16``
literals locally. The dtype-policy unit test enforces that no layer module
contains bare dtype literals.

Pinned assumptions (PRD R4)
---------------------------
* Main compute dtype is ``float16`` (the 310P has no native bfloat16 matmul
  path; the NVIDIA/AMD forks run bfloat16, we do NOT).
* Accumulation / reduction sites (router/softmax, attention scores, RMSNorm,
  gated-residual mixing, logits) run in ``float32`` for numerical stability at
  1M context, then cast back to ``float16``.
* The Mamba/GDN SSM *recurrent* state is kept in ``float32``; the conv state
  and KV/QSA caches ride the ``float16`` main dtype.

CANN VERIFICATION NOTE (hardware follow-up)
-------------------------------------------
The float16-main + fp32-accumulation split below is the *documented working
assumption*. It MUST be verified against the pinned CANN toolkit on real 310P
silicon (kernel dtype support for QSA, GDN short-conv, fused MoE, and the
indexer top-k). If a kernel only supports a narrower dtype set, override the
affected field here -- do NOT patch call sites.
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


@dataclass(frozen=True)
class Qwen4ExpDtypePolicy:
    """Immutable table pinning every Qwen4Exp dtype + cast site for 310P.

    All fields are ``torch.dtype``. Read them; never hardcode dtype literals in
    layer modules.
    """

    # --- top-level compute / accumulation ---------------------------------
    main_dtype: torch.dtype = _MAIN
    accumulation_dtype: torch.dtype = _ACCUM

    # --- embeddings / head / logits ---------------------------------------
    embedding_dtype: torch.dtype = _MAIN
    lm_head_dtype: torch.dtype = _MAIN
    logits_dtype: torch.dtype = _ACCUM

    # --- MoE (router runs fp32, experts fp16) -----------------------------
    router_dtype: torch.dtype = _ACCUM
    shared_expert_dtype: torch.dtype = _MAIN
    expert_dtype: torch.dtype = _MAIN

    # --- dense / QSA attention --------------------------------------------
    attention_dtype: torch.dtype = _MAIN
    attention_accumulation_dtype: torch.dtype = _ACCUM
    qsa_main_dtype: torch.dtype = _MAIN
    qsa_indexer_dtype: torch.dtype = _MAIN

    # --- caches -----------------------------------------------------------
    kv_cache_dtype: torch.dtype = _MAIN
    # SSM recurrent state stays fp32; conv state rides the main dtype.
    mamba_ssm_cache_dtype: torch.dtype = _ACCUM
    mamba_conv_cache_dtype: torch.dtype = _MAIN

    # --- PLE (parallel layer embedding) -----------------------------------
    ple_projection_dtype: torch.dtype = _MAIN
    ple_norm_accumulation_dtype: torch.dtype = _ACCUM

    # --- hyperconnection / gated residual ---------------------------------
    gated_residual_dtype: torch.dtype = _MAIN
    gated_residual_accumulation_dtype: torch.dtype = _ACCUM
    hyperconnection_params_dtype: torch.dtype = _MAIN

    # --- n-gram embedding -------------------------------------------------
    ngram_embedding_dtype: torch.dtype = _MAIN

    def cast_site(self, name: str) -> torch.dtype:
        """Resolve a PRD-named cast site to its pinned dtype.

        ``name`` is a key of :data:`REQUIRED_CAST_SITES`. This indirection lets
        call sites write ``policy.cast_site("router")`` and keeps the mapping
        auditable in one place.
        """
        try:
            attr = REQUIRED_CAST_SITES[name]
        except KeyError as exc:
            raise KeyError(f"unknown Qwen4Exp cast site {name!r}; known sites: {sorted(REQUIRED_CAST_SITES)}") from exc
        return getattr(self, attr)

    def as_dict(self) -> dict[str, torch.dtype]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def for_310p(cls) -> Qwen4ExpDtypePolicy:
        """Return the pinned 310P policy (float16 main + fp32 accumulation)."""
        return cls()

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig | None) -> Qwen4ExpDtypePolicy:
        """Derive a policy from a ``VllmConfig``.

        The 310P policy is authoritative and does not read the model dtype (we
        force float16). We only honour an explicit fp32 SSM-cache override via
        ``cache_config.mamba_ssm_cache_dtype`` when present; everything else is
        pinned. Downstream tasks that need config-driven KV-cache dtype should
        extend this hook rather than branching at call sites.
        """
        policy = cls.for_310p()
        if vllm_config is None:
            return policy
        cache_config = getattr(vllm_config, "cache_config", None)
        ssm = getattr(cache_config, "mamba_ssm_cache_dtype", None)
        if isinstance(ssm, torch.dtype) and ssm is not policy.mamba_ssm_cache_dtype:
            # Respect an explicit torch.dtype override only.
            return cls(**{**policy.as_dict(), "mamba_ssm_cache_dtype": ssm})
        return policy


# PRD-named cast sites -> policy field. These are the mandatory FP16/FP32 cast
# points every layer must route through the policy.
REQUIRED_CAST_SITES: dict[str, str] = {
    "attention": "attention_dtype",
    "attention_accumulation": "attention_accumulation_dtype",
    "router": "router_dtype",
    "shared_expert": "shared_expert_dtype",
    "expert": "expert_dtype",
    "ple_projection": "ple_projection_dtype",
    "ple_norm_accumulation": "ple_norm_accumulation_dtype",
    "gated_residual": "gated_residual_dtype",
    "gated_residual_accumulation": "gated_residual_accumulation_dtype",
    "lm_head": "lm_head_dtype",
    "logits": "logits_dtype",
    "qsa": "qsa_main_dtype",
    "qsa_indexer": "qsa_indexer_dtype",
    "kv_cache": "kv_cache_dtype",
    "mamba_ssm_cache": "mamba_ssm_cache_dtype",
    "mamba_conv_cache": "mamba_conv_cache_dtype",
    "embedding": "embedding_dtype",
    "ngram_embedding": "ngram_embedding_dtype",
    "hyperconnection": "hyperconnection_params_dtype",
}

# The process-wide authoritative policy every module imports.
ASCEND_QWEN4EXP_DTYPE_POLICY = Qwen4ExpDtypePolicy.for_310p()

__all__ = [
    "ASCEND_QWEN4EXP_DTYPE_POLICY",
    "REQUIRED_CAST_SITES",
    "Qwen4ExpDtypePolicy",
]
