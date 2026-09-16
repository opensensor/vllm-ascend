# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 MLA **latent** KV-cache specs + per-layer plan for Ascend 310P
(plan E2.2 -- ADAPT of shipped V4; Triton-free, host-safe).

This module is intentionally *host-safe*: it imports only ``torch``, the vLLM
base KV-cache specs, ``get_dtype_size`` / ``cdiv``, the authoritative
:mod:`deepseek_v41.dtype_policy`, and the reusable Qwen4Exp KV plumbing. It
pulls **no** ``torch_npu`` / Triton / CUDA / ``FusedMoEFactory`` module, so it
materializes and unit-tests on the 310P host lane (no NPU, no Triton kernels).

What is stored (DeepSeek Multi-head Latent Attention)
-----------------------------------------------------
DeepSeek MLA does *not* store per-head K and V. Each attention layer writes a
single **compressed latent** per token -- the ``kv_lora_rank`` (512) latent
concatenated with the **decoupled-RoPE key** (``qk_rope_head_dim`` 64) -- giving
one ``num_kv_heads=1`` cache of ``head_size = 512 + 64 = 576`` elements/token.
That is why the MLA latent KV is *much smaller* than a full-attention (per-head
K+V) cache: 1152 B/token/layer at fp16 versus ``2 * num_heads * qk_head_dim``
bytes for materialized MHA (see :func:`deepseekv41_kv_bytes_table`, the
``full_mha_reference`` block). The latent is shared by every query head, so it
is **TP-replicated** (not sharded) and, with PP=1, one chip holds the whole
per-request latent cache.

This maps cleanly onto vLLM's :class:`MLAAttentionSpec`, whose default
(non-fp8) ``real_page_size_bytes`` is
``storage_block_size * num_kv_heads * head_size * dtype_size`` -- a *single*
latent per token, no K/V doubling. The main MLA latent is uncompressed
(``compress_ratio == 1``); the sparse-attention **indexer's compressed history**
(E3.2) is a *second* MLA-shaped cache whose ``compress_ratio`` is the per-layer
value and whose ``head_size`` is ``index_head_dim`` (128).

V4.1 vs V4 per-layer plan
-------------------------
The shipped V4 used ``compress_ratios`` drawn from ``{0, 4, 128}`` (dense /
c4-indexer / c128-compressor). V4.1 draws its per-layer ratios from
``{0, 1, 2}`` instead (see the adaptation-seams doc and
``deepseek_v41.model.DEEPSEEKV41_COMPRESS_RATIOS``):

* ratio ``0`` -> **dense** MLA layer: MLA latent cache only, no indexer.
* ratio ``1`` -> **sparse** layer: MLA latent + indexer history at ratio 1
  (every index key stored; no CSA2 pooling).
* ratio ``2`` -> **sparse** layer: MLA latent + indexer history at ratio 2
  (CSA2 mean-pools 2 consecutive index keys into one stored latent).

:func:`build_deepseekv41_layer_plan` recomputes this per-layer classification
from a config's ``compress_ratios`` list; :func:`build_deepseekv41_kv_cache_config`
packages the resulting MLA-latent and indexer caches into hybrid KV-cache
groups. Both reuse the Qwen4Exp KV plumbing
(:mod:`vllm_ascend.models.qwen4_exp.kv_cache`) so E2.2 adds *specs + plan*, not
new packaging machinery.

Scope note: this module only *builds* specs/plans for E4.1 / the runner. It does
NOT edit ``deepseek_v41/model.py`` or any component module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MLAAttentionSpec

from vllm_ascend.models.qwen4_exp.kv_cache import (
    DEFAULT_ATTENTION_BLOCK_SIZE,
    _make_kv_cache_tensor,
    _mla_compression_kwarg,
    build_qwen4exp_kv_cache_groups,
    qsa_scheduler_block_size,
)

from .dtype_policy import ASCEND_DEEPSEEKV41_DTYPE_POLICY, DeepseekV41DtypePolicy

# =========================================================================
# DeepSeek V4.1 MLA / indexer geometry (config: DeepSeek-V4.1-Flash text_config)
# =========================================================================
# MLA compressed latent width + decoupled-RoPE key width. Mirrors
# tests/ut/deepseek_w2/reference/mla_reference.py (QK_ROPE_HEAD_DIM=64) so the
# spec byte-math and the E0.4 numerics reference agree on one geometry.
MLA_KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
# One MLA latent cache entry per token: the kv_lora latent + decoupled RoPE key.
MLA_LATENT_HEAD_DIM = MLA_KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
# The MLA latent is a single shared "head" attended by every query head.
MLA_LATENT_NUM_KV_HEADS = 1
# The main MLA latent is never compressed (only the indexer history is).
MLA_LATENT_COMPRESS_RATIO = 1

# Sparse-attention indexer history geometry (mirrors
# tests/ut/deepseek_w2/reference/indexer_reference.py INDEXER_HEAD_DIM=128).
INDEXER_HEAD_DIM = 128
INDEXER_NUM_KV_HEADS = 1

# V4.1 text-config geometry (see deepseek_v41/model.py + the plan).
DEEPSEEKV41_NUM_HIDDEN_LAYERS = 40
# The per-layer ``compress_ratios`` vocabulary. Shipped V4 used {0, 4, 128}.
DEEPSEEKV41_DISTINCT_COMPRESS_RATIOS = (0, 1, 2)
# Ratios that own an indexer / compressed history (shipped gate was ``== 4``).
DEEPSEEKV41_INDEXER_RATIOS = (1, 2)
# ratio 0 marks a dense MLA layer (no indexer history).
DEEPSEEKV41_DENSE_RATIO = 0

# PP=1: no pipeline sharding of the KV cache. The MLA latent has one KV head, so
# it is TP-replicated (not sharded); per chip == the full per-request latent.
DEEPSEEKV41_PIPELINE_PARALLEL_SIZE = 1

# 8K validation context, plus a documented long-context anchor for the C8 table.
CONTEXT_8K = 8192
CONTEXT_128K = 131072

# The C8 (8-bit) long-context indexer-history layout uses e4m3 (1 byte). The MLA
# latent itself always stays at the fp16 main dtype (precision-critical).
DEEPSEEKV41_C8_DTYPE = torch.float8_e4m3fn


# =========================================================================
# Per-layer plan
# =========================================================================
@dataclass(frozen=True)
class DeepseekV41LayerKVPlan:
    """The KV-cache plan for one decoder layer.

    ``compress_ratio`` is the layer's ``compress_ratios[layer_idx]`` value in
    ``{0, 1, 2}``. ``is_dense`` layers carry only the MLA latent cache;
    ``has_indexer`` layers additionally carry the indexer's compressed history.
    """

    layer_idx: int
    compress_ratio: int
    is_dense: bool
    has_indexer: bool
    mla_latent_spec: MLAAttentionSpec
    indexer_spec: MLAAttentionSpec | None


# =========================================================================
# Spec factories (exposed for E4.1 / the runner)
# =========================================================================
def make_mla_latent_spec(
    *,
    dtype_policy: DeepseekV41DtypePolicy = ASCEND_DEEPSEEKV41_DTYPE_POLICY,
    attention_block_size: int = DEFAULT_ATTENTION_BLOCK_SIZE,
    kv_lora_rank: int = MLA_KV_LORA_RANK,
    qk_rope_head_dim: int = QK_ROPE_HEAD_DIM,
) -> MLAAttentionSpec:
    """Materialize the DeepSeek MLA **latent** KV-cache spec for one layer.

    One ``num_kv_heads=1`` latent per token of width ``kv_lora_rank +
    qk_rope_head_dim`` (512 + 64 = 576), stored at the policy KV-cache dtype
    (fp16 on 310P). ``compress_ratio == 1``: the main latent is uncompressed, so
    ``storage_block_size == block_size`` and the page holds one latent per token.
    """
    head_size = kv_lora_rank + qk_rope_head_dim
    return MLAAttentionSpec(
        block_size=attention_block_size,
        num_kv_heads=MLA_LATENT_NUM_KV_HEADS,
        head_size=head_size,
        dtype=dtype_policy.kv_cache_dtype,
        **_mla_compression_kwarg(MLA_LATENT_COMPRESS_RATIO),
    )


def make_indexer_compressed_spec(
    compress_ratio: int,
    *,
    dtype_policy: DeepseekV41DtypePolicy = ASCEND_DEEPSEEKV41_DTYPE_POLICY,
    attention_block_size: int = DEFAULT_ATTENTION_BLOCK_SIZE,
    index_head_dim: int = INDEXER_HEAD_DIM,
    c8: bool = False,
) -> MLAAttentionSpec:
    """Materialize the sparse-attention **indexer** compressed-history spec.

    Mirrors the shipped ``AscendDeepseekV4IndexerCache.get_kv_cache_spec``:
    ``block_size = attention_block_size * compress_ratio`` so
    ``storage_block_size == attention_block_size`` holds one compressed latent
    per ``compress_ratio`` logical tokens (CSA2 pooling). ``c8`` selects the
    e4m3 8-bit long-context layout; otherwise the policy indexer-cache dtype
    (fp16). ``compress_ratio`` must be one of :data:`DEEPSEEKV41_INDEXER_RATIOS`.
    """
    if compress_ratio not in DEEPSEEKV41_INDEXER_RATIOS:
        raise ValueError(f"indexer compress_ratio must be one of {DEEPSEEKV41_INDEXER_RATIOS}; got {compress_ratio}")
    dtype = DEEPSEEKV41_C8_DTYPE if c8 else dtype_policy.indexer_cache_dtype
    return MLAAttentionSpec(
        block_size=attention_block_size * compress_ratio,
        num_kv_heads=INDEXER_NUM_KV_HEADS,
        head_size=index_head_dim,
        dtype=dtype,
        **_mla_compression_kwarg(compress_ratio),
    )


# =========================================================================
# Config helpers (host-safe; do NOT pull vllm_ascend.utils / torch_npu)
# =========================================================================
def _resolve_compress_ratios(config: Any) -> list[int]:
    """Return the per-layer ``compress_ratios`` list off a config.

    Accepts either a raw sequence of ints or a config object exposing a
    ``compress_ratios`` attribute (the shipped V4 path). Host-safe re-implement
    of ``get_dsv4_compress_ratio`` semantics (unspecified layers are dense).
    """
    if config is None:
        raise ValueError("a compress_ratios sequence or config is required")
    ratios = config if isinstance(config, (list, tuple)) else getattr(config, "compress_ratios", None)
    if ratios is None:
        raise ValueError("config exposes no compress_ratios")
    return [int(r) for r in ratios]


def _num_layers(config: Any, compress_ratios: list[int]) -> int:
    """Effective decoder-layer count (config override, else len(ratios))."""
    if not isinstance(config, (list, tuple)):
        n = getattr(config, "num_hidden_layers", None)
        if n is not None:
            return int(n)
    return len(compress_ratios)


def layer_compress_ratio(compress_ratios: list[int], layer_idx: int) -> int:
    """``compress_ratios[layer_idx]``, treating out-of-range layers as dense.

    Mirrors ``vllm_ascend.utils.get_dsv4_compress_ratio`` without importing the
    heavy (torch_npu-pulling) utils module.
    """
    if layer_idx < 0:
        raise ValueError("layer_idx must be non-negative")
    if layer_idx >= len(compress_ratios):
        return DEEPSEEKV41_DENSE_RATIO
    return compress_ratios[layer_idx]


def build_deepseekv41_layer_plan(
    config: Any,
    *,
    dtype_policy: DeepseekV41DtypePolicy = ASCEND_DEEPSEEKV41_DTYPE_POLICY,
    attention_block_size: int = DEFAULT_ATTENTION_BLOCK_SIZE,
    c8_indexer: bool = False,
) -> list[DeepseekV41LayerKVPlan]:
    """Recompute the V4.1 per-layer KV plan from ``compress_ratios`` in {0,1,2}.

    ``config`` is a raw ``compress_ratios`` sequence or an object exposing
    ``compress_ratios`` (+ optional ``num_hidden_layers``). Every layer gets an
    MLA latent spec; ratio-{1,2} layers additionally get an indexer spec at
    that ratio. Ratio 0 -> dense.
    """
    compress_ratios = _resolve_compress_ratios(config)
    num_layers = _num_layers(config, compress_ratios)
    mla_spec = make_mla_latent_spec(dtype_policy=dtype_policy, attention_block_size=attention_block_size)
    plan: list[DeepseekV41LayerKVPlan] = []
    for layer_idx in range(num_layers):
        ratio = layer_compress_ratio(compress_ratios, layer_idx)
        if ratio not in DEEPSEEKV41_DISTINCT_COMPRESS_RATIOS:
            raise ValueError(f"layer {layer_idx}: compress_ratio {ratio} not in {DEEPSEEKV41_DISTINCT_COMPRESS_RATIOS}")
        has_indexer = ratio in DEEPSEEKV41_INDEXER_RATIOS
        indexer_spec = (
            make_indexer_compressed_spec(
                ratio,
                dtype_policy=dtype_policy,
                attention_block_size=attention_block_size,
                c8=c8_indexer,
            )
            if has_indexer
            else None
        )
        plan.append(
            DeepseekV41LayerKVPlan(
                layer_idx=layer_idx,
                compress_ratio=ratio,
                is_dense=(ratio == DEEPSEEKV41_DENSE_RATIO),
                has_indexer=has_indexer,
                mla_latent_spec=mla_spec,
                indexer_spec=indexer_spec,
            )
        )
    return plan


# =========================================================================
# Hybrid group packaging (reuse Qwen4Exp KV plumbing)
# =========================================================================
def _plan_layer_specs(
    plan: list[DeepseekV41LayerKVPlan],
    *,
    mla_prefix: str = "mla",
    indexer_prefix: str = "indexer",
) -> tuple[dict[str, MLAAttentionSpec], dict[str, MLAAttentionSpec]]:
    """Split a plan into ``{layer_name: spec}`` dicts for MLA + indexer caches."""
    mla_layers: dict[str, MLAAttentionSpec] = {}
    indexer_layers: dict[str, MLAAttentionSpec] = {}
    for entry in plan:
        mla_layers[f"{mla_prefix}.{entry.layer_idx}"] = entry.mla_latent_spec
        if entry.indexer_spec is not None:
            indexer_layers[f"{indexer_prefix}.{entry.layer_idx}"] = entry.indexer_spec
    return mla_layers, indexer_layers


def build_deepseekv41_kv_cache_groups(
    plan: list[DeepseekV41LayerKVPlan],
) -> list[KVCacheGroupSpec]:
    """Package the plan's MLA-latent + indexer caches into hybrid groups.

    Delegates to :func:`build_qwen4exp_kv_cache_groups`: the MLA latent layers
    ride the ``full_attention`` slot (all identical -> one group) and the
    indexer histories ride the ``qsa_compressed`` slot (one group per distinct
    ratio). Layers sharing an identical spec are merged.
    """
    mla_layers, indexer_layers = _plan_layer_specs(plan)
    return build_qwen4exp_kv_cache_groups(
        full_attention_layers=mla_layers or None,
        qsa_compressed_layers=indexer_layers or None,
    )


def build_deepseekv41_kv_cache_config(
    plan: list[DeepseekV41LayerKVPlan],
    *,
    num_blocks: int = 1,
) -> KVCacheConfig:
    """Assemble a :class:`KVCacheConfig` for a V4.1 layer plan.

    Placeholder ``KVCacheTensor`` sizing (real strides are resolved by the
    worker via the version-tolerant :func:`_make_kv_cache_tensor`); the groups
    and block sizes are the meaningful, testable output for E4.1.
    """
    groups = build_deepseekv41_kv_cache_groups(plan)
    tensors = [
        _make_kv_cache_tensor(group.kv_cache_spec.page_size_bytes * num_blocks, list(group.layer_names))
        for group in groups
    ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )


def deepseekv41_scheduler_block_size(
    plan: list[DeepseekV41LayerKVPlan],
    *,
    attention_block_size: int = DEFAULT_ATTENTION_BLOCK_SIZE,
) -> int:
    """Scheduler block size = LCM of the attention block and every group block.

    Reuses :func:`qsa_scheduler_block_size`. With ratio-2 layers present this is
    ``lcm(128, 128, 256) == 256``.
    """
    group_blocks = [entry.indexer_spec.block_size for entry in plan if entry.indexer_spec is not None]
    return qsa_scheduler_block_size(attention_block_size, *group_blocks)


# =========================================================================
# Per-chip byte math (8K + long-context)
# =========================================================================
def deepseekv41_kv_bytes_per_chip(
    config: Any,
    *,
    max_model_len: int = CONTEXT_8K,
    dtype_policy: DeepseekV41DtypePolicy = ASCEND_DEEPSEEKV41_DTYPE_POLICY,
    attention_block_size: int = DEFAULT_ATTENTION_BLOCK_SIZE,
    c8_indexer: bool = False,
    num_attention_heads: int | None = None,
    qk_nope_head_dim: int | None = None,
) -> dict[str, Any]:
    """Exact per-chip MLA-latent + indexer KV footprint for one request.

    Values are computed from the materialized specs' ``page_size_bytes`` so the
    table validates spec materialization, not just hand arithmetic. PP=1 and the
    latent is TP-replicated, so ``per_chip == aggregate`` (no KV sharding).

    ``num_attention_heads`` + ``qk_nope_head_dim`` (optional) add a
    ``full_mha_reference`` row: the bytes a materialized per-head K+V cache would
    cost, to quantify how much smaller the MLA latent is.
    """
    plan = build_deepseekv41_layer_plan(
        config,
        dtype_policy=dtype_policy,
        attention_block_size=attention_block_size,
        c8_indexer=c8_indexer,
    )
    mla_spec = make_mla_latent_spec(dtype_policy=dtype_policy, attention_block_size=attention_block_size)
    mla_num_blocks = cdiv(max_model_len, attention_block_size)
    mla_per_layer_bytes = mla_num_blocks * mla_spec.page_size_bytes
    num_layers = len(plan)
    mla_total_bytes = mla_per_layer_bytes * num_layers

    indexer_bytes = 0
    indexer_layers = 0
    for entry in plan:
        if entry.indexer_spec is None:
            continue
        indexer_layers += 1
        spec = entry.indexer_spec
        # One page per attention block * ratio; storage rows compress the span.
        num_blocks = cdiv(max_model_len, spec.block_size)
        indexer_bytes += num_blocks * spec.page_size_bytes

    total_bytes = mla_total_bytes + indexer_bytes

    result: dict[str, Any] = {
        "max_model_len": max_model_len,
        "attention_block_size": attention_block_size,
        "pipeline_parallel_size": DEEPSEEKV41_PIPELINE_PARALLEL_SIZE,
        "num_layers": num_layers,
        "kv_lora_rank": MLA_KV_LORA_RANK,
        "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
        "mla_latent_head_dim": MLA_LATENT_HEAD_DIM,
        "mla_latent_dtype": str(mla_spec.dtype),
        "mla_latent_bytes_per_token": mla_spec.head_size * get_dtype_size(mla_spec.dtype),
        "mla_latent_page_bytes": mla_spec.page_size_bytes,
        "mla_num_blocks": mla_num_blocks,
        "mla_per_layer_bytes": mla_per_layer_bytes,
        "mla_total_bytes": mla_total_bytes,
        "indexer_layout": "C8" if c8_indexer else "BF16",
        "indexer_head_dim": INDEXER_HEAD_DIM,
        "indexer_layers": indexer_layers,
        "indexer_bytes": indexer_bytes,
        "total_bytes": total_bytes,
        "per_chip_bytes": total_bytes,  # PP=1, TP-replicated latent
    }

    # Optional comparison: a materialized per-head MHA K+V cache over the same
    # attention. MLA stores one 576-wide latent/token; MHA would store
    # 2 * num_heads * (qk_nope + qk_rope) per token -> orders of magnitude more.
    if num_attention_heads is not None and qk_nope_head_dim is not None:
        qk_head_dim = qk_nope_head_dim + QK_ROPE_HEAD_DIM
        mha_per_token = 2 * num_attention_heads * qk_head_dim * get_dtype_size(mla_spec.dtype)
        mha_total = mla_num_blocks * attention_block_size * mha_per_token * num_layers
        result["full_mha_reference"] = {
            "num_attention_heads": num_attention_heads,
            "qk_head_dim": qk_head_dim,
            "mha_bytes_per_token": mha_per_token,
            "mha_total_bytes": mha_total,
            "mla_shrink_factor": mha_total / mla_total_bytes if mla_total_bytes else 0.0,
        }
    return result


def deepseekv41_kv_bytes_table(
    config: Any,
    *,
    max_model_len: int = CONTEXT_8K,
    dtype_policy: DeepseekV41DtypePolicy = ASCEND_DEEPSEEKV41_DTYPE_POLICY,
    attention_block_size: int = DEFAULT_ATTENTION_BLOCK_SIZE,
    num_attention_heads: int | None = None,
    qk_nope_head_dim: int | None = None,
) -> dict[str, dict[str, Any]]:
    """BF16-vs-C8 (long-context indexer) per-chip byte table for ``config``.

    The MLA latent stays fp16 in both columns (precision-critical); only the
    indexer history switches to the e4m3 C8 layout for long context.
    """
    common = dict(
        max_model_len=max_model_len,
        dtype_policy=dtype_policy,
        attention_block_size=attention_block_size,
        num_attention_heads=num_attention_heads,
        qk_nope_head_dim=qk_nope_head_dim,
    )
    return {
        "BF16": deepseekv41_kv_bytes_per_chip(config, c8_indexer=False, **common),
        "C8": deepseekv41_kv_bytes_per_chip(config, c8_indexer=True, **common),
    }


__all__ = [
    "CONTEXT_128K",
    "CONTEXT_8K",
    "DEEPSEEKV41_C8_DTYPE",
    "DEEPSEEKV41_DENSE_RATIO",
    "DEEPSEEKV41_DISTINCT_COMPRESS_RATIOS",
    "DEEPSEEKV41_INDEXER_RATIOS",
    "DEEPSEEKV41_NUM_HIDDEN_LAYERS",
    "DEEPSEEKV41_PIPELINE_PARALLEL_SIZE",
    "INDEXER_HEAD_DIM",
    "MLA_KV_LORA_RANK",
    "MLA_LATENT_HEAD_DIM",
    "QK_ROPE_HEAD_DIM",
    "DeepseekV41LayerKVPlan",
    "build_deepseekv41_kv_cache_config",
    "build_deepseekv41_kv_cache_groups",
    "build_deepseekv41_layer_plan",
    "deepseekv41_kv_bytes_per_chip",
    "deepseekv41_kv_bytes_table",
    "deepseekv41_scheduler_block_size",
    "layer_compress_ratio",
    "make_indexer_compressed_spec",
    "make_mla_latent_spec",
]
