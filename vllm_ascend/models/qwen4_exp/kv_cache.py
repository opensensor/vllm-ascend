# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp QSA KV-cache specs, byte-math and slot mapping for Ascend 310P
(plan T1.4).

This module is intentionally *host-safe*: it imports only ``torch``, the vLLM
base KV-cache specs, ``get_dtype_size`` and the authoritative
:mod:`dtype_policy`. It pulls **no** ``torch_npu`` / Triton / CUDA module, so it
materializes and unit-tests on the 310P host lane (no NPU, no Triton kernels).

Spec shapes mirror the vLLM fork's ``models/qwen4_exp/common/qsa_cache.py``:

* ``QSAKeyStateCache``      -> ``CircularBufferSpec`` (raw index-key ring)
* ``QSACompressedKeyCache`` -> ``MLAAttentionSpec(tokens_per_state=ratio)``

DEVIATION (documented, anticipated risk)
----------------------------------------
The pinned/installed vLLM lane on 310P (v0.22.0) does **not** expose
``CircularBufferSpec`` nor the newer ``num_states`` / ``tokens_per_state``
``AttentionSpec`` API. Two consequences, both handled here additively:

1. The QSA raw ring is materialized as :class:`AscendQSARawRingSpec`, a
   *key-only* circular buffer. The generic ``AttentionSpec`` byte-math doubles
   the page for a V tensor; the QSA ring stores keys only, so we override the
   page-size math (this is the "granularity/bytes calculation for 310P").
2. The compressed index uses ``MLAAttentionSpec`` with the version-appropriate
   compression field (``tokens_per_state`` on main, ``compress_ratio`` on the
   0.22 lane), selected by :func:`_mla_compression_kwarg`.

Ring capacity must divide the attention block size so the scheduler block size
is ``lcm(attention_block, ring_capacity, compressed_block)``; see
:func:`qsa_ring_capacity` / :func:`qsa_scheduler_block_size`. When the ring
capacity does not divide the attention block size the LCM raises the scheduler
block size instead (handled in ``patch_kv_cache_utils``).

The QSA side caches run *out of band* via :class:`AscendQSAStateBackend`
(backend name ``QWEN4_EXP_EXP_QSA_STATE``). On Ascend there is no Triton, so
metadata + slot mapping always use the pure-torch fallback ported below
(mirrors the fork's ``_build_qsa_metadata_torch``); never the CUDA Triton
kernel.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
)

from .dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY, Qwen4ExpDtypePolicy

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# --- QSA / indexer geometry (plan T6.1 / references) ----------------------
# These mirror tests/ut/qwen38_1m/reference/qsa_indexer_reference.py so the
# spec byte-math and the numerics reference agree on one geometry.
QSA_STATE_BACKEND_NAME = "QWEN4_EXP_EXP_QSA_STATE"
INDEXER_N_HEADS = 4
INDEXER_KV_HEADS = 1
INDEXER_HEAD_DIM = 128
INDEXER_COMPRESS_RATIO = 4
INDEXER_TOKEN_BUDGET = 2048
# block_topk = token budget / ratio (512 for the pinned geometry).
INDEXER_BLOCK_TOPK = INDEXER_TOKEN_BUDGET // INDEXER_COMPRESS_RATIO

# 310P attention paged-block size (see _310p worker kernel-block sizing).
DEFAULT_ATTENTION_BLOCK_SIZE = 128

# 1M long-context window: native YaRN window 262_144 * factor 4 (see
# _310p/worker/v2/rope.py QWEN4EXP_YARN_FACTOR).
QWEN4EXP_NATIVE_MAX_POSITION = 262_144
QWEN4EXP_MAX_MODEL_LEN_1M = QWEN4EXP_NATIVE_MAX_POSITION * 4  # 1_048_576

# The C8 (8-bit) compressed-index layout uses e4m3 (1 byte) — permitted by the
# QSA state backend's supported_kv_cache_dtypes. The raw ring stays 2-byte.
QSA_C8_DTYPE = torch.float8_e4m3fn

# --- Candidate A (plan T8.1): C8 *main* QSA K/V ---------------------------
# The main QSA K/V cache is the full-context key/value store that sparse
# attention reads (distinct from the compressed *indexer* history above). For
# the Candidate A C8 layout it is stored in **signed symmetric INT8** (1 byte),
# exactly half the fp16 main dtype (2 bytes). INT8 (not e4m3) is used here
# because the quant is a dynamic per-(token, head) affine-free grid computed at
# cache-write time — see :mod:`vllm_ascend.models.qwen4_exp.qsa_c8`.
QSA_C8_MAIN_DTYPE = torch.int8

# Main QSA sparse-attention geometry (mirrors
# tests/ut/qwen38_1m/reference/qsa_attention_reference.py): 2 KV heads,
# head dim 256. The exported model has 12 QSA layers (models/qwen4_exp/model.py).
QSA_MAIN_KV_HEADS = 2
QSA_MAIN_HEAD_DIM = 256
QWEN4EXP_NUM_QSA_LAYERS = 12
# PRD §6 footprint assumption: sequence-sharded across 4 ranks, no TP
# duplication of the main K/V.
QSA_SEQUENCE_SHARD_RANKS = 4

# Out-of-band caches address their own pages; PAD marks an unmapped slot.
_PAD_SLOT_ID = -1


# =========================================================================
# Spec classes
# =========================================================================
@dataclass(frozen=True, kw_only=True)
class AscendQSARawRingSpec(AttentionSpec):
    """Key-only circular buffer for one QSA layer's open compression group.

    Mirrors the fork ``CircularBufferSpec``: one physical block per request for
    the request's whole lifetime, ``block_size`` == ring capacity in tokens.
    The buffer holds *keys only*, so the page-size math must not double for a V
    tensor the way the generic ``AttentionSpec`` does.
    """

    @property
    def real_page_size_bytes(self) -> int:
        return self.block_size * self.num_kv_heads * self.head_size * get_dtype_size(self.dtype)

    @property
    def page_size_bytes(self) -> int:
        padded = getattr(self, "page_size_padded", None)
        if padded is not None:
            assert padded >= self.real_page_size_bytes
            return padded
        return self.real_page_size_bytes

    @property
    def prefix_cacheable(self) -> bool:
        return False

    @property
    def block_table_token_alignment(self) -> int | None:
        return None

    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        del vllm_config, max_len
        return 1

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # The ring occupies exactly one block per request for its lifetime.
        del vllm_config
        return self.page_size_bytes


# =========================================================================
# Capacity / block-size (LCM) helpers
# =========================================================================
def qsa_ring_capacity(
    compress_ratio: int = INDEXER_COMPRESS_RATIO,
    num_speculative_tokens: int = 0,
) -> int:
    """Ring capacity in tokens, rounded up to whole compression groups.

    Mirrors the fork: hold the open group's committed keys plus every row a
    speculative step stores before acceptance is known, rounded up to whole
    groups so the ring divides the attention block size.
    """
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    if num_speculative_tokens < 0:
        raise ValueError("num_speculative_tokens must be non-negative")
    span = compress_ratio + num_speculative_tokens
    return compress_ratio * cdiv(span, compress_ratio)


def qsa_scheduler_block_size(
    attention_block_size: int,
    *group_block_sizes: int,
) -> int:
    """Scheduler block size = LCM of the attention block and every group's
    block size (the QSA ring capacity and compressed block size join here)."""
    sizes = [attention_block_size, *[b for b in group_block_sizes if b]]
    return math.lcm(*sizes)


def _mla_compression_kwarg(compress_ratio: int) -> dict[str, int]:
    """Return the version-appropriate MLA compression kwarg.

    vLLM #51718 replaced ``MLAAttentionSpec.compress_ratio`` with
    ``AttentionSpec.tokens_per_state``. Both express how many logical tokens one
    stored state covers.
    """
    field_names = {f.name for f in dataclasses.fields(MLAAttentionSpec)}
    if "compress_ratio" in field_names:
        return {"compress_ratio": compress_ratio}
    return {"tokens_per_state": compress_ratio}


def _qsa_state_dtype(dtype_policy: Qwen4ExpDtypePolicy, *, c8: bool) -> torch.dtype:
    """Element dtype for a QSA state cache.

    The raw ring and (BF16 layout) compressed cache read their element dtype
    from the authoritative policy (``qsa_indexer_dtype`` -> fp16 on 310P, i.e.
    the same 2-byte width the NVIDIA/AMD forks give bf16). The C8 layout
    quantizes the *compressed* cache to e4m3 (1 byte).
    """
    if c8:
        return QSA_C8_DTYPE
    return dtype_policy.qsa_indexer_dtype


# =========================================================================
# Spec factories
# =========================================================================
def make_qsa_raw_ring_spec(
    *,
    dtype_policy: Qwen4ExpDtypePolicy = ASCEND_QWEN4EXP_DTYPE_POLICY,
    compress_ratio: int = INDEXER_COMPRESS_RATIO,
    num_speculative_tokens: int = 0,
    head_size: int = INDEXER_HEAD_DIM,
    num_kv_heads: int = INDEXER_KV_HEADS,
) -> AscendQSARawRingSpec:
    """Materialize the QSA raw index-key ring (``QSAKeyStateCache`` shape).

    The raw ring always uses the policy's 2-byte state dtype (never quantized):
    the QSA state backend only advertises bf16 for the raw side cache.
    """
    capacity = qsa_ring_capacity(compress_ratio, num_speculative_tokens)
    return AscendQSARawRingSpec(
        block_size=capacity,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=_qsa_state_dtype(dtype_policy, c8=False),
    )


def make_qsa_compressed_spec(
    *,
    dtype_policy: Qwen4ExpDtypePolicy = ASCEND_QWEN4EXP_DTYPE_POLICY,
    attention_block_size: int = DEFAULT_ATTENTION_BLOCK_SIZE,
    compress_ratio: int = INDEXER_COMPRESS_RATIO,
    head_size: int = INDEXER_HEAD_DIM,
    num_kv_heads: int = INDEXER_KV_HEADS,
    c8: bool = False,
) -> MLAAttentionSpec:
    """Materialize the QSA compressed index (``QSACompressedKeyCache`` shape).

    One stored row per complete compression group; its block table follows the
    main KV-cache lifecycle (``block_size`` == attention block size). ``c8``
    selects the e4m3 8-bit compressed layout.
    """
    if attention_block_size % compress_ratio:
        raise ValueError(
            f"attention block size {attention_block_size} must be divisible by the compression ratio {compress_ratio}"
        )
    return MLAAttentionSpec(
        block_size=attention_block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=_qsa_state_dtype(dtype_policy, c8=c8),
        **_mla_compression_kwarg(compress_ratio),
    )


# =========================================================================
# Candidate A (T8.1): main QSA K/V spec + byte-math (ADDITIVE hook)
# =========================================================================
def _qsa_main_kv_dtype(dtype_policy: Qwen4ExpDtypePolicy, *, c8: bool) -> torch.dtype:
    """Element dtype for the *main* QSA K/V cache.

    BF16 layout reads the authoritative ``qsa_main_dtype`` (fp16 on 310P,
    2 bytes); the Candidate A C8 layout stores signed INT8 (1 byte).
    """
    if c8:
        return QSA_C8_MAIN_DTYPE
    return dtype_policy.qsa_main_dtype


def make_qsa_main_kv_spec(
    *,
    dtype_policy: Qwen4ExpDtypePolicy = ASCEND_QWEN4EXP_DTYPE_POLICY,
    attention_block_size: int = DEFAULT_ATTENTION_BLOCK_SIZE,
    num_kv_heads: int = QSA_MAIN_KV_HEADS,
    head_size: int = QSA_MAIN_HEAD_DIM,
    c8: bool = False,
) -> FullAttentionSpec:
    """Materialize the main QSA K/V cache spec (full-context key/value store).

    ``c8`` selects the Candidate A signed-INT8 layout (1 byte/element); its page
    is exactly half the fp16 (``c8=False``) page. A ``FullAttentionSpec`` is used
    because the main K/V stores both K and V over the full context (unlike the
    key-only raw ring and the ratio-compressed indexer cache).
    """
    return FullAttentionSpec(
        block_size=attention_block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=_qsa_main_kv_dtype(dtype_policy, c8=c8),
    )


def qsa_main_kv_bytes_per_chip(
    *,
    max_model_len: int = QWEN4EXP_MAX_MODEL_LEN_1M,
    dtype_policy: Qwen4ExpDtypePolicy = ASCEND_QWEN4EXP_DTYPE_POLICY,
    attention_block_size: int = DEFAULT_ATTENTION_BLOCK_SIZE,
    num_kv_heads: int = QSA_MAIN_KV_HEADS,
    head_size: int = QSA_MAIN_HEAD_DIM,
    num_qsa_layers: int = QWEN4EXP_NUM_QSA_LAYERS,
    shard_ranks: int = QSA_SEQUENCE_SHARD_RANKS,
    c8: bool = False,
) -> dict[str, Any]:
    """Exact main-QSA-K/V footprint for one 1M request (BF16 or C8 layout).

    Values are computed from the materialized ``FullAttentionSpec``'s
    ``page_size_bytes`` (K+V) so the table validates spec materialization, not
    just hand arithmetic. ``per_chip_bytes`` divides the aggregate over
    ``shard_ranks`` sequence-sharded ranks (PRD §6: no TP duplication).

    PRD §6 reconciliation (1M, 12 QSA layers, block 128, 2 KV heads, dim 256):
      * BF16 page (K+V) = 2*128*2*256*2 = 262_144 B; per layer = 262_144 * 8_192
        = 2.00 GiB; aggregate = 24.00 GiB; per chip (/4) = 6.00 GiB -> matches
        the "Logical BF16 QSA K/V at 1M" row.
      * C8 (INT8) halves every figure: 1.00 GiB/layer, 12.00 GiB aggregate,
        3.00 GiB/chip sequence-sharded. (The PRD "8-bit ... 6.00 GiB/chip" row
        is the *TP4-duplicated* comparison; Candidate A's sequence-sharded C8 is
        the 3.00 GiB/chip win.)
    """
    if shard_ranks <= 0:
        raise ValueError("shard_ranks must be positive")
    spec = make_qsa_main_kv_spec(
        dtype_policy=dtype_policy,
        attention_block_size=attention_block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        c8=c8,
    )
    num_blocks = cdiv(max_model_len, attention_block_size)
    per_layer_bytes = num_blocks * spec.page_size_bytes
    aggregate_bytes = per_layer_bytes * num_qsa_layers
    return {
        "layout": "C8" if c8 else "BF16",
        "max_model_len": max_model_len,
        "attention_block_size": attention_block_size,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "num_qsa_layers": num_qsa_layers,
        "shard_ranks": shard_ranks,
        "state_dtype": str(_qsa_main_kv_dtype(dtype_policy, c8=c8)),
        "element_size_bytes": get_dtype_size(_qsa_main_kv_dtype(dtype_policy, c8=c8)),
        "page_bytes": spec.page_size_bytes,
        "num_blocks": num_blocks,
        "per_layer_bytes": per_layer_bytes,
        "aggregate_bytes": aggregate_bytes,
        "per_chip_bytes": aggregate_bytes // shard_ranks,
    }


def qsa_main_kv_bytes_table(
    *,
    max_model_len: int = QWEN4EXP_MAX_MODEL_LEN_1M,
    dtype_policy: Qwen4ExpDtypePolicy = ASCEND_QWEN4EXP_DTYPE_POLICY,
    attention_block_size: int = DEFAULT_ATTENTION_BLOCK_SIZE,
    num_kv_heads: int = QSA_MAIN_KV_HEADS,
    head_size: int = QSA_MAIN_HEAD_DIM,
    num_qsa_layers: int = QWEN4EXP_NUM_QSA_LAYERS,
    shard_ranks: int = QSA_SEQUENCE_SHARD_RANKS,
) -> dict[str, dict[str, Any]]:
    """The BF16-vs-C8 main-QSA-K/V byte table for ``max_model_len``."""
    common = dict(
        max_model_len=max_model_len,
        dtype_policy=dtype_policy,
        attention_block_size=attention_block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        num_qsa_layers=num_qsa_layers,
        shard_ranks=shard_ranks,
    )
    return {
        "BF16": qsa_main_kv_bytes_per_chip(c8=False, **common),
        "C8": qsa_main_kv_bytes_per_chip(c8=True, **common),
    }


# =========================================================================
# 1M byte-math (feeds T8.x)
# =========================================================================
def qsa_cache_bytes_per_chip(
    *,
    max_model_len: int = QWEN4EXP_MAX_MODEL_LEN_1M,
    dtype_policy: Qwen4ExpDtypePolicy = ASCEND_QWEN4EXP_DTYPE_POLICY,
    compress_ratio: int = INDEXER_COMPRESS_RATIO,
    num_qsa_layers: int = 1,
    attention_block_size: int = DEFAULT_ATTENTION_BLOCK_SIZE,
    num_speculative_tokens: int = 0,
    c8: bool = False,
) -> dict[str, Any]:
    """Exact QSA side-cache footprint for one request at ``max_model_len``.

    Returns a per-layer + total breakdown for the BF16 (``c8=False``) or C8
    (``c8=True``) layout. Values are computed from the materialized specs'
    ``page_size_bytes`` so the table validates spec materialization, not just
    hand arithmetic.
    """
    ring = make_qsa_raw_ring_spec(
        dtype_policy=dtype_policy,
        compress_ratio=compress_ratio,
        num_speculative_tokens=num_speculative_tokens,
    )
    compressed = make_qsa_compressed_spec(
        dtype_policy=dtype_policy,
        attention_block_size=attention_block_size,
        compress_ratio=compress_ratio,
        c8=c8,
    )
    # Compressed index follows the main KV lifecycle: one page per attention
    # block over the full context (its storage rows are ratio-compressed inside
    # each page via the MLA storage_block_size).
    compressed_num_blocks = cdiv(max_model_len, attention_block_size)
    compressed_bytes = compressed_num_blocks * compressed.page_size_bytes
    # The raw ring is one block per request for its whole lifetime.
    ring_bytes = ring.page_size_bytes
    per_layer_bytes = ring_bytes + compressed_bytes
    return {
        "layout": "C8" if c8 else "BF16",
        "max_model_len": max_model_len,
        "compress_ratio": compress_ratio,
        "attention_block_size": attention_block_size,
        "num_qsa_layers": num_qsa_layers,
        "state_dtype": str(_qsa_state_dtype(dtype_policy, c8=c8)),
        "element_size_bytes": get_dtype_size(_qsa_state_dtype(dtype_policy, c8=c8)),
        "ring_capacity": ring.block_size,
        "ring_page_bytes": ring_bytes,
        "compressed_storage_rows": max_model_len // compress_ratio,
        "compressed_page_bytes": compressed.page_size_bytes,
        "compressed_num_blocks": compressed_num_blocks,
        "compressed_bytes": compressed_bytes,
        "per_layer_bytes": per_layer_bytes,
        "total_bytes": per_layer_bytes * num_qsa_layers,
    }


def qsa_cache_bytes_table(
    *,
    max_model_len: int = QWEN4EXP_MAX_MODEL_LEN_1M,
    dtype_policy: Qwen4ExpDtypePolicy = ASCEND_QWEN4EXP_DTYPE_POLICY,
    compress_ratio: int = INDEXER_COMPRESS_RATIO,
    num_qsa_layers: int = 1,
    attention_block_size: int = DEFAULT_ATTENTION_BLOCK_SIZE,
    num_speculative_tokens: int = 0,
) -> dict[str, dict[str, Any]]:
    """The BF16-vs-C8 byte table for ``max_model_len`` (keyed by layout)."""
    common = dict(
        max_model_len=max_model_len,
        dtype_policy=dtype_policy,
        compress_ratio=compress_ratio,
        num_qsa_layers=num_qsa_layers,
        attention_block_size=attention_block_size,
        num_speculative_tokens=num_speculative_tokens,
    )
    return {
        "BF16": qsa_cache_bytes_per_chip(c8=False, **common),
        "C8": qsa_cache_bytes_per_chip(c8=True, **common),
    }


# =========================================================================
# Hybrid group packaging
# =========================================================================
def _make_kv_cache_tensor(size: int, layer_names: list[str]) -> KVCacheTensor:
    """Version-tolerant ``KVCacheTensor`` (0.22 uses ``shared_by``)."""
    field_names = {f.name for f in dataclasses.fields(KVCacheTensor)}
    if "shared_by" in field_names:
        return KVCacheTensor(size=size, shared_by=layer_names)
    return KVCacheTensor(
        size=size,
        layers=layer_names,
        layer_stride=size,
        block_stride=size,
    )


def build_qwen4exp_kv_cache_groups(
    *,
    full_attention_layers: dict[str, KVCacheSpec] | None = None,
    mamba_layers: dict[str, KVCacheSpec] | None = None,
    qsa_raw_ring_layers: dict[str, AscendQSARawRingSpec] | None = None,
    qsa_compressed_layers: dict[str, MLAAttentionSpec] | None = None,
) -> list[KVCacheGroupSpec]:
    """Package Qwen4Exp caches into hybrid KV-cache groups.

    One group per cache class, keeping every QSA side cache in its own group
    (its slot layout differs from full attention and GDN): full-attention
    groups, the GDN ``MambaSpec`` group, the QSA raw-ring group and the QSA
    compressed-index group. Layers within a class that share a spec are merged.
    """
    groups: list[KVCacheGroupSpec] = []

    def _add(layers: dict[str, KVCacheSpec] | None) -> None:
        if not layers:
            return
        # Merge layers that share an identical spec into one group.
        by_spec: dict[Any, list[str]] = {}
        for name, spec in layers.items():
            by_spec.setdefault(spec, []).append(name)
        for spec, names in by_spec.items():
            groups.append(KVCacheGroupSpec(layer_names=names, kv_cache_spec=spec))

    _add(full_attention_layers)
    _add(mamba_layers)
    _add(qsa_raw_ring_layers)
    _add(qsa_compressed_layers)
    return groups


def build_qwen4exp_kv_cache_config(
    *,
    full_attention_layers: dict[str, KVCacheSpec] | None = None,
    mamba_layers: dict[str, KVCacheSpec] | None = None,
    qsa_raw_ring_layers: dict[str, AscendQSARawRingSpec] | None = None,
    qsa_compressed_layers: dict[str, MLAAttentionSpec] | None = None,
    num_blocks: int = 1,
) -> KVCacheConfig:
    """Assemble a :class:`KVCacheConfig` for a Qwen4Exp spec set.

    Placeholder ``KVCacheTensor`` sizing (real strides are resolved by the
    worker); the groups and block sizes are the meaningful, testable output.
    """
    groups = build_qwen4exp_kv_cache_groups(
        full_attention_layers=full_attention_layers,
        mamba_layers=mamba_layers,
        qsa_raw_ring_layers=qsa_raw_ring_layers,
        qsa_compressed_layers=qsa_compressed_layers,
    )
    tensors: list[KVCacheTensor] = []
    for group in groups:
        page = group.kv_cache_spec.page_size_bytes
        tensors.append(_make_kv_cache_tensor(page * num_blocks, list(group.layer_names)))
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )


# =========================================================================
# QSA state backend (out-of-band, Triton-free on Ascend)
# =========================================================================
class AscendQSAStateBackend:
    """Key-only out-of-band QSA side-cache backend for Ascend 310P.

    Mirrors the fork ``QSAStateBackend`` surface (name, supported dtypes) but is
    a plain host-safe class: it never imports Triton/CUDA. On Ascend the QSA
    metadata + slot mapping always use the pure-torch path below.
    """

    # The raw ring is stored in the 2-byte state dtype only.
    supported_dtypes: list[torch.dtype] = [torch.bfloat16, torch.float16]
    # fp8 entries allow the optional e4m3 compressed indexer (C8) cache.
    supported_kv_cache_dtypes: list[str] = [
        "auto",
        "bfloat16",
        "float16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_name() -> str:
        return QSA_STATE_BACKEND_NAME

    @staticmethod
    def uses_triton() -> bool:
        # Ascend has no Triton QSA kernel; pages/metadata are built in torch.
        return False

    @staticmethod
    def get_metadata_builder():
        """Return the Triton-free metadata builder used on Ascend."""
        return build_qsa_metadata_torch


# =========================================================================
# Pure-torch slot mapping (ported from the fork's _build_qsa_metadata_torch
# path; NO Triton). These address QSA pages directly on device.
# =========================================================================
def canonical_qsa_rope_positions(positions: torch.Tensor) -> torch.Tensor:
    """Return exact per-token positions as ``[tokens, 1, 3]`` int64 rows."""
    if positions.ndim == 1:
        positions = positions.unsqueeze(0).expand(3, -1)
    elif positions.ndim != 2 or positions.shape[0] not in (1, 3):
        raise ValueError("QSA RoPE positions must be [tokens] or [1|3, tokens]")
    if positions.shape[0] == 1:
        positions = positions.expand(3, -1)
    return positions.transpose(0, 1).unsqueeze(1).to(torch.int64)


def _logical_to_physical_qsa_slots(
    block_table: torch.Tensor,
    request_indices: torch.Tensor,
    logical_positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    if block_size <= 0:
        raise ValueError("QSA cache block size must be positive")
    if block_table.ndim != 2:
        raise ValueError("QSA block table must be two-dimensional")
    if request_indices.shape != logical_positions.shape:
        request_indices = torch.broadcast_to(request_indices, logical_positions.shape)

    requests = request_indices.to(device=block_table.device, dtype=torch.long)
    positions = logical_positions.to(device=block_table.device, dtype=torch.long)
    valid = (requests >= 0) & (requests < block_table.shape[0]) & (positions >= 0)
    logical_blocks = torch.div(positions.clamp_min(0), block_size, rounding_mode="floor")
    valid &= logical_blocks < block_table.shape[1]
    safe_requests = requests.clamp(0, max(block_table.shape[0] - 1, 0))
    safe_blocks = logical_blocks.clamp(0, max(block_table.shape[1] - 1, 0))
    if not all(block_table.shape):
        return torch.full_like(positions, _PAD_SLOT_ID)
    physical_blocks = block_table[safe_requests, safe_blocks].long()
    valid &= physical_blocks >= 0
    slots = physical_blocks * block_size + positions.remainder(block_size)
    return torch.where(valid, slots, _PAD_SLOT_ID)


def circular_qsa_slot_mapping(
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    compressor_state_size: int,
    query_start_loc: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map each token to its fixed physical block as a circular token ring."""
    if compressor_state_size <= 0:
        raise ValueError("QSA circular buffer size must be positive")
    if block_table.ndim != 2:
        raise ValueError("QSA block table must be two-dimensional")

    requests = token_to_req.to(device=block_table.device, dtype=torch.long)
    positions = logical_positions.to(device=block_table.device, dtype=torch.long)
    if not all(block_table.shape):
        slots = torch.full_like(positions, _PAD_SLOT_ID)
    else:
        valid = (requests >= 0) & (requests < block_table.shape[0]) & (positions >= 0)
        safe_requests = requests.clamp(0, block_table.shape[0] - 1)
        physical_blocks = block_table[safe_requests, 0].long()
        valid &= physical_blocks >= 0
        slots = physical_blocks * compressor_state_size + positions.remainder(compressor_state_size)
        slots = torch.where(valid, slots, _PAD_SLOT_ID)

    if query_start_loc is not None:
        if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
            raise ValueError("QSA query starts must contain a terminal offset")
        query_start_loc = query_start_loc.to(block_table.device)
        num_requests = query_start_loc.shape[0] - 1
        safe_requests = requests.clamp(0, num_requests - 1)
        request_ends = query_start_loc.index_select(0, safe_requests + 1)
        rows = torch.arange(slots.numel(), device=slots.device)
        keep = (requests >= 0) & (requests < num_requests) & (rows + compressor_state_size >= request_ends)
        slots = torch.where(keep, slots, _PAD_SLOT_ID)

    slots = slots.to(torch.int64)
    if out is not None:
        out.fill_(_PAD_SLOT_ID)
        out[: slots.numel()].copy_(slots)
        return out[: slots.numel()]
    return slots


def compressed_qsa_slot_mapping(
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    storage_block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build boundary-only slots for the compressed (MLA) QSA cache."""
    if storage_block_size <= 0 or compress_ratio <= 0:
        raise ValueError("QSA block size and compression ratio must be positive")
    compressed_positions = torch.div(logical_positions.clamp_min(0), compress_ratio, rounding_mode="floor")
    slots = _logical_to_physical_qsa_slots(
        block_table,
        token_to_req,
        compressed_positions,
        storage_block_size,
    )
    valid = (logical_positions >= 0) & ((logical_positions + 1).remainder(compress_ratio) == 0)
    slots = torch.where(valid, slots, _PAD_SLOT_ID).to(torch.int64)
    if out is not None:
        out.fill_(_PAD_SLOT_ID)
        out[: slots.numel()].copy_(slots)
        return out[: slots.numel()]
    return slots


def build_qsa_metadata_torch(
    *,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    storage_block_size: int,
    compress_ratio: int,
    circular_buffer_size: int = 0,
    query_start_loc: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-torch QSA slot mapping (no Triton), selecting ring vs compressed.

    Thin adaptation of the fork's ``_build_qsa_metadata_torch`` slot-mapping
    branch: a ``circular_buffer_size > 0`` builds the raw-ring mapping, else the
    compressed boundary mapping. Kept tensor-in/tensor-out so it unit-tests on
    CPU without a full ``CommonAttentionMetadata``.
    """
    if circular_buffer_size > 0:
        return circular_qsa_slot_mapping(
            block_table,
            token_to_req,
            logical_positions,
            circular_buffer_size,
            query_start_loc=query_start_loc,
        )
    return compressed_qsa_slot_mapping(
        block_table,
        token_to_req,
        logical_positions,
        storage_block_size,
        compress_ratio,
    )


__all__ = [
    "AscendQSARawRingSpec",
    "AscendQSAStateBackend",
    "DEFAULT_ATTENTION_BLOCK_SIZE",
    "INDEXER_BLOCK_TOPK",
    "INDEXER_COMPRESS_RATIO",
    "INDEXER_HEAD_DIM",
    "INDEXER_KV_HEADS",
    "INDEXER_N_HEADS",
    "INDEXER_TOKEN_BUDGET",
    "QSA_C8_DTYPE",
    "QSA_C8_MAIN_DTYPE",
    "QSA_MAIN_HEAD_DIM",
    "QSA_MAIN_KV_HEADS",
    "QSA_SEQUENCE_SHARD_RANKS",
    "QSA_STATE_BACKEND_NAME",
    "QWEN4EXP_MAX_MODEL_LEN_1M",
    "QWEN4EXP_NATIVE_MAX_POSITION",
    "QWEN4EXP_NUM_QSA_LAYERS",
    "build_qsa_metadata_torch",
    "build_qwen4exp_kv_cache_config",
    "build_qwen4exp_kv_cache_groups",
    "canonical_qsa_rope_positions",
    "circular_qsa_slot_mapping",
    "compressed_qsa_slot_mapping",
    "make_qsa_compressed_spec",
    "make_qsa_main_kv_spec",
    "make_qsa_raw_ring_spec",
    "qsa_cache_bytes_per_chip",
    "qsa_cache_bytes_table",
    "qsa_main_kv_bytes_per_chip",
    "qsa_main_kv_bytes_table",
    "qsa_ring_capacity",
    "qsa_scheduler_block_size",
]
