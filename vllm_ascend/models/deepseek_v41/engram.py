# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend DeepSeek V4.1 Engram host-table lookup for the 310P W2 path (E2.3).

DeepSeek V4.1 carries two Engram n-gram sub-blocks on the backbone layers listed
in ``engram_layer_ids`` (``[1, 14]``). Each writes an n-gram hash lookup into the
hyper-connection residual stream, gated by how well the looked-up key matches the
stream. This module ports the *formulas* of the fork
``vllm/models/deepseek_v41/common/engram.py`` (``EngramLayout``,
``compute_hash_multipliers``, ``_hash_ids_kernel``, ``_engram_lookup_kernel``,
``_fused_engram_post_wkv_kernel``) and ``nvidia/engram.py`` as **pure torch /
numpy** -- never the accelerator kernels, which the 310P lane must avoid.

Three stages, matching the fork pipeline and the E0.4 reference
(``tests/ut/deepseek_w2/reference/engram_reference.py``):

1. **Hash** -- :func:`compute_engram_hashes` walks the newest
   ``engram_max_ngram_size`` predecessors of every position, accumulates a
   rolling XOR of ``compressed_id * multiplier``, and emits one prime-bucketed
   hash id per (n-gram size, head). Exact integer arithmetic, so it matches the
   reference bit for bit.
2. **Gather** -- the two Engram tables are stored **~W4**: signed 4-bit row
   codes packed two-per-byte into ``uint8`` (:data:`engram_table_dtype`) plus a
   per-block ``float32`` scale, dequantized on gather. Storage and the batched,
   single-shared-copy row gather reuse the Qwen host-table machinery
   (:class:`~vllm_ascend.models.qwen4_exp.ngram_embedding.AscendPLEEmbeddingMethod`
   dual transport: pinned-UVA + ``/dev/shm`` mmap), wrapped here for the packed
   ``uint8`` storage DeepSeek needs. The optional async, de-duplicating prefetch
   reuses :class:`~vllm_ascend.models.qwen4_exp.ple_prefetch.AscendRowPrefetcher`
   unchanged.
3. **Projection** -- :func:`engram_gate_project` runs ``wkv`` over the gathered
   rows to produce one key per hyper-connection copy plus a shared value, then a
   normalized signed-sqrt sigmoid gate scores the residual stream against the key
   and writes ``hidden + gate * value`` back.

Every dtype reads from :data:`ASCEND_DEEPSEEKV41_DTYPE_POLICY` (packed ``uint8``
table, ``float16`` gather/compute output, ``float32`` post-mix accumulation); no
dtype literal is spelled at a compute site.

Host-byte accounting goes through
:class:`~vllm_ascend.observability.deepseek_w2_mem_accounting.DeepSeekW2MemoryAccountant`
under ``ENGRAM_HOST``: like the Qwen PLE table each Engram table is ONE shared
logical host copy, recorded identically on every rank and never multiplied by
``world_size``.

Scope: this module owns the hash + packed gather + gated projection and exposes a
clean ``forward`` / ``lookup`` surface. Wiring the two sub-blocks into
``model.py``'s ``_inject_engram`` hook (token map, lookback windows, slot cache)
is E4.1; this module does not edit ``model.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from vllm_ascend.models.qwen4_exp.ngram_embedding import (
    OS_TRANSFER_RESERVE_BYTES,
    AscendPLEPinnedHostEmbeddingMethod,
    AscendPLESharedMmapEmbeddingMethod,
    AscendPLETransport,
    PLETransportUnavailableError,
)
from vllm_ascend.observability.deepseek_w2_mem_accounting import (
    DeepSeekW2MemComponent,
    DeepSeekW2MemoryAccountant,
)

from .dtype_policy import ASCEND_DEEPSEEKV41_DTYPE_POLICY, DeepseekV41DtypePolicy

# --------------------------------------------------------------------------- #
# Pinned DeepSeek V4.1 Engram geometry (config defaults).
# --------------------------------------------------------------------------- #

# The two backbone layers that carry an Engram sub-block.
ENGRAM_LAYER_IDS: tuple[int, ...] = (1, 14)
# Longest n-gram hashed at each position (2-gram .. max).
ENGRAM_MAX_NGRAM_SIZE: int = 4
# Heads each n-gram size is split over.
ENGRAM_N_HEADS: int = 8
# Row width of a single hashed head (the gather table's embedding dim).
ENGRAM_HEAD_DIM: int = 256
# Compressed vocabulary the hash multipliers are derived from.
ENGRAM_COMPRESSED_VOCAB_SIZE: int = 99092
# Per-block quantization group along the row for the ~W4 block scale.
ENGRAM_BLOCK_SIZE: int = 32

# Cache value for tokens that take no part in an n-gram (image spans); matches
# the fork and the E0.4 reference ``DEAD_ID``.
DEAD_ID: int = -1
# Per-layer RNG stride from the fork's ``compute_hash_multipliers``.
_ENGRAM_LAYER_PRIME: int = 10007
# ue8m0 stores a power-of-two scale as its float32 exponent byte (bias 127).
_E8M0_BIAS: int = 127
# Magnitude floor before the projection sigmoid (fork ``Engram.clamp_value``).
_GATE_CLAMP: float = 1e-6
# Signed 4-bit code grid: values pack two-per-byte in two's complement.
_W4_NIBBLE_MASK: int = 0xF
_W4_SIGN_BIT: int = 8
_W4_MODULUS: int = 16


# --------------------------------------------------------------------------- #
# Stage 1: n-gram hashing (fork-faithful pure-torch/numpy port).
# --------------------------------------------------------------------------- #


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for ``n < 2**32`` (matches the fork)."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """Smallest prime above ``start`` not already handed out."""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, compressed_vocab_size: int
) -> torch.Tensor:
    """One odd, overflow-safe multiplier per (layer, lookback), per-layer RNG.

    Reproduces the fork exactly: ``np.random.default_rng(10007 * layer_id)``
    draws ``max_ngram_size`` values in ``[0, bound)`` and stores ``2v + 1``.
    """
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(_ENGRAM_LAYER_PRIME * layer_id)
        values = generator.integers(low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


@dataclass
class EngramHashLayout:
    """Prime-bucket layout + multipliers for the Engram hash tables.

    A position hashes as ``max_ngram_size - 1`` n-grams (2-gram .. max), each
    split over ``n_heads`` heads; every (n-gram size, head) pair owns a disjoint
    prime-sized bucket range, primes drawn in order and never reused. Mirrors the
    fork ``EngramLayout`` and the E0.4 reference layout so the hash ids agree bit
    for bit.
    """

    layer_ids: tuple[int, ...]
    max_ngram_size: int
    n_heads: int
    engram_vocab_size: int
    compressed_vocab_size: int
    pad_id: int
    head_dim: int = ENGRAM_HEAD_DIM
    primes: torch.Tensor = field(init=False)
    offsets: torch.Tensor = field(init=False)
    multipliers: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        if self.max_ngram_size < 2:
            raise ValueError("max_ngram_size must be >= 2")
        self.layer_ids = tuple(self.layer_ids)
        self.n_hash_cols = (self.max_ngram_size - 1) * self.n_heads
        primes: list[list[int]] = []
        seen: set[int] = set()
        for _ in self.layer_ids:
            flat: list[int] = []
            for _ in range(self.max_ngram_size - 1):
                current = self.engram_vocab_size - 1
                for _ in range(self.n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    flat.append(current)
            primes.append(flat)
        self.primes = torch.tensor(primes, dtype=torch.int64)  # [L, n_hash_cols]
        offsets = [np.cumsum([0, *row[:-1]]) for row in primes]
        self.offsets = torch.tensor(np.array(offsets), dtype=torch.int64)
        self.multipliers = compute_hash_multipliers(self.layer_ids, self.max_ngram_size, self.compressed_vocab_size)

    def num_embeddings(self, layer_hash_index: int) -> int:
        """Row count of one layer's gather table: the sum of its head primes."""
        return int(self.primes[layer_hash_index].sum().item())


def compute_engram_hashes(
    compressed_ids: torch.Tensor,
    layout: EngramHashLayout,
    dead_mask: torch.Tensor | None = None,
    history: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorized Engram hash ids for one contiguous segment.

    Args:
        compressed_ids: ``[T]`` compressed token ids (post token-map).
        layout: prime / multiplier layout.
        dead_mask: ``[T]`` bool; ``True`` marks a token that blocks the walk.
        history: ``[H]`` compressed ids preceding the segment; predecessors that
            reach before the segment read history, then block.

    Returns:
        ``[T, L, n_hash_cols]`` int64 hash ids, each in its column's prime bucket
        ``[offset, offset + prime)``.
    """
    compressed_ids = compressed_ids.reshape(-1).long()
    seq_len = compressed_ids.shape[0]
    num_layers = len(layout.layer_ids)
    max_ngram = layout.max_ngram_size
    n_heads = layout.n_heads
    if dead_mask is None:
        dead_mask = torch.zeros(seq_len, dtype=torch.bool)
    if history is None:
        history = torch.empty(0, dtype=torch.long)
    hist_len = history.shape[0]
    context = torch.cat([history.long(), compressed_ids])

    output = torch.empty(seq_len, num_layers, layout.n_hash_cols, dtype=torch.int64)
    for layer in range(num_layers):
        mult = layout.multipliers[layer]
        rolling = torch.zeros(seq_len, dtype=torch.int64)
        blocked = torch.zeros(seq_len, dtype=torch.bool)
        token_pos = torch.arange(seq_len)
        for shift in range(max_ngram):
            src_pos = token_pos - shift
            ctx_idx = src_pos + hist_len
            in_range = ctx_idx >= 0
            gathered = context[ctx_idx.clamp_min(0)]
            src_dead = torch.zeros(seq_len, dtype=torch.bool)
            seg_ok = src_pos >= 0
            src_dead[seg_ok] = dead_mask[src_pos[seg_ok]]
            source = torch.where(in_range & ~src_dead, gathered, torch.full_like(gathered, DEAD_ID))
            blocked = blocked | (src_pos < 0) | (~in_range) | (source == DEAD_ID)
            value = torch.where(blocked, torch.full_like(source, layout.pad_id), source)
            rolling = torch.bitwise_xor(rolling, value * mult[shift])
            if shift > 0:
                for head in range(n_heads):
                    col = (shift - 1) * n_heads + head
                    prime = int(layout.primes[layer, col].item())
                    offset = int(layout.offsets[layer, col].item())
                    output[:, layer, col] = torch.remainder(rolling, prime) + offset
    return output


# --------------------------------------------------------------------------- #
# Stage 2: ~W4 packed table (4-bit codes + fp32 block scale) and row gather.
# --------------------------------------------------------------------------- #


def e8m0_to_fp32_scale(exponent_bytes: torch.Tensor) -> torch.Tensor:
    """Decode ue8m0 exponent bytes to exact float32 power-of-two block scales.

    A ue8m0 byte is a float32 exponent field (bias 127), so ``2 ** (byte - 127)``
    is representable exactly in float32. Lets a checkpoint's ue8m0 scales feed the
    fp32 block-scale storage without loss (parity with the E0.4 reference).
    """
    return torch.pow(
        torch.tensor(2.0, dtype=torch.float32),
        exponent_bytes.to(torch.int64) - _E8M0_BIAS,
    )


def pack_w4_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack signed 4-bit row codes (``[-8, 7]``) two-per-byte into ``uint8``.

    Args:
        codes: ``[..., dim]`` integer codes, ``dim`` even. Even columns land in
            the low nibble, odd columns in the high nibble (two's complement).

    Returns:
        ``[..., dim // 2]`` ``uint8`` packed codes.
    """
    if codes.shape[-1] % 2:
        raise ValueError(f"W4 row dim must be even to pack two codes per byte, got {codes.shape[-1]}")
    codes = codes.to(torch.int64)
    nibbles = (codes & _W4_NIBBLE_MASK).to(torch.int64)
    low = nibbles[..., 0::2]
    high = nibbles[..., 1::2]
    return (low | (high << 4)).to(torch.uint8)


def unpack_w4_codes(packed: torch.Tensor, dim: int) -> torch.Tensor:
    """Unpack ``uint8`` two-per-byte nibbles back to signed int codes ``[-8, 7]``.

    Inverse of :func:`pack_w4_codes`. Returns ``[..., dim]`` int64 codes.
    """
    packed = packed.to(torch.int64)
    low = packed & _W4_NIBBLE_MASK
    high = (packed >> 4) & _W4_NIBBLE_MASK
    interleaved = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)
    codes = interleaved[..., :dim]
    # Sign-extend the two's-complement nibble: values >= 8 map to the negatives.
    return torch.where(codes >= _W4_SIGN_BIT, codes - _W4_MODULUS, codes)


def dequantize_w4_rows(
    packed: torch.Tensor,
    scale: torch.Tensor,
    dim: int,
    block_size: int = ENGRAM_BLOCK_SIZE,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize packed ~W4 rows to ``compute_dtype``.

    Args:
        packed: ``[..., dim // 2]`` ``uint8`` packed codes.
        scale: ``[..., dim // block_size]`` per-block scale.
        dim: unpacked row width.
        block_size: dim block owning one scale.
        compute_dtype: output / arithmetic dtype.

    Returns:
        ``[..., dim]`` dequantized rows.
    """
    codes = unpack_w4_codes(packed, dim).to(compute_dtype)
    scale = scale.to(compute_dtype).repeat_interleave(block_size, dim=-1)[..., :dim]
    return codes * scale


def engram_gather_packed(
    hash_ids: torch.Tensor,
    packed_codes: torch.Tensor,
    scale: torch.Tensor,
    dim: int,
    block_size: int = ENGRAM_BLOCK_SIZE,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Gather + dequantize one ~W4 table row per hash id.

    Args:
        hash_ids: ``[T, n_hash_cols]`` row indices.
        packed_codes: ``[num_embeddings, dim // 2]`` ``uint8`` packed rows.
        scale: ``[num_embeddings, dim // block_size]`` block scale.
        dim: unpacked row width.

    Returns:
        ``[T, n_hash_cols, dim]`` dequantized rows.
    """
    ids = hash_ids.long()
    rows = dequantize_w4_rows(packed_codes[ids], scale[ids], dim, block_size, compute_dtype)
    return rows


# --------------------------------------------------------------------------- #
# Stage 3: gated projection (fork ``_fused_engram_post_wkv_kernel`` port).
# --------------------------------------------------------------------------- #


def engram_gate_project(
    hidden_states: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    clamp_value: float = _GATE_CLAMP,
    token_mask: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Normalized signed-sqrt sigmoid gate + residual add.

    Args:
        hidden_states: ``[T, hc_mult, dim]`` residual stream copies.
        kv: ``[T, (hc_mult + 1) * dim]``: ``hc_mult`` keys then one shared value.
        q_weight, k_weight: ``[hc_mult, dim]`` per-copy gate weights.
        eps: RMSNorm epsilon.
        clamp_value: magnitude floor before the sigmoid.
        token_mask: ``[T]`` bool; ``False`` shuts the gate (pass-through).
        compute_dtype: arithmetic dtype (defaults to a float promotion of the
            input, so float16 inputs accumulate in float32 while a float64 parity
            harness stays in float64).

    Returns:
        ``[T, hc_mult, dim]`` output in ``compute_dtype``.
    """
    if compute_dtype is None:
        compute_dtype = torch.float64 if hidden_states.dtype == torch.float64 else torch.float32
    hidden_states = hidden_states.to(compute_dtype)
    kv = kv.to(compute_dtype)
    q_weight = q_weight.to(compute_dtype)
    k_weight = k_weight.to(compute_dtype)
    num_tokens, hc_mult, dim = hidden_states.shape
    keys = kv[:, : hc_mult * dim].view(num_tokens, hc_mult, dim)
    value = kv[:, hc_mult * dim :]  # [T, dim] shared across copies

    hidden_rms = torch.rsqrt(hidden_states.square().mean(dim=-1) + eps)
    key_rms = torch.rsqrt(keys.square().mean(dim=-1) + eps)
    dot = (hidden_states * q_weight * k_weight * keys).sum(dim=-1)
    dot = dot * hidden_rms * key_rms * (dim**-0.5)
    gate_input = torch.sqrt(torch.clamp(dot.abs(), min=clamp_value))
    gate_input = torch.where(dot < 0.0, -gate_input, gate_input)
    gate = torch.sigmoid(gate_input)
    if token_mask is not None:
        gate = torch.where(token_mask.bool().unsqueeze(-1), gate, torch.zeros_like(gate))
    return hidden_states + gate.unsqueeze(-1) * value.unsqueeze(1)


def engram_forward_packed(
    hidden_states: torch.Tensor,
    hash_ids: torch.Tensor,
    packed_codes: torch.Tensor,
    scale: torch.Tensor,
    wkv_weight: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    *,
    dim: int,
    block_size: int = ENGRAM_BLOCK_SIZE,
    clamp_value: float = _GATE_CLAMP,
    token_mask: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """End-to-end Engram: packed gather -> ``wkv`` -> gated projection.

    Functional twin of :meth:`AscendDeepseekV41Engram.forward` that takes the
    ``wkv`` weight explicitly, for direct parity against the E0.4 reference.
    """
    if compute_dtype is None:
        compute_dtype = torch.float64 if hidden_states.dtype == torch.float64 else torch.float32
    rows = engram_gather_packed(hash_ids, packed_codes, scale, dim, block_size, compute_dtype)
    kv = rows.reshape(rows.shape[0], -1) @ wkv_weight.to(compute_dtype).t()
    return engram_gate_project(hidden_states, kv, q_weight, k_weight, eps, clamp_value, token_mask, compute_dtype)


# --------------------------------------------------------------------------- #
# Host-resident ~W4 table storage (reuses the Qwen dual-transport machinery).
# --------------------------------------------------------------------------- #


class _RawDtypeShim:
    """Minimal ``dtype_policy`` stand-in that pins a raw storage dtype.

    The Qwen host-table base derives its table dtype from
    ``dtype_policy.cast_site("ngram_embedding")`` (a float16 embedding). DeepSeek
    stores raw packed ``uint8`` codes and ``float32`` block scales instead, so
    the wrappers below feed this shim to force the storage dtype without editing
    the Qwen module.
    """

    def __init__(self, dtype: torch.dtype) -> None:
        self._dtype = dtype

    def cast_site(self, name: str) -> torch.dtype:
        return self._dtype


def _raw_itemsize(dtype: torch.dtype) -> int:
    """Byte width of a raw storage dtype (float *or* integer)."""
    if dtype.is_floating_point:
        return torch.finfo(dtype).bits // 8
    return torch.iinfo(dtype).bits // 8


class _EngramRawMmapTable(AscendPLESharedMmapEmbeddingMethod):
    """Qwen ``/dev/shm`` shared-mmap transport specialized to a raw storage dtype.

    Inherits the single-shared-copy mmap ownership, the mismatched-size guard and
    the batched ``gather_rows``; overrides only the dtype source and ``itemsize``
    so a packed ``uint8`` (or ``float32`` scale) table is stored verbatim.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, *, storage_dtype: torch.dtype, **kwargs: object
    ) -> None:
        self._storage_dtype = storage_dtype
        super().__init__(num_embeddings, embedding_dim, dtype_policy=_RawDtypeShim(storage_dtype), **kwargs)  # type: ignore[arg-type]

    @property
    def itemsize(self) -> int:
        return _raw_itemsize(self._storage_dtype)


class _EngramRawPinnedTable(AscendPLEPinnedHostEmbeddingMethod):
    """Qwen pinned-UVA transport specialized to a raw storage dtype (D1 path)."""

    def __init__(
        self, num_embeddings: int, embedding_dim: int, *, storage_dtype: torch.dtype, **kwargs: object
    ) -> None:
        self._storage_dtype = storage_dtype
        super().__init__(num_embeddings, embedding_dim, dtype_policy=_RawDtypeShim(storage_dtype), **kwargs)  # type: ignore[arg-type]

    @property
    def itemsize(self) -> int:
        return _raw_itemsize(self._storage_dtype)


def _build_raw_host_table(
    num_embeddings: int,
    embedding_dim: int,
    *,
    storage_dtype: torch.dtype,
    transport: AscendPLETransport,
    shm_path: str | None,
    table_source: object,
    world_size: int,
    host_total_bytes: int | None,
    reserve_bytes: int,
    prefix: str,
):
    """Build one raw host table, mirroring ``create_ple_embedding_method``.

    Prefers pinned-UVA when available (AUTO / explicit) and falls back to the
    shared mmap transport otherwise, so the host-only dev path always lands on
    ``/dev/shm`` while the device path (D1) uses pinned UVA.
    """
    mmap_kwargs = dict(
        storage_dtype=storage_dtype,
        shm_path=shm_path,
        create=True,
        table_source=table_source,
        world_size=world_size,
        host_total_bytes=host_total_bytes,
        reserve_bytes=reserve_bytes,
        prefix=prefix,
    )

    def _build_mmap() -> _EngramRawMmapTable:
        return _EngramRawMmapTable(num_embeddings, embedding_dim, **mmap_kwargs)  # type: ignore[arg-type]

    if transport == AscendPLETransport.SHARED_MMAP:
        return _build_mmap()

    want_pinned = transport in (AscendPLETransport.AUTO, AscendPLETransport.PINNED_UVA)
    if want_pinned and _EngramRawPinnedTable.is_available():
        try:
            return _EngramRawPinnedTable(
                num_embeddings,
                embedding_dim,
                storage_dtype=storage_dtype,
                table_source=table_source,
                world_size=world_size,
                host_total_bytes=host_total_bytes,
                reserve_bytes=reserve_bytes,
                prefix=prefix,
            )
        except PLETransportUnavailableError:
            return _build_mmap()
    return _build_mmap()


class DeepseekEngramLayerTable:
    """One Engram layer's ~W4 host table: packed codes + fp32 block scale.

    Owns two single-shared-copy host tables (a ``uint8`` packed-code table and a
    ``float32`` block-scale table) built on the Qwen dual transport, and exposes a
    batched, duck-typed :meth:`gather_rows` compatible with
    :class:`~vllm_ascend.models.qwen4_exp.ple_prefetch.AscendRowPrefetcher`
    (``embedding_dim`` / ``itemsize`` / ``dtype`` / ``gather_rows``). The two
    tables together are the layer's single logical ~W4 copy in host RAM.
    """

    def __init__(
        self,
        *,
        layer_id: int,
        layer_hash_index: int,
        num_embeddings: int,
        head_dim: int = ENGRAM_HEAD_DIM,
        block_size: int = ENGRAM_BLOCK_SIZE,
        dtype_policy: DeepseekV41DtypePolicy = ASCEND_DEEPSEEKV41_DTYPE_POLICY,
        transport: AscendPLETransport = AscendPLETransport.AUTO,
        shm_dir: str | None = None,
        code_source: object = None,
        scale_source: object = None,
        world_size: int = 1,
        host_total_bytes: int | None = None,
        reserve_bytes: int = OS_TRANSFER_RESERVE_BYTES,
        prefix: str = "",
    ) -> None:
        if head_dim % 2:
            raise ValueError(f"engram head_dim must be even for ~W4 packing, got {head_dim}")
        if head_dim % block_size:
            raise ValueError(f"engram head_dim ({head_dim}) must be divisible by block_size ({block_size})")
        self.layer_id = int(layer_id)
        self.layer_hash_index = int(layer_hash_index)
        self.num_embeddings = int(num_embeddings)
        self.head_dim = int(head_dim)
        self.block_size = int(block_size)
        self.dtype_policy = dtype_policy
        # Storage dtypes and the dequantized-output dtype all read from the policy.
        self._code_dtype = dtype_policy.cast_site("engram_table")  # packed uint8
        self._scale_dtype = dtype_policy.cast_site("main")  # fp32 block scale storage
        self.dtype = dtype_policy.cast_site("engram")  # fp16 gather output
        self._accum_dtype = dtype_policy.cast_site("engram_accumulation")  # fp32

        packed_dim = head_dim // 2
        num_blocks = head_dim // block_size
        base_shm = shm_dir if shm_dir is not None else None

        def _shm(kind: str) -> str | None:
            name = f"vllm_ascend_engram_{prefix or 'default'}_L{layer_id}_{kind}.bin"
            return os.path.join(base_shm, name) if base_shm is not None else None

        common = dict(
            transport=transport,
            world_size=world_size,
            host_total_bytes=host_total_bytes,
            reserve_bytes=reserve_bytes,
        )
        self.codes = _build_raw_host_table(
            num_embeddings,
            packed_dim,
            storage_dtype=self._code_dtype,
            shm_path=_shm("codes"),
            table_source=code_source,
            prefix=f"{prefix}_L{layer_id}_codes",
            **common,  # type: ignore[arg-type]
        )
        self.scales = _build_raw_host_table(
            num_embeddings,
            num_blocks,
            storage_dtype=self._scale_dtype,
            shm_path=_shm("scale"),
            table_source=scale_source,
            prefix=f"{prefix}_L{layer_id}_scale",
            **common,  # type: ignore[arg-type]
        )

    # -- prefetcher-compatible surface ------------------------------------- #

    @property
    def embedding_dim(self) -> int:
        """Dequantized row width (what a gather returns), for the prefetcher."""
        return self.head_dim

    @property
    def itemsize(self) -> int:
        return _raw_itemsize(self.dtype)

    @property
    def physical_bytes(self) -> int:
        """Host bytes of this layer's ~W4 copy: packed codes + block scales."""
        return self.codes.physical_bytes + self.scales.physical_bytes

    def gather_rows(self, ids: torch.Tensor, compute_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Batched gather + dequant: flat ``ids -> [num_ids, head_dim]``.

        One batched fetch from each shared table (never per row), then unpack +
        block-scale. ``compute_dtype`` defaults to the policy gather dtype;
        parity harnesses pass ``float64``.
        """
        ids = ids.reshape(-1).long()
        out_dtype = compute_dtype if compute_dtype is not None else self.dtype
        if ids.numel() == 0:
            return torch.empty((0, self.head_dim), dtype=out_dtype)
        packed = self.codes.gather_rows(ids)
        scale = self.scales.gather_rows(ids)
        return dequantize_w4_rows(packed, scale, self.head_dim, self.block_size, out_dtype)

    def close(self) -> None:
        self.codes.close()
        self.scales.close()

    def __enter__(self) -> DeepseekEngramLayerTable:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class DeepseekEngramHostTables:
    """The two Engram layer tables (``engram_layer_ids``) as one shared host copy.

    Records host bytes ONCE under ``ENGRAM_HOST``: every rank reports the same
    total and the accountant's :meth:`host_table_bytes` returns the single shared
    value (never ``x world_size``), exactly like the Qwen PLE table.
    """

    def __init__(self, tables: list[DeepseekEngramLayerTable]) -> None:
        self.tables = tables
        self._by_hash_index = {t.layer_hash_index: t for t in tables}

    def __getitem__(self, layer_hash_index: int) -> DeepseekEngramLayerTable:
        return self._by_hash_index[layer_hash_index]

    def __len__(self) -> int:
        return len(self.tables)

    @property
    def physical_bytes(self) -> int:
        return sum(t.physical_bytes for t in self.tables)

    def record_host_bytes(self, accountant: DeepSeekW2MemoryAccountant, rank: int) -> None:
        """Record the shared Engram host copy as ``ENGRAM_HOST`` on ``rank``."""
        accountant.rank_report(rank).add(DeepSeekW2MemComponent.ENGRAM_HOST, self.physical_bytes)

    def close(self) -> None:
        for table in self.tables:
            table.close()


def create_engram_host_tables(
    layout: EngramHashLayout,
    *,
    head_dim: int = ENGRAM_HEAD_DIM,
    block_size: int = ENGRAM_BLOCK_SIZE,
    dtype_policy: DeepseekV41DtypePolicy = ASCEND_DEEPSEEKV41_DTYPE_POLICY,
    transport: AscendPLETransport = AscendPLETransport.AUTO,
    shm_dir: str | None = None,
    code_sources: list[object] | None = None,
    scale_sources: list[object] | None = None,
    world_size: int = 1,
    host_total_bytes: int | None = None,
    reserve_bytes: int = OS_TRANSFER_RESERVE_BYTES,
    prefix: str = "",
) -> DeepseekEngramHostTables:
    """Build both Engram layer tables from a hash layout (dual transport)."""
    tables: list[DeepseekEngramLayerTable] = []
    for hash_index, layer_id in enumerate(layout.layer_ids):
        tables.append(
            DeepseekEngramLayerTable(
                layer_id=layer_id,
                layer_hash_index=hash_index,
                num_embeddings=layout.num_embeddings(hash_index),
                head_dim=head_dim,
                block_size=block_size,
                dtype_policy=dtype_policy,
                transport=transport,
                shm_dir=shm_dir,
                code_source=None if code_sources is None else code_sources[hash_index],
                scale_source=None if scale_sources is None else scale_sources[hash_index],
                world_size=world_size,
                host_total_bytes=host_total_bytes,
                reserve_bytes=reserve_bytes,
                prefix=prefix,
            )
        )
    return DeepseekEngramHostTables(tables)


# --------------------------------------------------------------------------- #
# nn.Module surface for E4.1 (``_inject_engram``).
# --------------------------------------------------------------------------- #


def _layout_from_config(config: object) -> EngramHashLayout:
    """Build the hash layout from a text config, defaulting to pinned geometry."""
    layer_ids = tuple(getattr(config, "engram_layer_ids", None) or ENGRAM_LAYER_IDS)
    max_ngram = int(getattr(config, "engram_max_ngram_size", ENGRAM_MAX_NGRAM_SIZE))
    n_heads = int(getattr(config, "engram_n_heads", ENGRAM_N_HEADS))
    head_dim = int(getattr(config, "engram_head_dim", ENGRAM_HEAD_DIM))
    compressed = int(getattr(config, "engram_compressed_vocab_size", ENGRAM_COMPRESSED_VOCAB_SIZE))
    # ``engram_vocab_size`` seeds the prime search; the fork stores it on config.
    engram_vocab_size = int(getattr(config, "engram_vocab_size", compressed))
    pad_token_id = int(getattr(config, "engram_pad_token_id", 0))
    return EngramHashLayout(
        layer_ids=layer_ids,
        max_ngram_size=max_ngram,
        n_heads=n_heads,
        engram_vocab_size=engram_vocab_size,
        compressed_vocab_size=compressed,
        pad_id=pad_token_id,
        head_dim=head_dim,
    )


class DeepseekEngramHasher(nn.Module):
    """Stateless n-gram hasher wrapping :func:`compute_engram_hashes`.

    Exposes the fork ``NgramHashState`` role for E4.1: given a segment of
    compressed ids (and optional dead mask / history), produce
    ``[T, num_layers, n_hash_cols]`` hash ids. The runner-side token map,
    lookback windows and slot cache are E4.1's to assemble; this module owns the
    hashing math only.
    """

    def __init__(self, layout: EngramHashLayout) -> None:
        super().__init__()
        self.layout = layout
        self.register_buffer("primes", layout.primes, persistent=False)
        self.register_buffer("offsets", layout.offsets, persistent=False)
        self.register_buffer("multipliers", layout.multipliers, persistent=False)

    @property
    def n_hash_cols(self) -> int:
        return self.layout.n_hash_cols

    def forward(
        self,
        compressed_ids: torch.Tensor,
        dead_mask: torch.Tensor | None = None,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return compute_engram_hashes(compressed_ids, self.layout, dead_mask, history)


class AscendDeepseekV41Engram(nn.Module):
    """One DeepSeek V4.1 Engram sub-block (host ~W4 gather + gated projection).

    Mirrors the fork ``Engram`` surface so E4.1's ``_inject_engram`` can wire it
    into the residual stream at an ``engram_layer_ids`` layer with a stable call
    site: ``forward(hidden_states, hash_ids, token_mask)``. Row lookup is the
    single shared host ~W4 table; ``wkv`` / gate run in pure torch (no kernels).
    """

    def __init__(
        self,
        *,
        config: object,
        host_table: DeepseekEngramLayerTable,
        layer_hash_index: int,
        layout: EngramHashLayout | None = None,
        dtype_policy: DeepseekV41DtypePolicy = ASCEND_DEEPSEEKV41_DTYPE_POLICY,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_hash_index = int(layer_hash_index)
        self.dtype_policy = dtype_policy
        self.host_table = host_table
        self.layout = layout if layout is not None else _layout_from_config(config)

        self.dim = int(config.hidden_size)
        self.hc_mult = int(config.hc_mult)
        self.eps = float(config.rms_norm_eps)
        self.clamp_value = _GATE_CLAMP
        self.head_dim = self.layout.head_dim
        self.n_hash_cols = self.layout.n_hash_cols

        self._dtype = dtype_policy.cast_site("engram")  # fp16 compute surface
        self._accum_dtype = dtype_policy.cast_site("engram_accumulation")  # fp32

        # ``wkv`` maps the flattened gathered rows to hc_mult keys + one value.
        self.wkv = nn.Linear(
            self.n_hash_cols * self.head_dim,
            self.dim * (self.hc_mult + 1),
            bias=False,
            dtype=self._dtype,
        )
        self.q_weight = nn.Parameter(torch.empty(self.hc_mult, self.dim, dtype=self._dtype), requires_grad=False)
        self.k_weight = nn.Parameter(torch.empty(self.hc_mult, self.dim, dtype=self._dtype), requires_grad=False)

    def embed(self, hash_ids: torch.Tensor, compute_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Gather + dequant rows for a hash-id block -> ``[T, n_hash_cols, head_dim]``."""
        if hash_ids.ndim != 2 or hash_ids.shape[1] != self.n_hash_cols:
            raise ValueError(f"hash_ids must be [T, {self.n_hash_cols}], got {tuple(hash_ids.shape)}")
        rows = self.host_table.gather_rows(hash_ids.reshape(-1), compute_dtype=compute_dtype)
        return rows.reshape(hash_ids.shape[0], self.n_hash_cols, self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write the gated n-gram lookup into the residual stream.

        Args:
            hidden_states: ``[T, hc_mult, dim]`` residual stream copies.
            hash_ids: ``[T, n_hash_cols]`` this layer's hash ids.
            token_mask: ``[T]`` bool; ``False`` shuts the gate (pass-through).

        Returns:
            ``[T, hc_mult, dim]`` in the Engram compute dtype.
        """
        num_tokens, hc_mult, dim = hidden_states.shape
        if hc_mult != self.hc_mult or dim != self.dim:
            raise ValueError(
                f"hidden_states {tuple(hidden_states.shape)} does not match (*, {self.hc_mult}, {self.dim})"
            )
        if num_tokens == 0:
            return torch.empty_like(hidden_states)
        rows = self.embed(hash_ids, compute_dtype=self._dtype)
        kv = self.wkv(rows.reshape(num_tokens, -1))
        out = engram_gate_project(
            hidden_states,
            kv,
            self.q_weight,
            self.k_weight,
            self.eps,
            self.clamp_value,
            token_mask,
            compute_dtype=self._accum_dtype,
        )
        return out.to(hidden_states.dtype)


__all__ = [
    "ENGRAM_LAYER_IDS",
    "ENGRAM_MAX_NGRAM_SIZE",
    "ENGRAM_N_HEADS",
    "ENGRAM_HEAD_DIM",
    "ENGRAM_COMPRESSED_VOCAB_SIZE",
    "ENGRAM_BLOCK_SIZE",
    "DEAD_ID",
    "find_next_prime",
    "compute_hash_multipliers",
    "EngramHashLayout",
    "compute_engram_hashes",
    "e8m0_to_fp32_scale",
    "pack_w4_codes",
    "unpack_w4_codes",
    "dequantize_w4_rows",
    "engram_gather_packed",
    "engram_gate_project",
    "engram_forward_packed",
    "DeepseekEngramLayerTable",
    "DeepseekEngramHostTables",
    "create_engram_host_tables",
    "DeepseekEngramHasher",
    "AscendDeepseekV41Engram",
]
