# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-PyTorch reference for DeepSeek V4.1 Engram (hash -> gather -> projection).

Ports the *formulas* (never the Triton kernels) from the fork
``vllm/models/deepseek_v41/common/engram.py`` (``EngramLayout``,
``compute_hash_multipliers``, ``_hash_ids_kernel``, ``_engram_lookup_kernel``,
``_fused_engram_post_wkv_kernel``) and ``nvidia/engram.py``.

Three stages, matching the on-device pipeline:

1. **Hash** (``NgramHashState`` / ``_hash_ids_kernel``): for each position ``p``
   and each engram layer, walk the newest ``max_ngram_size`` predecessors,
   accumulate a rolling XOR of ``compressed_id * multiplier[layer, shift]``, and
   for every n-gram size ``2..max`` and head, emit ``rolling % prime + offset``.
   A predecessor before the sequence start or a "dead" token (image span) blocks
   the walk: from that shift on the value becomes ``pad_id``.

2. **Gather** (``ParallelEngramEmbedding`` / ``_engram_lookup_kernel``): each
   hash id gathers one quantized table row and applies its per-block ue8m0
   (power-of-two) scale. The reference stores the table as int8 codes plus
   power-of-two block scales — an exact, testable stand-in for fp8+e8m0fnu.

3. **Projection** (``Engram.forward`` / ``_fused_engram_post_wkv_kernel``): the
   gathered rows feed ``wkv`` to produce one key per hyper-connection copy plus
   a shared value; a normalized signed-sqrt sigmoid gate scores the residual
   stream against the key, and ``hidden + gate * value`` is written back.

The tokenizer-driven compressed-vocab map is *not* portable, so the hash stage
takes already-compressed token ids plus a dead mask directly; the multiplier and
prime construction (which the on-device tables depend on) are reproduced exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

# Cache value for tokens that take no part in an n-gram (image spans).
DEAD_ID = -1
# Per-layer RNG stride from the fork's compute_hash_multipliers.
_ENGRAM_LAYER_PRIME = 10007
# ue8m0 stores a power-of-two scale as its float32 exponent byte (bias 127).
_E8M0_BIAS = 127
# Magnitude floor before the projection sigmoid (Engram.clamp_value).
_GATE_CLAMP = 1e-6


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
    """Prime-bucket layout + multipliers for the engram hash tables.

    A position hashes as ``max_ngram_size - 1`` n-grams (2-gram .. max), each
    split over ``n_heads`` heads. Every (n-gram size, head) pair owns a disjoint
    prime-sized bucket range; primes are drawn in order and never reused.
    """

    layer_ids: tuple[int, ...]
    max_ngram_size: int
    n_heads: int
    engram_vocab_size: int
    compressed_vocab_size: int
    pad_id: int
    seed_reuse_across_layers: bool = True
    primes: torch.Tensor = field(init=False)
    offsets: torch.Tensor = field(init=False)
    multipliers: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        if self.max_ngram_size < 2:
            raise ValueError("max_ngram_size must be >= 2")
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


def compute_engram_hashes(
    compressed_ids: torch.Tensor,
    layout: EngramHashLayout,
    dead_mask: torch.Tensor | None = None,
    history: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorized engram hash ids for one segment.

    Args:
        compressed_ids: ``[T]`` compressed token ids (post token-map).
        layout: prime/multiplier layout.
        dead_mask: ``[T]`` bool; True marks a token that blocks the walk.
        history: ``[H]`` compressed ids preceding the segment (default: none);
            predecessors reaching before the segment read history, then block.

    Returns:
        ``[T, L, n_hash_cols]`` int64 hash ids.
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
            src_pos = token_pos - shift  # position in segment
            ctx_idx = src_pos + hist_len
            in_range = ctx_idx >= 0
            gathered = context[ctx_idx.clamp_min(0)]
            src_dead = torch.zeros(seq_len, dtype=torch.bool)
            # Dead only applies to in-segment predecessors we can index.
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


def bruteforce_engram_hashes(
    compressed_ids: torch.Tensor,
    layout: EngramHashLayout,
    dead_mask: torch.Tensor | None = None,
    history: torch.Tensor | None = None,
) -> torch.Tensor:
    """Independent per-token Python re-derivation (test oracle)."""
    ids = compressed_ids.reshape(-1).long().tolist()
    seq_len = len(ids)
    dead = [False] * seq_len if dead_mask is None else dead_mask.reshape(-1).bool().tolist()
    hist = [] if history is None else history.reshape(-1).long().tolist()
    context = hist + ids
    hist_len = len(hist)
    num_layers = len(layout.layer_ids)
    out = torch.empty(seq_len, num_layers, layout.n_hash_cols, dtype=torch.int64)
    for layer in range(num_layers):
        mult = layout.multipliers[layer].tolist()
        for t in range(seq_len):
            rolling = 0
            blocked = False
            for shift in range(layout.max_ngram_size):
                src_pos = t - shift
                ctx_idx = src_pos + hist_len
                if src_pos < 0 or ctx_idx < 0 or dead[src_pos]:
                    source = DEAD_ID
                else:
                    source = context[ctx_idx]
                if src_pos < 0 or source == DEAD_ID:
                    blocked = True
                value = layout.pad_id if blocked else source
                rolling ^= value * mult[shift]
                if shift > 0:
                    for head in range(layout.n_heads):
                        col = (shift - 1) * layout.n_heads + head
                        prime = int(layout.primes[layer, col].item())
                        offset = int(layout.offsets[layer, col].item())
                        out[t, layer, col] = rolling % prime + offset
    return out


# --- Stage 2: row gather ----------------------------------------------------


def e8m0_scale(exponent_bytes: torch.Tensor) -> torch.Tensor:
    """Decode ue8m0 exponent bytes to float64 power-of-two scales."""
    return torch.pow(
        torch.tensor(2.0, dtype=torch.float64),
        exponent_bytes.to(torch.int64) - _E8M0_BIAS,
    )


def engram_gather(
    hash_ids: torch.Tensor,
    table_codes: torch.Tensor,
    table_scale_exp: torch.Tensor,
    block_size: int = 32,
) -> torch.Tensor:
    """Gather one dequantized table row per hash id.

    Args:
        hash_ids: ``[T, n_hash_cols]`` int row indices.
        table_codes: ``[num_embeddings, dim]`` int8 row codes (fp8 stand-in).
        table_scale_exp: ``[num_embeddings, dim // block_size]`` uint8 ue8m0
            exponent bytes (power-of-two scales).
        block_size: dim block for the per-block scale.

    Returns:
        ``[T, n_hash_cols, dim]`` float64 dequantized rows.
    """
    dim = table_codes.shape[1]
    codes = table_codes[hash_ids].double()  # [T, C, dim]
    scale = e8m0_scale(table_scale_exp[hash_ids])  # [T, C, dim//bs]
    scale = scale.repeat_interleave(block_size, dim=-1)[..., :dim]
    return codes * scale


def bruteforce_engram_gather(
    hash_ids: torch.Tensor,
    table_codes: torch.Tensor,
    table_scale_exp: torch.Tensor,
    block_size: int = 32,
) -> torch.Tensor:
    """Per-(token, head) loop gather (test oracle)."""
    num_tokens, n_cols = hash_ids.shape
    dim = table_codes.shape[1]
    out = torch.empty(num_tokens, n_cols, dim, dtype=torch.float64)
    for t in range(num_tokens):
        for c in range(n_cols):
            idx = int(hash_ids[t, c].item())
            row = table_codes[idx].double()
            for d in range(dim):
                exp = int(table_scale_exp[idx, d // block_size].item())
                out[t, c, d] = row[d] * (2.0 ** (exp - _E8M0_BIAS))
    return out


# --- Stage 3: gated projection ---------------------------------------------


def engram_gate_project(
    hidden_states: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    clamp_value: float = _GATE_CLAMP,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Signed-sqrt sigmoid gate + residual add (``_fused_engram_post_wkv_kernel``).

    Args:
        hidden_states: ``[T, hc_mult, dim]`` residual stream copies.
        kv: ``[T, (hc_mult + 1) * dim]``: ``hc_mult`` keys then one shared value.
        q_weight, k_weight: ``[hc_mult, dim]`` per-copy gate weights.
        eps: RMSNorm epsilon.
        clamp_value: magnitude floor before the sigmoid.
        token_mask: ``[T]`` bool; False shuts the gate (pass-through).

    Returns:
        ``[T, hc_mult, dim]`` float64 output.
    """
    hidden_states = hidden_states.double()
    kv = kv.double()
    q_weight = q_weight.double()
    k_weight = k_weight.double()
    num_tokens, hc_mult, dim = hidden_states.shape
    keys = kv[:, : hc_mult * dim].view(num_tokens, hc_mult, dim)
    value = kv[:, hc_mult * dim :]  # [T, dim] shared across copies

    hidden_rms = torch.rsqrt(hidden_states.square().mean(dim=-1) + eps)  # [T, hc]
    key_rms = torch.rsqrt(keys.square().mean(dim=-1) + eps)  # [T, hc]
    dot = (hidden_states * q_weight * k_weight * keys).sum(dim=-1)  # [T, hc]
    dot = dot * hidden_rms * key_rms * (dim**-0.5)
    gate_input = torch.sqrt(torch.clamp(dot.abs(), min=clamp_value))
    gate_input = torch.where(dot < 0.0, -gate_input, gate_input)
    gate = torch.sigmoid(gate_input)  # [T, hc]
    if token_mask is not None:
        gate = torch.where(token_mask.bool().unsqueeze(-1), gate, torch.zeros_like(gate))
    return hidden_states + gate.unsqueeze(-1) * value.unsqueeze(1)


def bruteforce_engram_gate_project(
    hidden_states: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    clamp_value: float = _GATE_CLAMP,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Explicit per-(token, copy) loop for the projection (test oracle)."""
    hidden_states = hidden_states.double()
    kv = kv.double()
    q_weight = q_weight.double()
    k_weight = k_weight.double()
    num_tokens, hc_mult, dim = hidden_states.shape
    out = torch.empty_like(hidden_states)
    value_all = kv[:, hc_mult * dim :]
    for t in range(num_tokens):
        active = True if token_mask is None else bool(token_mask[t].item())
        value = value_all[t]
        for c in range(hc_mult):
            hidden = hidden_states[t, c]
            key = kv[t, c * dim : (c + 1) * dim]
            h_rms = float(torch.rsqrt(hidden.square().mean() + eps))
            k_rms = float(torch.rsqrt(key.square().mean() + eps))
            dot = float((hidden * q_weight[c] * k_weight[c] * key).sum())
            dot = dot * h_rms * k_rms * (dim**-0.5)
            mag = math.sqrt(max(abs(dot), clamp_value))
            gate_input = -mag if dot < 0.0 else mag
            gate = 1.0 / (1.0 + math.exp(-gate_input))
            if not active:
                gate = 0.0
            out[t, c] = hidden + gate * value
    return out


def engram_forward(
    hidden_states: torch.Tensor,
    hash_ids: torch.Tensor,
    table_codes: torch.Tensor,
    table_scale_exp: torch.Tensor,
    wkv_weight: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    *,
    block_size: int = 32,
    clamp_value: float = _GATE_CLAMP,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """End-to-end Engram: gather -> wkv -> gated projection.

    Args:
        hidden_states: ``[T, hc_mult, dim]``.
        hash_ids: ``[T, n_hash_cols]`` (one layer's hash columns).
        table_codes, table_scale_exp: gather table (see :func:`engram_gather`).
        wkv_weight: ``[(hc_mult + 1) * dim, n_hash_cols * head_dim]`` linear.
        q_weight, k_weight: ``[hc_mult, dim]``.
        eps: RMSNorm epsilon.

    Returns:
        ``[T, hc_mult, dim]`` float64 output.
    """
    rows = engram_gather(hash_ids, table_codes, table_scale_exp, block_size)
    kv = rows.reshape(rows.shape[0], -1) @ wkv_weight.double().t()
    return engram_gate_project(hidden_states, kv, q_weight, k_weight, eps, clamp_value, token_mask)


__all__ = [
    "DEAD_ID",
    "find_next_prime",
    "compute_hash_multipliers",
    "EngramHashLayout",
    "compute_engram_hashes",
    "bruteforce_engram_hashes",
    "e8m0_scale",
    "engram_gather",
    "bruteforce_engram_gather",
    "engram_gate_project",
    "bruteforce_engram_gate_project",
    "engram_forward",
]
