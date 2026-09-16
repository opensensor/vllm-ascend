# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-PyTorch reference for Qwen4Exp n-gram embedding-id hashing.

Ports the *formulas* from:
  * ``vllm/models/qwen4_exp/nvidia/ngram_embedding.py`` -- the deterministic
    SplitMix64 multiplier construction, prime vocab layout, and the CPU
    ``compute_ngram_ids`` path (EOS boundary handling, history prefix).
  * ``vllm/models/qwen4_exp/nvidia/ops/ple.py`` (``_ple_ngram_ids_kernel``) --
    the canonical newest->oldest predecessor walk with EOS "crossing".

The n-gram id for a token at position ``t``, for order ``n`` (2..ngram_size) and
head ``h`` is::

    mixed = XOR_{s=0..n-1} token[t-s] * multiplier[s]
    id    = offset[h] + (mixed mod vocab_size[h])

where predecessors that fall before the previous EOS in the segment (or before
the sequence start) are replaced by ``eos_token_id``, and once a predecessor is
EOS every older predecessor is EOS too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PLE_LAYER_PRIME = 10007


def splitmix64(value: int) -> int:
    """Mix an integer into a deterministic unsigned 64-bit value."""
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime_64(value: int) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    """Return the ``count``-th prime strictly greater than ``start``."""
    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not _is_prime_64(candidate):
            candidate += 2
        prime = candidate
    return prime


def make_layer_multipliers(
    *,
    ngram_size: int,
    unigram_vocab_size: int,
    seed: int,
    ple_dense_layer_id: int,
) -> list[int]:
    """Build deterministic hash multipliers for one PLE layer."""
    max_multiplier = ((1 << 63) - 1) // unigram_vocab_size
    half_bound = max(1, max_multiplier // 2)
    base_seed = seed + _PLE_LAYER_PRIME * ple_dense_layer_id
    multipliers = []
    for index in range(ngram_size):
        value = base_seed + _SPLITMIX_GAMMA * (index + 1)
        multipliers.append(2 * (splitmix64(value) % half_bound) + 1)
    return multipliers


def make_vocab_layout(
    *,
    ngram_vocab_size_base: int,
    ngram_heads: int,
    ple_dense_layer_id: int,
) -> tuple[list[int], list[int], int]:
    """Build per-head vocab sizes, offsets, and the total row count."""
    sizes: list[int] = []
    offsets: list[int] = []
    offset = 0
    for local_head in range(ngram_heads):
        global_head = ple_dense_layer_id * ngram_heads + local_head
        size = _nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
        sizes.append(size)
        offsets.append(offset)
        offset += size
    return sizes, offsets, offset


@dataclass
class NGramHashConfig:
    """Deterministic per-layer n-gram hashing parameters."""

    ngram_size: int
    heads_per_ngram: int
    eos_token_id: int
    unigram_vocab_size: int
    ngram_vocab_size_base: int
    seed: int = 1234
    ple_dense_layer_id: int = 0
    multipliers: list[int] = field(init=False)
    sizes: list[int] = field(init=False)
    offsets: list[int] = field(init=False)

    def __post_init__(self) -> None:
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError("heads_per_ngram must be > 0")
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.multipliers = make_layer_multipliers(
            ngram_size=self.ngram_size,
            unigram_vocab_size=self.unigram_vocab_size,
            seed=self.seed,
            ple_dense_layer_id=self.ple_dense_layer_id,
        )
        self.sizes, self.offsets, self.total_vocab_size = make_vocab_layout(
            ngram_vocab_size_base=self.ngram_vocab_size_base,
            ngram_heads=self.ngram_heads,
            ple_dense_layer_id=self.ple_dense_layer_id,
        )


def _predecessors(
    tokens: torch.Tensor,
    history: torch.Tensor,
    cfg: NGramHashConfig,
) -> list[torch.Tensor]:
    """Return predecessor token tensors [shift] each ``[T]`` with EOS crossing.

    ``history`` is the ``ngram_size-1`` tokens preceding ``tokens`` (EOS-padded
    for a fresh segment). This models the ``ngram_context`` ring and the segment
    EOS boundary: the newest->oldest walk stops (fills EOS) at the first EOS.
    """
    seq_len = tokens.shape[0]
    context = torch.cat([history, tokens]).long()
    base = torch.arange(seq_len, dtype=torch.long) + (cfg.ngram_size - 1)
    preds = [tokens.long()]
    crossed = torch.zeros(seq_len, dtype=torch.bool)
    for shift in range(1, cfg.ngram_size):
        cand = context[base - shift].clone()
        cand = torch.where(crossed, torch.full_like(cand, cfg.eos_token_id), cand)
        crossed = crossed | (cand == cfg.eos_token_id)
        preds.append(cand)
    return preds


def compute_ngram_ids(
    tokens: torch.Tensor,
    cfg: NGramHashConfig,
    history: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorized reference for n-gram embedding ids.

    Args:
        tokens: ``[T]`` int token ids for one segment.
        cfg: hashing parameters.
        history: ``[ngram_size-1]`` preceding tokens (default: all EOS).

    Returns:
        ``[T, ngram_heads]`` int64 embedding ids.
    """
    tokens = tokens.reshape(-1).long()
    if history is None:
        history = torch.full((cfg.ngram_size - 1,), cfg.eos_token_id, dtype=torch.long)
    else:
        history = history.reshape(-1).long()
        if history.shape[0] != cfg.ngram_size - 1:
            raise ValueError("history must have length ngram_size-1")

    preds = _predecessors(tokens, history, cfg)
    mults = cfg.multipliers
    sizes = torch.tensor(cfg.sizes, dtype=torch.long)
    offsets = torch.tensor(cfg.offsets, dtype=torch.long)

    id_blocks = []
    for ngram in range(2, cfg.ngram_size + 1):
        mixed = preds[0] * mults[0]
        for s in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, preds[s] * mults[s])
        start = (ngram - 2) * cfg.heads_per_ngram
        end = start + cfg.heads_per_ngram
        head_sizes = sizes[start:end]
        head_offsets = offsets[start:end]
        ids = torch.remainder(mixed.unsqueeze(-1), head_sizes) + head_offsets
        id_blocks.append(ids)
    return torch.cat(id_blocks, dim=-1)


def bruteforce_ngram_ids(
    tokens: torch.Tensor,
    cfg: NGramHashConfig,
    history: torch.Tensor | None = None,
) -> torch.Tensor:
    """Independent per-token Python re-derivation (test oracle).

    Uses only Python ints and an explicit newest->oldest EOS walk, deriving each
    id from first principles without any vectorized tensor trick.
    """
    tok = tokens.reshape(-1).long().tolist()
    seq_len = len(tok)
    if history is None:
        hist = [cfg.eos_token_id] * (cfg.ngram_size - 1)
    else:
        hist = history.reshape(-1).long().tolist()
    context = hist + tok
    out = torch.empty(seq_len, cfg.ngram_heads, dtype=torch.long)
    for t in range(seq_len):
        # Build predecessor list with EOS crossing (newest -> oldest).
        preds = [tok[t]]
        crossed = False
        for shift in range(1, cfg.ngram_size):
            cand = context[t + (cfg.ngram_size - 1) - shift]
            if crossed:
                cand = cfg.eos_token_id
            if cand == cfg.eos_token_id:
                crossed = True
            preds.append(cand)
        for ngram in range(2, cfg.ngram_size + 1):
            mixed = preds[0] * cfg.multipliers[0]
            for s in range(1, ngram):
                mixed ^= preds[s] * cfg.multipliers[s]
            for local in range(cfg.heads_per_ngram):
                head = (ngram - 2) * cfg.heads_per_ngram + local
                out[t, head] = (mixed % cfg.sizes[head]) + cfg.offsets[head]
    return out


__all__ = [
    "NGramHashConfig",
    "splitmix64",
    "make_layer_multipliers",
    "make_vocab_layout",
    "compute_ngram_ids",
    "bruteforce_ngram_ids",
]
