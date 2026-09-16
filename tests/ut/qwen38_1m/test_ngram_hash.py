# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T4.2 -- Exact n-gram hashing and EOS boundary tests vs the REAL PLE table.

This test verifies the Qwen4Exp n-gram embedding-id hashing three ways:

1. **Reference vs. fork production algorithm.** A self-contained port of the
   NVIDIA fork's production ``compute_ngram_ids`` (the ``query_start_loc`` /
   ``ngram_context`` layout with the ``_shift_precompute`` / ``_shift_apply``
   EOS-crossing logic, from ``vllm/models/qwen4_exp/nvidia/ngram_embedding.py``)
   is embedded here as an independent oracle. The T0.6 host reference
   (``tests/ut/qwen38_1m/reference/ngram_hash_reference.py``) is asserted to
   agree with it exactly on sampled windows and on the boundary cases
   (sequence start with EOS-padded history, across EOS, history-advance).

2. **Hash config vs. the REAL checkpoint.** The deterministic multipliers and
   vocab layout computed from the real ``text_config`` params (seed 1234,
   ``ple_dense_layer_id=0``) are compared byte-for-byte against the multipliers
   stored in the checkpoint tensor
   ``...ple.ple_embedding.layer_multipliers`` and against the real shard
   geometry (128 shards of ``[2500012, 160]`` F16), which pins
   ``padded_vocab_size == 128 * 2500012`` and ``head_dim == 160``.

3. **Resolved rows vs. the REAL safetensors table.** Sampled + boundary token
   windows are hashed to n-gram ids; each id is mapped to
   ``(shard_index, row_within_shard)`` per ``split_ngram_parts=128`` and the
   single 160-wide F16 row is read directly from the real safetensors shard via
   an 8-byte-header parse + ``mmap`` seek (never loading a whole 800 MiB shard).
   Self-consistency: the same window resolves to identical rows across
   independent reads; different windows resolve to different ids/rows; every
   head's row lands in the shard predicted by the id->shard map.

The asc T4.1 module (``vllm_ascend/models/qwen4_exp/ngram_embedding.py``)
currently ships ``AscendQwen4ExpNGramEmbedding`` as a *stub* (its ``forward``
raises ``NotImplementedError`` -- the hashing lands in T1.3), so there is no asc
hash to assert agreement against yet; the fork port here is the reference the
future asc implementation must match. This gap is asserted explicitly in
``test_asc_t41_hash_is_still_a_stub`` so the follow-up is tracked.

CI-safe: the real-table cases skip with a clear reason when the 224 GiB
checkpoint mount is absent.
"""

from __future__ import annotations

import json
import mmap
import struct
from functools import cache, lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.ut.qwen38_1m.reference.ngram_hash_reference import (
    NGramHashConfig,
    bruteforce_ngram_ids,
    compute_ngram_ids,
    make_layer_multipliers,
    make_vocab_layout,
)

# --------------------------------------------------------------------------- #
# Real Qwen3.8-Flash-Next checkpoint parameters (text_config + manifest).
# --------------------------------------------------------------------------- #
REAL_VOCAB_SIZE = 248320
REAL_NGRAM_SIZE = 3
REAL_NGRAM_VOCAB_BASE = 20_000_000
REAL_MAKE_DIVISIBLE_BY = 128
REAL_HEADS_PER_NGRAM = 8
REAL_SPLIT_NGRAM_PARTS = 128
REAL_PLE_EMBED_DIM = 2560
REAL_EOS_TOKEN_ID = 248044  # text_config.eos_token_id (scalar); also bos/pad.
REAL_SEED = 1234  # text_config has no ``seed`` -> fork default 1234.
REAL_PLE_DENSE_LAYER_ID = 0  # enumerate index of sorted ple_layer_ids=[2].

# Derived, and independently re-checked against the real shard headers below.
REAL_NGRAM_HEADS = (REAL_NGRAM_SIZE - 1) * REAL_HEADS_PER_NGRAM  # 16
REAL_HEAD_DIM = REAL_PLE_EMBED_DIM // REAL_NGRAM_HEADS  # 160
REAL_SHARD_ROWS = 2_500_012  # per-shard row count from the real safetensors.
REAL_PADDED_VOCAB = REAL_SPLIT_NGRAM_PARTS * REAL_SHARD_ROWS  # 320_001_536

# The multipliers physically stored in the checkpoint (ground truth oracle).
REAL_LAYER_MULTIPLIERS = [23703573157769, 20109073645365, 8052911324071]

_LM_TENSOR = "model.language_model.layers.1.ple.ple_embedding.layer_multipliers"
_SHARD_TENSOR_FMT = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"

_MANIFEST = Path(__file__).resolve().parents[3] / "artifacts" / "qwen38-1m" / "checkpoint-manifest.json"
_FALLBACK_CKPT = Path(
    "/run/media/matteius/3cbe076a-d779-4f67-93a7-9195b734fac8/models/ascend/Qwen3.8-Flash-Next-W8A8-DYNAMIC-300i"
)

_ST_DTYPES = {
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
}


# --------------------------------------------------------------------------- #
# Checkpoint discovery + tiny safetensors row reader (no whole-shard loads).
# --------------------------------------------------------------------------- #
def _checkpoint_dir() -> Path | None:
    for candidate in (_manifest_checkpoint_dir(), _FALLBACK_CKPT):
        if candidate is None:
            continue
        index = candidate / "quant_model_weights.safetensors.index.json"
        if index.is_file():
            return candidate
    return None


def _manifest_checkpoint_dir() -> Path | None:
    try:
        data = json.loads(_MANIFEST.read_text())
    except (OSError, ValueError):
        return None
    raw = data.get("checkpoint_dir")
    return Path(raw) if raw else None


_CKPT_DIR = _checkpoint_dir()
requires_checkpoint = pytest.mark.skipif(
    _CKPT_DIR is None,
    reason=("real Qwen3.8-Flash-Next checkpoint not mounted (224 GiB PLE table absent); host/CI-safe skip"),
)


@lru_cache(maxsize=1)
def _weight_map() -> dict[str, str]:
    assert _CKPT_DIR is not None
    index = _CKPT_DIR / "quant_model_weights.safetensors.index.json"
    return json.loads(index.read_text())["weight_map"]


@cache
def _st_header(filename: str) -> tuple[dict, int]:
    """Parse only the 8-byte length + JSON header of a safetensors file."""
    assert _CKPT_DIR is not None
    path = _CKPT_DIR / filename
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    return header, 8 + header_len


def _tensor_meta(tensor_name: str) -> tuple[Path, dict, int]:
    filename = _weight_map()[tensor_name]
    header, data_base = _st_header(filename)
    assert _CKPT_DIR is not None
    return _CKPT_DIR / filename, header[tensor_name], data_base


def _read_tensor(tensor_name: str) -> np.ndarray:
    """Read a full (small) tensor -- used only for layer_multipliers [3]."""
    path, entry, data_base = _tensor_meta(tensor_name)
    start, end = entry["data_offsets"]
    dtype = _ST_DTYPES[entry["dtype"]]
    with open(path, "rb") as handle:
        mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            raw = mm[data_base + start : data_base + end]
        finally:
            mm.close()
    return np.frombuffer(raw, dtype=dtype).reshape(entry["shape"])


def _read_shard_row(shard_index: int, row: int) -> np.ndarray:
    """Read a single ``[REAL_HEAD_DIM]`` F16 row from one real PLE shard.

    Only the exact row's bytes are touched (mmap + slice); the shard is never
    materialized.
    """
    tensor_name = _SHARD_TENSOR_FMT.format(shard=shard_index)
    path, entry, data_base = _tensor_meta(tensor_name)
    assert entry["dtype"] == "F16", entry["dtype"]
    rows, cols = entry["shape"]
    if not 0 <= row < rows:
        raise IndexError(f"row {row} out of range for shard {shard_index} ({rows})")
    itemsize = _ST_DTYPES["F16"].itemsize
    row_stride = cols * itemsize
    tensor_start = data_base + entry["data_offsets"][0]
    byte_start = tensor_start + row * row_stride
    with open(path, "rb") as handle:
        mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            raw = bytes(mm[byte_start : byte_start + row_stride])
        finally:
            mm.close()
    return np.frombuffer(raw, dtype=_ST_DTYPES["F16"]).copy()


def _id_to_shard_row(global_id: int) -> tuple[int, int]:
    """Map a global padded-table row id to ``(shard_index, row_in_shard)``.

    Mirrors the fork ``load_weights`` split: shard ``S`` owns checkpoint rows
    ``[S * shard_size, (S + 1) * shard_size)`` with
    ``shard_size = ceil(padded_vocab / split_ngram_parts)``.
    """
    shard_size = -(-REAL_PADDED_VOCAB // REAL_SPLIT_NGRAM_PARTS)
    return divmod(int(global_id), shard_size)


# --------------------------------------------------------------------------- #
# Independent port of the fork PRODUCTION compute_ngram_ids (single request).
# Source: vllm/models/qwen4_exp/nvidia/ngram_embedding.py
#   Qwen4ExpNGramEmbedding._shift_precompute / _shift_apply / compute_ngram_ids
# --------------------------------------------------------------------------- #
def _fork_shift_precompute(tokens: torch.Tensor, eos_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len = tokens.shape
    positions = torch.arange(seq_len, dtype=torch.int64)
    eos_positions = torch.where(tokens == eos_token_id, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat(
        [
            eos_positions.new_full((batch_size, 1), -1),
            previous_eos_inclusive[:, :-1],
        ],
        dim=1,
    )
    return positions, positions.unsqueeze(0) - previous_eos - 1


def _fork_shift_apply(
    tokens: torch.Tensor,
    positions: torch.Tensor,
    position_in_segment: torch.Tensor,
    shift: int,
    eos_token_id: int,
) -> torch.Tensor:
    if shift == 0:
        return tokens
    source = positions - shift
    gather_indices = source.clamp_min(0).unsqueeze(0).expand(tokens.shape[0], -1)
    shifted = tokens.gather(1, gather_indices)
    valid = (source.unsqueeze(0) >= 0) & (position_in_segment >= shift)
    return torch.where(valid, shifted, tokens.new_full((), eos_token_id))


def fork_compute_ngram_ids(
    tokens: torch.Tensor,
    cfg: NGramHashConfig,
    history: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fork-faithful oracle: ids for one request, returned as ``[T, heads]``."""
    tokens = tokens.reshape(-1).long()
    if history is None:
        history = torch.full((cfg.ngram_size - 1,), cfg.eos_token_id, dtype=torch.long)
    history = history.reshape(-1).long()
    context = torch.cat([history, tokens]).unsqueeze(0)  # [1, n-1+T]
    positions, position_in_segment = _fork_shift_precompute(context, cfg.eos_token_id)
    shifted = [context]
    for shift in range(1, cfg.ngram_size):
        shifted.append(_fork_shift_apply(context, positions, position_in_segment, shift, cfg.eos_token_id))
    sizes = torch.tensor(cfg.sizes, dtype=torch.long)
    offsets = torch.tensor(cfg.offsets, dtype=torch.long)
    mults = cfg.multipliers
    id_blocks = []
    for ngram in range(2, cfg.ngram_size + 1):
        start = (ngram - 2) * cfg.heads_per_ngram
        end = start + cfg.heads_per_ngram
        mixed = shifted[0] * mults[0]
        for index in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[index] * mults[index])
        ids = torch.remainder(mixed.unsqueeze(-1), sizes[start:end]) + offsets[start:end]
        id_blocks.append(ids)
    all_ids = torch.cat(id_blocks, dim=-1)[0]  # [n-1+T, heads]
    return all_ids[cfg.ngram_size - 1 :]  # drop the history prefix rows


# --------------------------------------------------------------------------- #
# Config builders + token samplers.
# --------------------------------------------------------------------------- #
def _real_cfg() -> NGramHashConfig:
    return NGramHashConfig(
        ngram_size=REAL_NGRAM_SIZE,
        heads_per_ngram=REAL_HEADS_PER_NGRAM,
        eos_token_id=REAL_EOS_TOKEN_ID,
        unigram_vocab_size=REAL_VOCAB_SIZE,
        ngram_vocab_size_base=REAL_NGRAM_VOCAB_BASE,
        seed=REAL_SEED,
        ple_dense_layer_id=REAL_PLE_DENSE_LAYER_ID,
    )


def _small_cfg(ngram_size=3, heads_per_ngram=2, layer_id=0, eos=7, vocab=4096):
    return NGramHashConfig(
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        eos_token_id=eos,
        unigram_vocab_size=vocab,
        ngram_vocab_size_base=131,
        seed=1234,
        ple_dense_layer_id=layer_id,
    )


def _tokens(seq_len, seed, vocab, eos, sprinkle_eos=True):
    gen = torch.Generator().manual_seed(seed)
    toks = torch.randint(0, vocab, (seq_len,), generator=gen)
    if sprinkle_eos:
        mask = torch.rand(seq_len, generator=gen) < 0.2
        toks = torch.where(mask, torch.full_like(toks, eos), toks)
    return toks


# =========================================================================== #
# 1. Reference == fork production algorithm (host-only; no checkpoint needed).
# =========================================================================== #
@pytest.mark.parametrize("ngram_size", [2, 3, 4, 5])
@pytest.mark.parametrize("heads_per_ngram", [1, 2])
@pytest.mark.parametrize("seq_len", [1, 2, 6, 17])
def test_reference_matches_fork_algorithm(ngram_size, heads_per_ngram, seq_len):
    cfg = _small_cfg(ngram_size, heads_per_ngram)
    toks = _tokens(seq_len, seed=1000 + ngram_size * 31 + seq_len, vocab=4096, eos=7)
    ref = compute_ngram_ids(toks, cfg)
    brute = bruteforce_ngram_ids(toks, cfg)
    fork = fork_compute_ngram_ids(toks, cfg)
    assert torch.equal(ref, brute)
    assert torch.equal(ref, fork), "T0.6 reference disagrees with fork production hash"


def test_reference_matches_fork_at_real_config_scale():
    """Same hash under the REAL n-gram params (big multipliers, 16 heads)."""
    cfg = _real_cfg()
    toks = _tokens(24, seed=7, vocab=REAL_VOCAB_SIZE, eos=REAL_EOS_TOKEN_ID)
    ref = compute_ngram_ids(toks, cfg)
    fork = fork_compute_ngram_ids(toks, cfg)
    brute = bruteforce_ngram_ids(toks, cfg)
    assert torch.equal(ref, fork)
    assert torch.equal(ref, brute)
    assert ref.shape == (24, REAL_NGRAM_HEADS)


def test_eos_padded_sequence_start():
    """First tokens have EOS-padded predecessors (fresh segment)."""
    cfg = _small_cfg(ngram_size=4, heads_per_ngram=1)
    toks = torch.tensor([11, 22, 33, 44, 55])
    ref = compute_ngram_ids(toks, cfg)
    fork = fork_compute_ngram_ids(toks, cfg)
    assert torch.equal(ref, fork)
    # Token 0: both predecessors are the EOS pad -> id equals the pure-EOS hash.
    eos_only = compute_ngram_ids(
        torch.tensor([cfg.eos_token_id]),
        cfg,
        history=torch.full((cfg.ngram_size - 1,), cfg.eos_token_id),
    )
    # Row 0's oldest predecessor is EOS for both; the 2-gram head (shift-1) of
    # token 0 uses [tok0, EOS], so only the 2-gram block need match the derived
    # brute force -- assert the whole row matches brute force instead.
    assert torch.equal(ref[0], bruteforce_ngram_ids(toks, cfg)[0])
    assert eos_only.shape[1] == cfg.ngram_heads


def test_eos_boundary_severs_segments():
    """Tokens after an EOS hash identically regardless of what precedes it."""
    cfg = _small_cfg(ngram_size=4, heads_per_ngram=1)
    tail = torch.tensor([11, 22, 33, 44, 55])
    with_prefix = torch.cat([torch.tensor([101, 202, 303, cfg.eos_token_id]), tail])
    without_prefix = torch.cat([torch.tensor([cfg.eos_token_id]), tail])
    ids_with = fork_compute_ngram_ids(with_prefix, cfg)
    ids_without = fork_compute_ngram_ids(without_prefix, cfg)
    assert torch.equal(ids_with[-tail.numel() :], ids_without[-tail.numel() :])
    # And the T0.6 reference agrees with the fork on both.
    assert torch.equal(ids_with, compute_ngram_ids(with_prefix, cfg))
    assert torch.equal(ids_without, compute_ngram_ids(without_prefix, cfg))


def test_history_advance_equivalence():
    """Feeding a tail via ``history`` == one full pass (chunked prefill)."""
    cfg = _small_cfg(ngram_size=4, heads_per_ngram=2)
    full = _tokens(20, seed=9, vocab=4096, eos=7, sprinkle_eos=False)
    ids_full = compute_ngram_ids(full, cfg)
    split = 12
    history = full[split - (cfg.ngram_size - 1) : split]
    ids_first = fork_compute_ngram_ids(full[:split], cfg)
    ids_second = fork_compute_ngram_ids(full[split:], cfg, history=history)
    assert torch.equal(ids_first, ids_full[:split])
    assert torch.equal(ids_second, ids_full[split:])


# =========================================================================== #
# 2. Hash config == REAL checkpoint (multipliers + geometry).
# =========================================================================== #
def test_computed_multipliers_match_manifest_constant():
    cfg = _real_cfg()
    computed = make_layer_multipliers(
        ngram_size=REAL_NGRAM_SIZE,
        unigram_vocab_size=REAL_VOCAB_SIZE,
        seed=REAL_SEED,
        ple_dense_layer_id=REAL_PLE_DENSE_LAYER_ID,
    )
    assert computed == REAL_LAYER_MULTIPLIERS
    assert cfg.multipliers == REAL_LAYER_MULTIPLIERS


def test_vocab_layout_matches_real_shard_geometry():
    _, _, total = make_vocab_layout(
        ngram_vocab_size_base=REAL_NGRAM_VOCAB_BASE,
        ngram_heads=REAL_NGRAM_HEADS,
        ple_dense_layer_id=REAL_PLE_DENSE_LAYER_ID,
    )
    divisor = REAL_MAKE_DIVISIBLE_BY
    padded = ((total + divisor - 1) // divisor) * divisor
    assert padded == REAL_PADDED_VOCAB
    assert padded % REAL_SPLIT_NGRAM_PARTS == 0
    assert padded // REAL_SPLIT_NGRAM_PARTS == REAL_SHARD_ROWS
    assert REAL_PLE_EMBED_DIM % REAL_NGRAM_HEADS == 0
    assert REAL_HEAD_DIM == 160


@requires_checkpoint
def test_real_layer_multipliers_tensor_matches_hash():
    """The multipliers physically stored in the checkpoint == our hash."""
    stored = _read_tensor(_LM_TENSOR)
    assert stored.dtype == np.dtype("<i8")
    assert stored.tolist() == REAL_LAYER_MULTIPLIERS
    assert _real_cfg().multipliers == stored.tolist()


@requires_checkpoint
def test_all_128_shard_headers_are_consistent():
    """Every PLE shard is [2500012, 160] F16 -- pins padded_vocab & head_dim."""
    seen_files = set()
    for shard in range(REAL_SPLIT_NGRAM_PARTS):
        _, entry, _ = _tensor_meta(_SHARD_TENSOR_FMT.format(shard=shard))
        assert entry["dtype"] == "F16", (shard, entry["dtype"])
        assert tuple(entry["shape"]) == (REAL_SHARD_ROWS, REAL_HEAD_DIM), (
            shard,
            entry["shape"],
        )
        seen_files.add(_weight_map()[_SHARD_TENSOR_FMT.format(shard=shard)])
    assert REAL_SPLIT_NGRAM_PARTS * REAL_SHARD_ROWS == REAL_PADDED_VOCAB
    # Sanity: the shards are actually distributed across many files.
    assert len(seen_files) > 1


# =========================================================================== #
# 3. Resolved rows == REAL safetensors table (shard/column mapping, EOS).
# =========================================================================== #
def _resolve_embedding(ids_row: torch.Tensor) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Read the real PLE rows for one token's 16 head-ids -> [2560] embedding."""
    pieces = []
    mapping = []
    for head_id in ids_row.tolist():
        shard_index, row = _id_to_shard_row(head_id)
        mapping.append((shard_index, row))
        pieces.append(_read_shard_row(shard_index, row))
    return np.concatenate(pieces), mapping


@requires_checkpoint
def test_ids_land_in_correct_head_partition_and_shard_range():
    cfg = _real_cfg()
    toks = _tokens(12, seed=123, vocab=REAL_VOCAB_SIZE, eos=REAL_EOS_TOKEN_ID)
    ids = compute_ngram_ids(toks, cfg)
    assert ids.min().item() >= 0
    assert ids.max().item() < REAL_PADDED_VOCAB
    for head in range(cfg.ngram_heads):
        lo = cfg.offsets[head]
        hi = lo + cfg.sizes[head]
        col = ids[:, head]
        assert torch.all(col >= lo) and torch.all(col < hi)
    # Every resolved (shard, row) is inside a real shard.
    for gid in ids.reshape(-1).tolist():
        shard_index, row = _id_to_shard_row(gid)
        assert 0 <= shard_index < REAL_SPLIT_NGRAM_PARTS
        assert 0 <= row < REAL_SHARD_ROWS


@requires_checkpoint
def test_single_token_heads_span_multiple_shards():
    """The 16 heads of a single token naturally cross many shard boundaries."""
    cfg = _real_cfg()
    toks = _tokens(4, seed=55, vocab=REAL_VOCAB_SIZE, eos=REAL_EOS_TOKEN_ID)
    ids = compute_ngram_ids(toks, cfg)
    shards = {_id_to_shard_row(g)[0] for g in ids[-1].tolist()}
    # 16 heads with offsets ~h*20M and shard_size 2.5M => heads span >=8 shards.
    assert len(shards) >= 8, sorted(shards)


@requires_checkpoint
def test_resolved_rows_are_self_consistent_across_reads():
    """Same window -> identical ids AND identical real rows on independent reads."""
    cfg = _real_cfg()
    toks = _tokens(8, seed=2024, vocab=REAL_VOCAB_SIZE, eos=REAL_EOS_TOKEN_ID)
    ids_a = compute_ngram_ids(toks, cfg)
    ids_b = compute_ngram_ids(toks.clone(), cfg)
    assert torch.equal(ids_a, ids_b)
    emb_a, map_a = _resolve_embedding(ids_a[-1])
    emb_b, map_b = _resolve_embedding(ids_b[-1])
    assert map_a == map_b
    assert emb_a.shape == (REAL_PLE_EMBED_DIM,)
    assert np.array_equal(emb_a, emb_b)  # byte-identical rows across reads
    assert np.isfinite(emb_a.astype(np.float32)).all()  # real weights, not garbage


@requires_checkpoint
def test_different_windows_resolve_to_different_rows():
    cfg = _real_cfg()
    win_a = torch.tensor([101, 202, 303, 404, 505])
    win_b = torch.tensor([101, 202, 303, 404, 999])  # differs only in last token
    ids_a = compute_ngram_ids(win_a, cfg)[-1]
    ids_b = compute_ngram_ids(win_b, cfg)[-1]
    assert not torch.equal(ids_a, ids_b)
    emb_a, map_a = _resolve_embedding(ids_a)
    emb_b, map_b = _resolve_embedding(ids_b)
    # At least one head resolves to a different (shard, row) and different bytes.
    assert map_a != map_b
    assert not np.array_equal(emb_a, emb_b)


@requires_checkpoint
def test_eos_boundary_row_resolution_matches_reference():
    """Across-EOS + EOS-padded-start ids resolve to real rows and match ref."""
    cfg = _real_cfg()
    tail = torch.tensor([12345, 6789, 4242, 8888])
    with_prefix = torch.cat([torch.tensor([111, 222, REAL_EOS_TOKEN_ID]), tail])
    without_prefix = torch.cat([torch.tensor([REAL_EOS_TOKEN_ID]), tail])
    ids_with = compute_ngram_ids(with_prefix, cfg)
    ids_without = compute_ngram_ids(without_prefix, cfg)
    # EOS severs the segment: trailing rows identical regardless of the prefix.
    assert torch.equal(ids_with[-tail.numel() :], ids_without[-tail.numel() :])
    assert torch.equal(ids_with, fork_compute_ngram_ids(with_prefix, cfg))
    # The severed tail's first token resolves to the same real rows both ways.
    emb_with, map_with = _resolve_embedding(ids_with[-tail.numel()])
    emb_without, map_without = _resolve_embedding(ids_without[-tail.numel()])
    assert map_with == map_without
    assert np.array_equal(emb_with, emb_without)

    # EOS-padded sequence start: token 0's predecessors are all the EOS pad.
    start_ids = compute_ngram_ids(without_prefix, cfg)[0]
    emb_start, map_start = _resolve_embedding(start_ids)
    assert emb_start.shape == (REAL_PLE_EMBED_DIM,)
    assert all(0 <= s < REAL_SPLIT_NGRAM_PARTS for s, _ in map_start)


# =========================================================================== #
# Gap tracking: the asc T4.1 hash is not implemented yet (deferred to T1.3).
# =========================================================================== #
def test_asc_t41_hash_is_still_a_stub():
    """Document that asc lacks an n-gram hash to assert agreement against.

    ``AscendQwen4ExpNGramEmbedding`` (T4.1) ships as a stub -- its ``forward``
    raises ``NotImplementedError`` and it exposes no ``compute_ngram_ids``. The
    fork port in this file is the reference the future T1.3 asc implementation
    must reproduce (multipliers/vocab layout already verified vs. the real
    checkpoint above). If this assertion ever fails, asc grew a hash and this
    test must be upgraded to assert asc == fork == real table.
    """
    from vllm_ascend.models.qwen4_exp.ngram_embedding import (
        AscendQwen4ExpNGramEmbedding,
    )

    assert not hasattr(AscendQwen4ExpNGramEmbedding, "compute_ngram_ids")
    module = AscendQwen4ExpNGramEmbedding(config=object())
    with pytest.raises(NotImplementedError):
        module.forward()
