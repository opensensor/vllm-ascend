# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the DeepSeek V4.1 Engram host lookup (E2.3).

Parity target is the E0.4 eager reference
(``tests/ut/deepseek_w2/reference/engram_reference.py``) at the declared
tolerances (``reference/tolerances.py``):

  * n-gram hash ids match the reference vectorized *and* brute-force paths bit
    for bit (``ENGRAM_HASH_EXACT``);
  * ~W4 packed row gather matches the reference int8+ue8m0 gather at rounding
    level (``ENGRAM_GATHER_*``);
  * the gated projection and the end-to-end forward match the reference
    (``ENGRAM_PROJ_*``);
  * the host ~W4 table is a single shared copy (mmap) and its host bytes are
    counted once under ``ENGRAM_HOST`` (never x world_size).

Runs host-side only (no NPU / Triton). The shared ``tests/ut/conftest.py`` fails
to import here, so run with ``--noconftest``:

    python3 -m pytest -q --noconftest tests/ut/deepseek_w2/test_engram.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.ut.deepseek_w2.reference import engram_reference as ref
from tests.ut.deepseek_w2.reference.tolerances import (
    ENGRAM_GATHER_ATOL,
    ENGRAM_GATHER_RTOL,
    ENGRAM_PROJ_ATOL,
    ENGRAM_PROJ_RTOL,
)
from vllm_ascend.models.deepseek_v41.dtype_policy import (
    ASCEND_DEEPSEEKV41_DTYPE_POLICY,
)
from vllm_ascend.models.deepseek_v41.engram import (
    ENGRAM_LAYER_IDS,
    AscendDeepseekV41Engram,
    DeepseekEngramHasher,
    EngramHashLayout,
    compute_engram_hashes,
    create_engram_host_tables,
    e8m0_to_fp32_scale,
    engram_forward_packed,
    engram_gate_project,
    engram_gather_packed,
    pack_w4_codes,
    unpack_w4_codes,
)
from vllm_ascend.models.qwen4_exp.ngram_embedding import (
    AscendPLETransport,
)
from vllm_ascend.models.qwen4_exp.ple_prefetch import AscendRowPrefetcher
from vllm_ascend.observability.deepseek_w2_mem_accounting import (
    DeepSeekW2MemComponent,
    DeepSeekW2MemoryAccountant,
)

_EPS = 1e-6
_E8M0_BIAS = 127


# --------------------------------------------------------------------------- #
# Layouts / helpers (small synthetic geometry -- never the 224 GB table).
# --------------------------------------------------------------------------- #


def _layout(seed_layers=(0, 3, 7)) -> EngramHashLayout:
    return EngramHashLayout(
        layer_ids=seed_layers,
        max_ngram_size=4,
        n_heads=2,
        engram_vocab_size=257,
        compressed_vocab_size=193,
        pad_id=0,
        head_dim=16,
    )


def _ref_layout(seed_layers=(0, 3, 7)) -> ref.EngramHashLayout:
    return ref.EngramHashLayout(
        layer_ids=seed_layers,
        max_ngram_size=4,
        n_heads=2,
        engram_vocab_size=257,
        compressed_vocab_size=193,
        pad_id=0,
    )


def _rand_ids(seq_len, seed, vocab=193):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (seq_len,), generator=gen)


def _make_ref_table(num_embeddings, dim, seed, block_size):
    """int8 codes in [-8, 7] (the signed 4-bit grid) + ue8m0 exponent bytes."""
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(-8, 8, (num_embeddings, dim), generator=gen, dtype=torch.int8)
    exps = torch.randint(
        _E8M0_BIAS - 2,
        _E8M0_BIAS + 3,
        (num_embeddings, dim // block_size),
        generator=gen,
        dtype=torch.uint8,
    )
    return codes, exps


# --------------------------------------------------------------------------- #
# Stage 1: n-gram hash -- exact parity vs the E0.4 reference.
# --------------------------------------------------------------------------- #


def test_layout_matches_reference_exactly():
    mine, theirs = _layout(), _ref_layout()
    assert torch.equal(mine.primes, theirs.primes)
    assert torch.equal(mine.offsets, theirs.offsets)
    assert torch.equal(mine.multipliers, theirs.multipliers)
    assert mine.n_hash_cols == theirs.n_hash_cols


def test_primes_disjoint_and_unique():
    flat = _layout().primes.flatten().tolist()
    assert len(set(flat)) == len(flat)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("seq_len", [1, 5, 16, 33])
def test_hash_matches_reference_and_bruteforce(seed, seq_len):
    ids = _rand_ids(seq_len, seed)
    mine = compute_engram_hashes(ids, _layout())
    ref_vec = ref.compute_engram_hashes(ids, _ref_layout())
    ref_brute = ref.bruteforce_engram_hashes(ids, _ref_layout())
    assert torch.equal(mine, ref_vec)  # parity with the reference vectorized path
    assert torch.equal(mine, ref_brute)  # hash == brute force (ENGRAM_HASH_EXACT)


def test_hash_bucket_ranges():
    ids = _rand_ids(20, seed=5)
    layout = _layout()
    out = compute_engram_hashes(ids, layout)
    assert out.shape == (20, len(layout.layer_ids), layout.n_hash_cols)
    for layer in range(len(layout.layer_ids)):
        for col in range(layout.n_hash_cols):
            offset = int(layout.offsets[layer, col].item())
            prime = int(layout.primes[layer, col].item())
            col_ids = out[:, layer, col]
            assert torch.all(col_ids >= offset)
            assert torch.all(col_ids < offset + prime)


@pytest.mark.parametrize("seed", [10, 11])
def test_hash_with_dead_mask(seed):
    ids = _rand_ids(12, seed)
    dead = torch.zeros(12, dtype=torch.bool)
    dead[4] = True
    dead[9] = True
    mine = compute_engram_hashes(ids, _layout(), dead_mask=dead)
    brute = ref.bruteforce_engram_hashes(ids, _ref_layout(), dead_mask=dead)
    assert torch.equal(mine, brute)


@pytest.mark.parametrize("seed", [20, 21])
def test_hash_with_history(seed):
    ids = _rand_ids(10, seed)
    history = _rand_ids(3, seed + 1000)
    mine = compute_engram_hashes(ids, _layout(), history=history)
    brute = ref.bruteforce_engram_hashes(ids, _ref_layout(), history=history)
    assert torch.equal(mine, brute)


def test_hasher_module_matches_function():
    layout = _layout()
    hasher = DeepseekEngramHasher(layout)
    ids = _rand_ids(17, seed=3)
    out = hasher(ids)
    assert out.shape == (17, len(layout.layer_ids), layout.n_hash_cols)
    assert torch.equal(out, ref.bruteforce_engram_hashes(ids, _ref_layout()))


# --------------------------------------------------------------------------- #
# Stage 2: ~W4 pack / unpack and gather -- parity vs reference int8+ue8m0.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_w4_pack_roundtrip(seed):
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(-8, 8, (7, 64), generator=gen, dtype=torch.int64)
    packed = pack_w4_codes(codes)
    assert packed.dtype == torch.uint8
    assert packed.shape == (7, 32)  # two codes per byte
    restored = unpack_w4_codes(packed, dim=64)
    assert torch.equal(restored, codes)


def test_pack_rejects_odd_dim():
    with pytest.raises(ValueError):
        pack_w4_codes(torch.zeros(3, dtype=torch.int64))


def test_e8m0_scale_is_power_of_two():
    exps = torch.tensor([125, 126, 127, 128, 129], dtype=torch.uint8)
    scales = e8m0_to_fp32_scale(exps)
    expected = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0], dtype=torch.float32)
    torch.testing.assert_close(scales, expected, rtol=0, atol=0)


@pytest.mark.parametrize("seed", [0, 1])
def test_gather_matches_reference(seed):
    dim, block_size = 64, 32
    codes, exps = _make_ref_table(50, dim, seed, block_size)
    gen = torch.Generator().manual_seed(seed + 5)
    hash_ids = torch.randint(0, 50, (8, 3), generator=gen)

    # ~W4 storage built from the same logical codes/scales as the reference.
    packed = pack_w4_codes(codes.to(torch.int64))
    scale = e8m0_to_fp32_scale(exps)
    mine = engram_gather_packed(hash_ids, packed, scale, dim, block_size, compute_dtype=torch.float64)
    theirs = ref.engram_gather(hash_ids, codes, exps, block_size)
    torch.testing.assert_close(mine, theirs, rtol=ENGRAM_GATHER_RTOL, atol=ENGRAM_GATHER_ATOL)


def test_dequantize_matches_reference_bruteforce():
    dim, block_size = 32, 16
    codes, exps = _make_ref_table(12, dim, seed=9, block_size=block_size)
    packed = pack_w4_codes(codes.to(torch.int64))
    scale = e8m0_to_fp32_scale(exps)
    ids = torch.arange(12).reshape(4, 3)
    mine = engram_gather_packed(ids, packed, scale, dim, block_size, compute_dtype=torch.float64)
    brute = ref.bruteforce_engram_gather(ids, codes, exps, block_size)
    torch.testing.assert_close(mine, brute, rtol=ENGRAM_GATHER_RTOL, atol=ENGRAM_GATHER_ATOL)


# --------------------------------------------------------------------------- #
# Stage 3: gated projection + end-to-end forward -- parity vs reference.
# --------------------------------------------------------------------------- #


def _rand(shape, seed, scale=1.0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.float64) * scale


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_projection_matches_reference(seed):
    num_tokens, hc_mult, dim = 7, 3, 16
    hidden = _rand((num_tokens, hc_mult, dim), seed)
    kv = _rand((num_tokens, (hc_mult + 1) * dim), seed + 1)
    q_w = _rand((hc_mult, dim), seed + 2)
    k_w = _rand((hc_mult, dim), seed + 3)
    mine = engram_gate_project(hidden, kv, q_w, k_w, _EPS)
    theirs = ref.engram_gate_project(hidden, kv, q_w, k_w, _EPS)
    torch.testing.assert_close(mine, theirs, rtol=ENGRAM_PROJ_RTOL, atol=ENGRAM_PROJ_ATOL)


def test_projection_token_mask_passthrough():
    num_tokens, hc_mult, dim = 4, 2, 8
    hidden = _rand((num_tokens, hc_mult, dim), 30)
    kv = _rand((num_tokens, (hc_mult + 1) * dim), 31)
    q_w = _rand((hc_mult, dim), 32)
    k_w = _rand((hc_mult, dim), 33)
    mask = torch.tensor([True, False, True, False])
    out = engram_gate_project(hidden, kv, q_w, k_w, _EPS, token_mask=mask)
    torch.testing.assert_close(out[1], hidden[1], rtol=0, atol=0)
    torch.testing.assert_close(out[3], hidden[3], rtol=0, atol=0)
    theirs = ref.engram_gate_project(hidden, kv, q_w, k_w, _EPS, token_mask=mask)
    torch.testing.assert_close(out, theirs, rtol=ENGRAM_PROJ_RTOL, atol=ENGRAM_PROJ_ATOL)


def test_end_to_end_forward_matches_reference():
    num_tokens, hc_mult, dim = 6, 2, 32
    n_hash_cols, head_dim, block_size = 4, 16, 16
    hidden = _rand((num_tokens, hc_mult, dim), 40)
    codes, exps = _make_ref_table(60, head_dim, 41, block_size)
    gen = torch.Generator().manual_seed(42)
    hash_ids = torch.randint(0, 60, (num_tokens, n_hash_cols), generator=gen)
    wkv = _rand(((hc_mult + 1) * dim, n_hash_cols * head_dim), 43, scale=0.1)
    q_w = _rand((hc_mult, dim), 44)
    k_w = _rand((hc_mult, dim), 45)

    packed = pack_w4_codes(codes.to(torch.int64))
    scale = e8m0_to_fp32_scale(exps)
    mine = engram_forward_packed(
        hidden, hash_ids, packed, scale, wkv, q_w, k_w, _EPS, dim=head_dim, block_size=block_size
    )
    theirs = ref.engram_forward(hidden, hash_ids, codes, exps, wkv, q_w, k_w, _EPS, block_size=block_size)
    torch.testing.assert_close(mine, theirs, rtol=ENGRAM_PROJ_RTOL, atol=ENGRAM_PROJ_ATOL)


# --------------------------------------------------------------------------- #
# Host ~W4 table: single shared copy, gather correctness, host-byte accounting.
# --------------------------------------------------------------------------- #


def _synthetic_sources(layout, head_dim, block_size, seed):
    """Packed-code + fp32-scale sources for each layer's small W4 table."""
    from vllm_ascend.models.deepseek_v41.engram import (
        e8m0_to_fp32_scale as _scale,
    )
    from vllm_ascend.models.deepseek_v41.engram import (
        pack_w4_codes as _pack,
    )

    code_sources, scale_sources, ref_tables = [], [], []
    for hash_index in range(len(layout.layer_ids)):
        rows = layout.num_embeddings(hash_index)
        codes, exps = _make_ref_table(rows, head_dim, seed + hash_index, block_size)
        code_sources.append(_pack(codes.to(torch.int64)))
        scale_sources.append(_scale(exps))
        ref_tables.append((codes, exps))
    return code_sources, scale_sources, ref_tables


def test_host_table_gather_and_single_shared_copy(tmp_path):
    layout = _layout()
    head_dim, block_size = 16, 16
    code_sources, scale_sources, ref_tables = _synthetic_sources(layout, head_dim, block_size, seed=100)
    tables = create_engram_host_tables(
        layout,
        head_dim=head_dim,
        block_size=block_size,
        transport=AscendPLETransport.SHARED_MMAP,
        shm_dir=str(tmp_path),
        code_sources=code_sources,
        scale_sources=scale_sources,
        world_size=4,
        reserve_bytes=0,  # tiny synthetic table -- skip the 48 GiB reserve guard
    )
    try:
        assert len(tables) == len(layout.layer_ids)
        # Gather correctness vs the reference gather, per layer.
        for hash_index in range(len(layout.layer_ids)):
            layer_table = tables[hash_index]
            rows = layout.num_embeddings(hash_index)
            ids = torch.arange(min(rows, 20))
            got = layer_table.gather_rows(ids, compute_dtype=torch.float64)
            codes, exps = ref_tables[hash_index]
            want = ref.engram_gather(ids.reshape(-1, 1), codes, exps, block_size).squeeze(1)
            torch.testing.assert_close(got, want, rtol=ENGRAM_GATHER_RTOL, atol=ENGRAM_GATHER_ATOL)
            # Storage is a single shared mmap copy (never x world_size).
            assert layer_table.codes.mmap_length == layer_table.codes.table_bytes
            assert layer_table.codes.physical_bytes == layer_table.codes.table_bytes
            assert layer_table.scales.physical_bytes == layer_table.scales.table_bytes
    finally:
        tables.close()


def test_host_bytes_counted_once(tmp_path):
    layout = _layout()
    head_dim, block_size = 16, 16
    code_sources, scale_sources, _ = _synthetic_sources(layout, head_dim, block_size, seed=200)
    world_size = 4
    tables = create_engram_host_tables(
        layout,
        head_dim=head_dim,
        block_size=block_size,
        transport=AscendPLETransport.SHARED_MMAP,
        shm_dir=str(tmp_path),
        code_sources=code_sources,
        scale_sources=scale_sources,
        world_size=world_size,
        reserve_bytes=0,
    )
    try:
        accountant = DeepSeekW2MemoryAccountant(world_size=world_size)
        for rank in range(world_size):
            tables.record_host_bytes(accountant, rank)
        shared = tables.physical_bytes
        assert shared > 0
        # ENGRAM_HOST is one shared logical copy: the accountant returns the
        # single value, never the per-rank sum.
        assert accountant.host_table_bytes() == shared
        assert accountant.host_table_bytes() != shared * world_size
        # Host bytes are excluded from every rank's device total.
        for rank in range(world_size):
            report = accountant.rank_report(rank)
            assert report.components[DeepSeekW2MemComponent.ENGRAM_HOST] == shared
            assert report.device_bytes() == 0
    finally:
        tables.close()


def test_host_table_diverging_bytes_rejected(tmp_path):
    """A per-rank (x world_size) Engram footprint must be rejected."""
    layout = _layout()
    head_dim, block_size = 16, 16
    code_sources, scale_sources, _ = _synthetic_sources(layout, head_dim, block_size, seed=300)
    tables = create_engram_host_tables(
        layout,
        head_dim=head_dim,
        block_size=block_size,
        transport=AscendPLETransport.SHARED_MMAP,
        shm_dir=str(tmp_path),
        code_sources=code_sources,
        scale_sources=scale_sources,
        world_size=2,
        reserve_bytes=0,
    )
    try:
        accountant = DeepSeekW2MemoryAccountant(world_size=2)
        accountant.rank_report(0).add(DeepSeekW2MemComponent.ENGRAM_HOST, tables.physical_bytes)
        accountant.rank_report(1).add(DeepSeekW2MemComponent.ENGRAM_HOST, tables.physical_bytes * 2)
        with pytest.raises(ValueError):
            accountant.host_table_bytes()
    finally:
        tables.close()


def test_prefetcher_reuse_over_host_table(tmp_path):
    """The Qwen async row prefetcher drives the DeepSeek host table unchanged."""
    layout = _layout()
    head_dim, block_size = 16, 16
    code_sources, scale_sources, ref_tables = _synthetic_sources(layout, head_dim, block_size, seed=400)
    tables = create_engram_host_tables(
        layout,
        head_dim=head_dim,
        block_size=block_size,
        transport=AscendPLETransport.SHARED_MMAP,
        shm_dir=str(tmp_path),
        code_sources=code_sources,
        scale_sources=scale_sources,
        reserve_bytes=0,
    )
    try:
        layer_table = tables[0]
        prefetcher = AscendRowPrefetcher(layer_table)
        ids = torch.tensor([[0, 1, 2], [2, 1, 0]])  # repeated hot rows across the step
        out = prefetcher.decode_step(ids)
        assert out.shape == (2, 3, head_dim)
        # One batched wait for the whole step, not one per row.
        assert prefetcher.stream.wait_count == 1
        # Dedup removed the repeats (6 requested -> 3 unique rows).
        assert prefetcher.metrics.requested_rows == 6
        assert prefetcher.metrics.unique_rows == 3
        # Values still match a direct dequantized gather.
        direct = layer_table.gather_rows(ids.reshape(-1)).reshape(2, 3, head_dim)
        torch.testing.assert_close(out, direct, rtol=0, atol=0)
    finally:
        tables.close()


# --------------------------------------------------------------------------- #
# nn.Module surface E4.1 wires into ``_inject_engram``.
# --------------------------------------------------------------------------- #


def _engram_config(hidden_size=32, hc_mult=2):
    return SimpleNamespace(
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        rms_norm_eps=1e-5,
        engram_layer_ids=(0, 3, 7),
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=16,
        engram_compressed_vocab_size=193,
        engram_vocab_size=257,
        engram_pad_token_id=0,
    )


def test_module_forward_shape_and_dtype(tmp_path):
    layout = _layout()
    head_dim, block_size = 16, 16
    code_sources, scale_sources, _ = _synthetic_sources(layout, head_dim, block_size, seed=500)
    tables = create_engram_host_tables(
        layout,
        head_dim=head_dim,
        block_size=block_size,
        transport=AscendPLETransport.SHARED_MMAP,
        shm_dir=str(tmp_path),
        code_sources=code_sources,
        scale_sources=scale_sources,
        reserve_bytes=0,
    )
    try:
        config = _engram_config()
        engram = AscendDeepseekV41Engram(
            config=config,
            host_table=tables[0],
            layer_hash_index=0,
            layout=layout,
        )
        # Dtype surface reads from the authoritative policy.
        policy = ASCEND_DEEPSEEKV41_DTYPE_POLICY
        assert engram.wkv.weight.dtype == policy.cast_site("engram")
        assert engram.host_table.dtype == policy.cast_site("engram")

        num_tokens = 5
        hidden = torch.randn(num_tokens, config.hc_mult, config.hidden_size, dtype=torch.float16)
        ids = _rand_ids(num_tokens, seed=1)
        hash_ids = compute_engram_hashes(ids, layout)[:, 0]  # this layer's columns
        out = engram(hidden, hash_ids)
        assert out.shape == hidden.shape
        assert out.dtype == hidden.dtype

        # A shut gate (token_mask False) passes the residual stream through.
        mask = torch.zeros(num_tokens, dtype=torch.bool)
        passthrough = engram(hidden, hash_ids, token_mask=mask)
        torch.testing.assert_close(passthrough, hidden, rtol=0, atol=0)
    finally:
        tables.close()


def test_module_matches_reference_forward(tmp_path):
    """The module forward equals the reference end-to-end forward on shared weights."""
    layout = _layout()
    head_dim, block_size = 16, 16
    code_sources, scale_sources, ref_tables = _synthetic_sources(layout, head_dim, block_size, seed=600)
    tables = create_engram_host_tables(
        layout,
        head_dim=head_dim,
        block_size=block_size,
        transport=AscendPLETransport.SHARED_MMAP,
        shm_dir=str(tmp_path),
        code_sources=code_sources,
        scale_sources=scale_sources,
        reserve_bytes=0,
    )
    try:
        config = _engram_config(hidden_size=16, hc_mult=2)
        engram = AscendDeepseekV41Engram(config=config, host_table=tables[0], layer_hash_index=0, layout=layout)
        n_hash_cols = layout.n_hash_cols
        # Load deterministic float64 weights, mirrored into the fp16 module.
        gen = torch.Generator().manual_seed(7)
        wkv = (
            torch.randn(
                config.hidden_size * (config.hc_mult + 1),
                n_hash_cols * head_dim,
                generator=gen,
                dtype=torch.float64,
            )
            * 0.1
        )
        q_w = torch.randn(config.hc_mult, config.hidden_size, generator=gen, dtype=torch.float64)
        k_w = torch.randn(config.hc_mult, config.hidden_size, generator=gen, dtype=torch.float64)
        with torch.no_grad():
            engram.wkv.weight.copy_(wkv.to(torch.float16))
            engram.q_weight.copy_(q_w.to(torch.float16))
            engram.k_weight.copy_(k_w.to(torch.float16))

        num_tokens = 6
        hidden64 = _rand((num_tokens, config.hc_mult, config.hidden_size), 8)
        ids = _rand_ids(num_tokens, seed=2)
        hash_ids = compute_engram_hashes(ids, layout)[:, 0]

        out = engram(hidden64.to(torch.float16), hash_ids).to(torch.float64)
        codes, exps = ref_tables[0]
        expected = ref.engram_forward(
            hidden64, hash_ids, codes, exps, wkv, q_w, k_w, config.rms_norm_eps, block_size=block_size
        )
        # fp16 storage of hidden/weights/rows: compare at half-precision tolerance.
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)
    finally:
        tables.close()


def test_default_layer_ids_constant():
    assert ENGRAM_LAYER_IDS == (1, 14)
