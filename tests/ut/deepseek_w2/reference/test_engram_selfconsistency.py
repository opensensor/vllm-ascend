# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-consistency tests for the Engram reference (E0.4, priority 2).

Acceptance:
  * Engram hash vs. brute force (exact integer equality).
  * gather / projection self-consistent (vectorized vs. per-element oracle).
"""

import pytest
import torch

from tests.ut.deepseek_w2.reference.engram_reference import (
    DEAD_ID,
    EngramHashLayout,
    bruteforce_engram_gate_project,
    bruteforce_engram_gather,
    bruteforce_engram_hashes,
    compute_engram_hashes,
    e8m0_scale,
    engram_forward,
    engram_gate_project,
    engram_gather,
)
from tests.ut.deepseek_w2.reference.tolerances import (
    ENGRAM_GATHER_ATOL,
    ENGRAM_GATHER_RTOL,
    ENGRAM_PROJ_ATOL,
    ENGRAM_PROJ_RTOL,
)

_EPS = 1e-6
_E8M0_BIAS = 127


def _layout(seed_layers=(0, 3, 7)):
    return EngramHashLayout(
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


# --- Stage 1: hash ----------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("seq_len", [1, 5, 16, 33])
def test_hash_matches_bruteforce(seed, seq_len):
    layout = _layout()
    ids = _rand_ids(seq_len, seed)
    vec = compute_engram_hashes(ids, layout)
    brute = bruteforce_engram_hashes(ids, layout)
    assert torch.equal(vec, brute)


def test_hash_shape_and_bucket_ranges():
    layout = _layout()
    ids = _rand_ids(20, seed=5)
    out = compute_engram_hashes(ids, layout)
    assert out.shape == (20, len(layout.layer_ids), layout.n_hash_cols)
    # Every id lands in its column's prime bucket range [offset, offset+prime).
    for layer in range(len(layout.layer_ids)):
        for col in range(layout.n_hash_cols):
            offset = int(layout.offsets[layer, col].item())
            prime = int(layout.primes[layer, col].item())
            col_ids = out[:, layer, col]
            assert torch.all(col_ids >= offset)
            assert torch.all(col_ids < offset + prime)


def test_primes_disjoint_and_unique():
    layout = _layout()
    flat = layout.primes.flatten().tolist()
    assert len(set(flat)) == len(flat)  # never reused across (layer, ngram, head)


@pytest.mark.parametrize("seed", [10, 11])
def test_hash_with_dead_mask(seed):
    layout = _layout()
    ids = _rand_ids(12, seed)
    dead = torch.zeros(12, dtype=torch.bool)
    dead[4] = True
    dead[9] = True
    vec = compute_engram_hashes(ids, layout, dead_mask=dead)
    brute = bruteforce_engram_hashes(ids, layout, dead_mask=dead)
    assert torch.equal(vec, brute)


@pytest.mark.parametrize("seed", [20, 21])
def test_hash_with_history(seed):
    ids = _rand_ids(10, seed)
    history = _rand_ids(3, seed + 1000)
    vec = compute_engram_hashes(ids, _layout(), history=history)
    brute = bruteforce_engram_hashes(ids, _layout(), history=history)
    assert torch.equal(vec, brute)


def test_dead_token_blocks_older_predecessors():
    """A dead token at the query position collapses the whole n-gram to pad."""
    layout = _layout()
    ids = _rand_ids(6, seed=99)
    dead = torch.zeros(6, dtype=torch.bool)
    dead[0] = True  # first token dead -> all its n-grams use pad only
    out = compute_engram_hashes(ids, layout, dead_mask=dead)
    brute = bruteforce_engram_hashes(ids, layout, dead_mask=dead)
    assert torch.equal(out, brute)


# --- Stage 2: gather --------------------------------------------------------


def _make_table(num_embeddings, dim, seed, block_size=32):
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(-8, 8, (num_embeddings, dim), generator=gen, dtype=torch.int8)
    # ue8m0 exponent bytes near the bias -> scales around 1.
    exps = torch.randint(
        _E8M0_BIAS - 2,
        _E8M0_BIAS + 3,
        (num_embeddings, dim // block_size),
        generator=gen,
        dtype=torch.uint8,
    )
    return codes, exps


@pytest.mark.parametrize("seed", [0, 1])
def test_gather_matches_bruteforce(seed):
    dim = 64
    codes, exps = _make_table(50, dim, seed)
    gen = torch.Generator().manual_seed(seed + 5)
    hash_ids = torch.randint(0, 50, (8, 3), generator=gen)
    vec = engram_gather(hash_ids, codes, exps)
    brute = bruteforce_engram_gather(hash_ids, codes, exps)
    torch.testing.assert_close(vec, brute, rtol=ENGRAM_GATHER_RTOL, atol=ENGRAM_GATHER_ATOL)


def test_e8m0_scale_is_power_of_two():
    exps = torch.tensor([125, 126, 127, 128, 129], dtype=torch.uint8)
    scales = e8m0_scale(exps)
    expected = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0], dtype=torch.float64)
    torch.testing.assert_close(scales, expected, rtol=0, atol=0)


# --- Stage 3: projection ----------------------------------------------------


def _rand(shape, seed, scale=1.0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.float64) * scale


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_projection_matches_bruteforce(seed):
    num_tokens, hc_mult, dim = 7, 3, 16
    hidden = _rand((num_tokens, hc_mult, dim), seed)
    kv = _rand((num_tokens, (hc_mult + 1) * dim), seed + 1)
    q_w = _rand((hc_mult, dim), seed + 2)
    k_w = _rand((hc_mult, dim), seed + 3)
    vec = engram_gate_project(hidden, kv, q_w, k_w, _EPS)
    brute = bruteforce_engram_gate_project(hidden, kv, q_w, k_w, _EPS)
    torch.testing.assert_close(vec, brute, rtol=ENGRAM_PROJ_RTOL, atol=ENGRAM_PROJ_ATOL)


def test_projection_token_mask_passthrough():
    """A masked-off token passes hidden through unchanged (gate == 0)."""
    num_tokens, hc_mult, dim = 4, 2, 8
    hidden = _rand((num_tokens, hc_mult, dim), 30)
    kv = _rand((num_tokens, (hc_mult + 1) * dim), 31)
    q_w = _rand((hc_mult, dim), 32)
    k_w = _rand((hc_mult, dim), 33)
    mask = torch.tensor([True, False, True, False])
    out = engram_gate_project(hidden, kv, q_w, k_w, _EPS, token_mask=mask)
    # Masked rows equal hidden exactly.
    torch.testing.assert_close(out[1], hidden[1], rtol=0, atol=0)
    torch.testing.assert_close(out[3], hidden[3], rtol=0, atol=0)
    # Unmasked rows generally differ from hidden.
    assert not torch.allclose(out[0], hidden[0])
    # And match the brute-force oracle where active.
    brute = bruteforce_engram_gate_project(hidden, kv, q_w, k_w, _EPS, token_mask=mask)
    torch.testing.assert_close(out, brute, rtol=ENGRAM_PROJ_RTOL, atol=ENGRAM_PROJ_ATOL)


def test_end_to_end_forward_self_consistent():
    """gather -> wkv -> projection agrees with a staged re-derivation."""
    num_tokens, hc_mult, dim = 6, 2, 32
    n_hash_cols, head_dim = 4, 16
    hidden = _rand((num_tokens, hc_mult, dim), 40)
    codes, exps = _make_table(60, head_dim, 41, block_size=16)
    gen = torch.Generator().manual_seed(42)
    hash_ids = torch.randint(0, 60, (num_tokens, n_hash_cols), generator=gen)
    wkv = _rand(((hc_mult + 1) * dim, n_hash_cols * head_dim), 43, scale=0.1)
    q_w = _rand((hc_mult, dim), 44)
    k_w = _rand((hc_mult, dim), 45)

    out = engram_forward(hidden, hash_ids, codes, exps, wkv, q_w, k_w, _EPS, block_size=16)
    # Staged oracle: brute gather -> matmul -> brute projection.
    rows = bruteforce_engram_gather(hash_ids, codes, exps, block_size=16)
    kv = rows.reshape(num_tokens, -1) @ wkv.t()
    brute = bruteforce_engram_gate_project(hidden, kv, q_w, k_w, _EPS)
    torch.testing.assert_close(out, brute, rtol=ENGRAM_PROJ_RTOL, atol=ENGRAM_PROJ_ATOL)


def test_dead_id_constant():
    assert DEAD_ID == -1
