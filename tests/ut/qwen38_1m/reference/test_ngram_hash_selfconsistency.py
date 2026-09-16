# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-consistency tests for the n-gram hashing reference.

Acceptance (T0.6): n-gram hash identities == a brute-force re-derivation, exact.
Also validates EOS boundary padding and history-advance equivalence.
"""

import pytest
import torch

from tests.ut.qwen38_1m.reference.ngram_hash_reference import (
    NGramHashConfig,
    bruteforce_ngram_ids,
    compute_ngram_ids,
    make_layer_multipliers,
)

_EOS = 7
_VOCAB = 4096


def _cfg(ngram_size=4, heads_per_ngram=2, layer_id=0):
    return NGramHashConfig(
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        eos_token_id=_EOS,
        unigram_vocab_size=_VOCAB,
        ngram_vocab_size_base=131,
        seed=1234,
        ple_dense_layer_id=layer_id,
    )


def _random_tokens(seq_len, seed, include_eos=True):
    gen = torch.Generator().manual_seed(seed)
    high = _VOCAB
    tokens = torch.randint(0, high, (seq_len,), generator=gen)
    if include_eos:
        # sprinkle EOS to exercise the boundary logic
        mask = torch.rand(seq_len, generator=gen) < 0.2
        tokens = torch.where(mask, torch.full_like(tokens, _EOS), tokens)
    return tokens


@pytest.mark.parametrize("ngram_size", [2, 3, 4, 5])
@pytest.mark.parametrize("heads_per_ngram", [1, 2])
@pytest.mark.parametrize("seq_len", [1, 2, 5, 33])
def test_vectorized_equals_bruteforce(ngram_size, heads_per_ngram, seq_len):
    cfg = _cfg(ngram_size, heads_per_ngram)
    tokens = _random_tokens(seq_len, seed=100 + ngram_size * 10 + seq_len)
    vec = compute_ngram_ids(tokens, cfg)
    brute = bruteforce_ngram_ids(tokens, cfg)
    assert torch.equal(vec, brute)


def test_ids_within_partitions():
    """Every id lands in its head's [offset, offset+size) partition."""
    cfg = _cfg()
    tokens = _random_tokens(40, seed=5)
    ids = compute_ngram_ids(tokens, cfg)
    for head in range(cfg.ngram_heads):
        lo = cfg.offsets[head]
        hi = lo + cfg.sizes[head]
        col = ids[:, head]
        assert torch.all(col >= lo) and torch.all(col < hi)


def test_eos_boundary_isolates_segments():
    """An EOS between two segments must sever cross-segment predecessors.

    Tokens after an EOS must hash identically whether or not tokens precede the
    EOS, because predecessors cannot cross the EOS boundary.
    """
    cfg = _cfg(ngram_size=4, heads_per_ngram=1)
    tail = torch.tensor([11, 22, 33, 44, 55])
    with_prefix = torch.cat([torch.tensor([101, 202, 303]), torch.tensor([_EOS]), tail])
    without_prefix = torch.cat([torch.tensor([_EOS]), tail])
    ids_with = compute_ngram_ids(with_prefix, cfg)
    ids_without = compute_ngram_ids(without_prefix, cfg)
    # Compare the trailing `tail` rows in each.
    assert torch.equal(ids_with[-tail.numel() :], ids_without[-tail.numel() :])


def test_history_advance_matches_full_sequence():
    """Splitting a sequence and feeding the tail via `history` == one pass."""
    cfg = _cfg(ngram_size=4, heads_per_ngram=2)
    full = _random_tokens(20, seed=9, include_eos=False)
    ids_full = compute_ngram_ids(full, cfg)

    split = 12
    first = full[:split]
    second = full[split:]
    history = full[split - (cfg.ngram_size - 1) : split]
    ids_first = compute_ngram_ids(first, cfg)
    ids_second = compute_ngram_ids(second, cfg, history=history)

    assert torch.equal(ids_first, ids_full[:split])
    assert torch.equal(ids_second, ids_full[split:])


def test_history_advance_bruteforce_agreement():
    cfg = _cfg(ngram_size=5, heads_per_ngram=1)
    tokens = _random_tokens(15, seed=21)
    history = torch.tensor([1, 2, 3, 4])
    vec = compute_ngram_ids(tokens, cfg, history=history)
    brute = bruteforce_ngram_ids(tokens, cfg, history=history)
    assert torch.equal(vec, brute)


def test_multipliers_are_odd_and_deterministic():
    """Multipliers are odd (2*x+1) and fully determined by seed + layer id."""
    m1 = make_layer_multipliers(ngram_size=4, unigram_vocab_size=_VOCAB, seed=1234, ple_dense_layer_id=3)
    m2 = make_layer_multipliers(ngram_size=4, unigram_vocab_size=_VOCAB, seed=1234, ple_dense_layer_id=3)
    assert m1 == m2
    assert all(m % 2 == 1 for m in m1)


def test_layer_id_changes_hashing():
    """Different PLE layers must not share multipliers / vocab layout."""
    cfg0 = _cfg(layer_id=0)
    cfg1 = _cfg(layer_id=1)
    assert cfg0.multipliers != cfg1.multipliers
