# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T1.3 -- AscendQwen4ExpNGramEmbedding.compute_ngram_ids + forward.

Closes the T4.1 stub: the Ascend n-gram embedding now hashes tokens to per-head
global PLE row ids and gathers ``[num_tokens, ple_embed_dim]`` embeddings.

Two things are verified with NO NPU / NO 224 GiB checkpoint:

1. ``compute_ngram_ids`` reproduces the T0.6/T4.2-verified reference
   (``tests/ut/qwen38_1m/reference/ngram_hash_reference.py``) byte-for-byte on
   sampled windows and on the boundary cases the fork stresses -- EOS-padded
   sequence start, across-EOS severing, and the multi-request
   ``query_start_loc`` / ``ngram_context`` packing.
2. ``forward`` gathers the correct rows: a SMALL synthetic PLE table is placed
   via the real T4.1 ``/dev/shm`` shared-mmap transport
   (``AscendPLESharedMmapEmbeddingMethod``) with row ``i`` == ``i``, so the
   assembled embedding is exactly the per-head ids tiled across ``head_dim``.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from types import SimpleNamespace

import pytest
import torch

from tests.ut.qwen38_1m.reference.ngram_hash_reference import (
    NGramHashConfig,
)
from tests.ut.qwen38_1m.reference.ngram_hash_reference import (
    compute_ngram_ids as ref_compute_ngram_ids,
)
from vllm_ascend.models.qwen4_exp.ngram_embedding import (
    AscendPLESharedMmapEmbeddingMethod,
    AscendQwen4ExpNGramEmbedding,
)

# Small, host-friendly n-gram + PLE geometry (mirrors the real 3-gram shape:
# ngram_heads == (ngram_size - 1) * heads_per_ngram; ple_embed_dim divisible by
# it). Kept tiny so the /dev/shm table is a few KiB, never the real 224 GiB.
_EOS = 7
_VOCAB = 4096
_NGRAM_VOCAB_BASE = 131
_SEED = 1234


def _make_config(*, ngram_size=3, heads_per_ngram=2, head_dim=5):
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    return SimpleNamespace(
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        eos_token_id=_EOS,
        vocab_size=_VOCAB,
        ngram_vocab_size_base=_NGRAM_VOCAB_BASE,
        seed=_SEED,
        ple_embed_dim=ngram_heads * head_dim,
    )


def _ref_cfg(config, ple_dense_layer_id=0):
    return NGramHashConfig(
        ngram_size=config.ngram_size,
        heads_per_ngram=config.heads_per_ngram,
        eos_token_id=config.eos_token_id,
        unigram_vocab_size=config.vocab_size,
        ngram_vocab_size_base=config.ngram_vocab_size_base,
        seed=config.seed,
        ple_dense_layer_id=ple_dense_layer_id,
    )


def _module(config, ple_method=None, ple_dense_layer_id=0):
    return AscendQwen4ExpNGramEmbedding(
        config=config,
        ple_method=ple_method,
        ple_dense_layer_id=ple_dense_layer_id,
    )


def _single_request(module, tokens, history):
    """Run module.compute_ngram_ids for one request (query_start_loc=[0, T])."""
    tokens = tokens.reshape(-1)
    qsl = torch.tensor([0, tokens.numel()], dtype=torch.long)
    ctx = history.reshape(1, -1)
    return module.compute_ngram_ids(tokens, qsl, ctx)


def _tokens(seq_len, seed, *, sprinkle_eos=True):
    gen = torch.Generator().manual_seed(seed)
    toks = torch.randint(0, _VOCAB, (seq_len,), generator=gen)
    if sprinkle_eos:
        mask = torch.rand(seq_len, generator=gen) < 0.2
        toks = torch.where(mask, torch.full_like(toks, _EOS), toks)
    return toks


# --------------------------------------------------------------------------- #
# Small T4.1 /dev/shm PLE table with row i == i (so gather is verifiable).
# --------------------------------------------------------------------------- #
def _identity_table(num_embeddings, embedding_dim, dtype):
    rows = torch.arange(num_embeddings, dtype=torch.float32).unsqueeze(1)
    return rows.expand(num_embeddings, embedding_dim).to(dtype)


@pytest.fixture
def shm_ple_method(request):
    """Build a tiny shared-mmap PLE table sized to a given module's vocab."""
    created: list[AscendPLESharedMmapEmbeddingMethod] = []

    def _build(module):
        path = os.path.join("/dev/shm", f"vllm_ascend_test_ple_{uuid.uuid4().hex}.bin")
        method = AscendPLESharedMmapEmbeddingMethod(
            module.total_vocab_size,
            module.head_dim,
            shm_path=path,
            create=True,
            table_source=_identity_table,
        )
        created.append(method)
        return method

    yield _build

    for method in created:
        path = method.shm_path
        method.close()
        with contextlib.suppress(OSError):
            os.unlink(path)


# =========================================================================== #
# 1. compute_ngram_ids == T0.6/T4.2 reference.
# =========================================================================== #
@pytest.mark.parametrize("ngram_size", [2, 3, 4, 5])
@pytest.mark.parametrize("heads_per_ngram", [1, 2])
@pytest.mark.parametrize("seq_len", [1, 2, 6, 17])
def test_compute_ngram_ids_matches_reference_sampled(ngram_size, heads_per_ngram, seq_len):
    config = _make_config(ngram_size=ngram_size, heads_per_ngram=heads_per_ngram)
    module = _module(config)
    cfg = _ref_cfg(config)
    toks = _tokens(seq_len, seed=2000 + ngram_size * 31 + seq_len)
    history = torch.full((ngram_size - 1,), _EOS, dtype=torch.long)

    asc = _single_request(module, toks, history)
    ref = ref_compute_ngram_ids(toks, cfg)
    assert asc.shape == (seq_len, module.ngram_heads)
    assert torch.equal(asc, ref)


def test_compute_ngram_ids_ple_dense_layer_id_shifts_hash():
    """A non-zero PLE dense layer id changes multipliers/offsets (and matches)."""
    config = _make_config(ngram_size=3, heads_per_ngram=2)
    module0 = _module(config, ple_dense_layer_id=0)
    module1 = _module(config, ple_dense_layer_id=1)
    toks = _tokens(9, seed=99, sprinkle_eos=False)
    history = torch.full((config.ngram_size - 1,), _EOS, dtype=torch.long)

    ids0 = _single_request(module0, toks, history)
    ids1 = _single_request(module1, toks, history)
    assert not torch.equal(ids0, ids1)
    assert torch.equal(ids0, ref_compute_ngram_ids(toks, _ref_cfg(config, 0)))
    assert torch.equal(ids1, ref_compute_ngram_ids(toks, _ref_cfg(config, 1)))


def test_eos_padded_sequence_start_matches_reference():
    """Token 0's predecessors are all the EOS pad (fresh segment)."""
    config = _make_config(ngram_size=4, heads_per_ngram=1)
    module = _module(config)
    cfg = _ref_cfg(config)
    toks = torch.tensor([11, 22, 33, 44, 55])
    history = torch.full((config.ngram_size - 1,), _EOS, dtype=torch.long)
    asc = _single_request(module, toks, history)
    assert torch.equal(asc, ref_compute_ngram_ids(toks, cfg))


def test_across_eos_severs_segment_matches_reference():
    """Tokens after an internal EOS hash independently of the earlier prefix."""
    config = _make_config(ngram_size=4, heads_per_ngram=1)
    module = _module(config)
    cfg = _ref_cfg(config)
    tail = torch.tensor([11, 22, 33, 44, 55])
    with_prefix = torch.cat([torch.tensor([101, 202, 303, _EOS]), tail])
    without_prefix = torch.cat([torch.tensor([_EOS]), tail])
    history = torch.full((config.ngram_size - 1,), _EOS, dtype=torch.long)

    ids_with = _single_request(module, with_prefix, history)
    ids_without = _single_request(module, without_prefix, history)
    # EOS severs the segment: the trailing tail hashes identically both ways.
    assert torch.equal(ids_with[-tail.numel() :], ids_without[-tail.numel() :])
    # And both agree with the reference exactly.
    assert torch.equal(ids_with, ref_compute_ngram_ids(with_prefix, cfg))
    assert torch.equal(ids_without, ref_compute_ngram_ids(without_prefix, cfg))


def test_history_context_matches_reference():
    """A non-EOS ``ngram_context`` history (chunked prefill tail) is honored."""
    config = _make_config(ngram_size=4, heads_per_ngram=2)
    module = _module(config)
    cfg = _ref_cfg(config)
    full = _tokens(20, seed=9, sprinkle_eos=False)
    split = 12
    history = full[split - (config.ngram_size - 1) : split]
    tail = full[split:]
    asc = _single_request(module, tail, history)
    ref = ref_compute_ngram_ids(tail, cfg, history=history)
    assert torch.equal(asc, ref)


def test_multi_request_packing_matches_per_request_reference():
    """query_start_loc packs several requests; each slice matches the reference."""
    config = _make_config(ngram_size=3, heads_per_ngram=2)
    module = _module(config)
    cfg = _ref_cfg(config)

    req_a = _tokens(5, seed=11)
    req_b = _tokens(7, seed=22)
    req_c = _tokens(1, seed=33)
    input_ids = torch.cat([req_a, req_b, req_c])
    qsl = torch.tensor([0, 5, 12, 13], dtype=torch.long)
    history = torch.full((3, config.ngram_size - 1), _EOS, dtype=torch.long)

    packed = module.compute_ngram_ids(input_ids, qsl, history)
    assert packed.shape == (13, module.ngram_heads)
    assert torch.equal(packed[0:5], ref_compute_ngram_ids(req_a, cfg))
    assert torch.equal(packed[5:12], ref_compute_ngram_ids(req_b, cfg))
    assert torch.equal(packed[12:13], ref_compute_ngram_ids(req_c, cfg))


# =========================================================================== #
# 2. forward gathers correct rows via the T4.1 /dev/shm transport.
# =========================================================================== #
def test_forward_gathers_correct_rows(shm_ple_method):
    config = _make_config(ngram_size=3, heads_per_ngram=2, head_dim=5)
    module = _module(config)
    method = shm_ple_method(module)
    module.ple_method = method

    toks = _tokens(11, seed=777)
    input_ids = toks
    qsl = torch.tensor([0, toks.numel()], dtype=torch.long)
    history = torch.full((1, config.ngram_size - 1), _EOS, dtype=torch.long)
    hidden_states = torch.zeros(toks.numel(), config.ple_embed_dim, dtype=module.embedding_dtype)

    out = module.forward(hidden_states, input_ids, qsl, history)

    assert out.shape == (toks.numel(), config.ple_embed_dim)
    assert out.dtype == module.embedding_dtype
    # Row i of the synthetic table == i, so the gathered embedding is exactly the
    # per-head ids tiled across head_dim.
    ids = module.compute_ngram_ids(input_ids, qsl, history)
    expected = ids.to(module.embedding_dtype).repeat_interleave(module.head_dim, dim=1)
    assert torch.equal(out, expected)


def test_forward_matches_manual_gather(shm_ple_method):
    """forward == compute_ngram_ids then a straight per-head table lookup."""
    config = _make_config(ngram_size=4, heads_per_ngram=2, head_dim=8)
    module = _module(config)
    method = shm_ple_method(module)
    module.ple_method = method

    toks = _tokens(6, seed=4242)
    qsl = torch.tensor([0, toks.numel()], dtype=torch.long)
    history = torch.full((1, config.ngram_size - 1), _EOS, dtype=torch.long)
    out = module.forward(torch.empty(0), toks, qsl, history)

    ids = module.compute_ngram_ids(toks, qsl, history)
    table = method.weight
    manual = table.index_select(0, ids.reshape(-1).long()).reshape(toks.numel(), config.ple_embed_dim)
    assert torch.equal(out, manual)


def test_forward_requires_ple_method():
    config = _make_config()
    module = _module(config)  # no ple_method
    toks = _tokens(3, seed=1)
    qsl = torch.tensor([0, 3], dtype=torch.long)
    history = torch.full((1, config.ngram_size - 1), _EOS, dtype=torch.long)
    with pytest.raises(RuntimeError):
        module.forward(torch.empty(0), toks, qsl, history)
