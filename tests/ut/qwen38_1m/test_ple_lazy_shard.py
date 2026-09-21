# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T4.1c -- lazy per-shard mmap PLE transport (transport c).

The lazy transport reads PLE rows on demand from the checkpoint's 128 n-gram
shard tensors via per-shard read-only ``numpy.memmap``, never materializing the
95.43 GiB host table. Host tests cover:

1. A synthetic multi-shard checkpoint: ``id -> (shard, row) = divmod(id,
   shard_rows)`` mapping, correct gather vs a fully-materialized reference table,
   multi-shard / boundary / empty / 2D-id gathers, the no-op host budget, and the
   zero physical footprint.
2. The real 224 GiB checkpoint (skip-if-absent): rows gathered through the lazy
   transport match the independent byte-exact mmap reader from
   ``test_ngram_hash``, with negligible host footprint.

Run with ``--noconftest`` (the shared tests/ut/conftest.py fails to import on
this host):

    python3 -m pytest -q --noconftest tests/ut/qwen38_1m/test_ple_lazy_shard.py
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from vllm_ascend.models.qwen4_exp.ngram_embedding import (
    AscendPLELazyShardEmbeddingMethod,
    AscendPLETransport,
    PLETransportUnavailableError,
    create_ple_embedding_method,
)

_SHARD_FMT = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
_INDEX_NAME = "quant_model_weights.safetensors.index.json"
_FILE_NAME = "quant_model_weights-00001-of-00001.safetensors"

# Small synthetic geometry (real geometry is 128 shards of [2500012, 160]).
_SPLIT = 4
_SHARD_ROWS = 4
_NUM_EMBEDDINGS = _SPLIT * _SHARD_ROWS  # 16
_DIM = 4


def _write_checkpoint(tmp_path, shards: list[torch.Tensor]) -> None:
    """Write one tiny safetensors file + index for the synthetic shards."""
    tensors = {_SHARD_FMT.format(shard=i): t for i, t in enumerate(shards)}
    save_file(tensors, tmp_path / _FILE_NAME)
    index = {"metadata": {}, "weight_map": {name: _FILE_NAME for name in tensors}}
    (tmp_path / _INDEX_NAME).write_text(json.dumps(index))


def _make_shards(split: int = _SPLIT, shard_rows: int = _SHARD_ROWS, dim: int = _DIM) -> list[torch.Tensor]:
    """Deterministic, distinct, exactly-fp16-representable small integers."""
    return [
        (torch.arange(shard_rows * dim, dtype=torch.int64).view(shard_rows, dim) + s * 1000).to(torch.float16)
        for s in range(split)
    ]


def _make_method(tmp_path, *, split=_SPLIT, num_embeddings=_NUM_EMBEDDINGS, dim=_DIM):
    return AscendPLELazyShardEmbeddingMethod(
        num_embeddings,
        dim,
        checkpoint_dir=tmp_path,
        shard_tensor_fmt=_SHARD_FMT,
        split_ngram_parts=split,
    )


def _ref_table(shards: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(shards, dim=0)


# --------------------------------------------------------------------------- #
# 1. synthetic multi-shard checkpoint
# --------------------------------------------------------------------------- #
def test_gather_matches_full_table(tmp_path):
    shards = _make_shards()
    _write_checkpoint(tmp_path, shards)
    method = _make_method(tmp_path)
    ref = _ref_table(shards)
    ids = torch.tensor([0, 1, 3, 4, 7, 8, 11, 15])
    got = method.gather_rows(ids)
    assert got.dtype == torch.float16
    assert got.shape == (8, _DIM)
    assert torch.equal(got, ref[ids])


def test_id_to_shard_row_boundaries(tmp_path):
    """id 3 -> shard 0 row 3; id 4 -> shard 1 row 0; id 15 -> shard 3 row 3."""
    shards = _make_shards()
    _write_checkpoint(tmp_path, shards)
    method = _make_method(tmp_path)
    ref = _ref_table(shards)
    ids = torch.arange(_NUM_EMBEDDINGS)
    got = method.gather_rows(ids)
    assert torch.equal(got, ref)  # whole-table round-trip through 4 shards


def test_gather_skew_concentrates_on_few_shards(tmp_path):
    shards = _make_shards()
    _write_checkpoint(tmp_path, shards)
    method = _make_method(tmp_path)
    ref = _ref_table(shards)
    # All rows in shard 3 -> only one mmap is opened.
    ids = torch.tensor([12, 13, 14, 15])
    got = method.gather_rows(ids)
    assert torch.equal(got, ref[ids])
    assert len(method._shard_memmaps) == 1


def test_gather_empty_and_2d_ids(tmp_path):
    shards = _make_shards()
    _write_checkpoint(tmp_path, shards)
    method = _make_method(tmp_path)
    assert method.gather_rows(torch.empty(0, dtype=torch.int64)).shape == (0, _DIM)
    ids2d = torch.tensor([[0, 1], [2, 3]])
    got = method.gather_rows(ids2d)
    assert got.shape == (4, _DIM)
    assert torch.equal(got, _ref_table(shards)[ids2d.reshape(-1)])


def test_no_host_budget_and_zero_footprint(tmp_path):
    shards = _make_shards()
    _write_checkpoint(tmp_path, shards)
    method = _make_method(tmp_path)
    assert method.physical_bytes == 0
    # A 1-byte host budget would reject the full table; the lazy path is a no-op.
    method.verify_host_budget(host_total_bytes=1)
    # The logical table size is still reported (documentation), not the resident size.
    assert method.table_bytes == _NUM_EMBEDDINGS * _DIM * 2


def test_out_of_range_id_raises(tmp_path):
    shards = _make_shards()
    _write_checkpoint(tmp_path, shards)
    method = _make_method(tmp_path)
    with pytest.raises(IndexError):
        method.gather_rows(torch.tensor([_NUM_EMBEDDINGS]))  # one past the padded vocab


def test_missing_checkpoint_dir_raises():
    with pytest.raises(PLETransportUnavailableError):
        AscendPLELazyShardEmbeddingMethod(
            _NUM_EMBEDDINGS, _DIM, checkpoint_dir=None, shard_tensor_fmt=_SHARD_FMT, split_ngram_parts=_SPLIT
        )


def test_missing_shard_tensor_raises(tmp_path):
    _write_checkpoint(tmp_path, _make_shards())
    # A shard tensor name that is absent from the index must fail loudly.
    bad = AscendPLELazyShardEmbeddingMethod(
        _NUM_EMBEDDINGS,
        _DIM,
        checkpoint_dir=tmp_path,
        shard_tensor_fmt="nope.shard_{shard}.weight",
        split_ngram_parts=_SPLIT,
    )
    with pytest.raises(PLETransportUnavailableError):
        bad.gather_rows(torch.tensor([0]))


def test_factory_lazy_shard_branch(tmp_path):
    _write_checkpoint(tmp_path, _make_shards())
    method = create_ple_embedding_method(
        num_embeddings=_NUM_EMBEDDINGS,
        embedding_dim=_DIM,
        transport=AscendPLETransport.LAZY_SHARD,
        checkpoint_dir=tmp_path,
        shard_tensor_fmt=_SHARD_FMT,
        split_ngram_parts=_SPLIT,
    )
    assert isinstance(method, AscendPLELazyShardEmbeddingMethod)
    ref = _ref_table(_make_shards())
    assert torch.equal(method.gather_rows(torch.tensor([0, 5, 15])), ref[torch.tensor([0, 5, 15])])


# --------------------------------------------------------------------------- #
# 2. real checkpoint (skip-if-absent)
# --------------------------------------------------------------------------- #
def _real_checkpoint_params():
    """Real geometry + independent byte-exact reader from test_ngram_hash."""
    from tests.ut.qwen38_1m.test_ngram_hash import (
        _CKPT_DIR,
        REAL_HEAD_DIM,
        REAL_PADDED_VOCAB,
        REAL_SHARD_ROWS,
        REAL_SPLIT_NGRAM_PARTS,
        _read_shard_row,
    )

    return REAL_HEAD_DIM, REAL_PADDED_VOCAB, REAL_SHARD_ROWS, REAL_SPLIT_NGRAM_PARTS, _CKPT_DIR, _read_shard_row


def test_real_checkpoint_rows_match_independent_reader():
    try:
        head_dim, padded_vocab, shard_rows, split, ckpt_dir, read_shard_row = _real_checkpoint_params()
    except Exception:  # pragma: no cover - checkpoint not mounted
        pytest.skip("real checkpoint not mounted")
    if ckpt_dir is None:
        pytest.skip("real checkpoint not mounted")

    method = AscendPLELazyShardEmbeddingMethod(padded_vocab, head_dim, checkpoint_dir=ckpt_dir, split_ngram_parts=split)
    assert method.shard_rows == shard_rows
    assert method.physical_bytes == 0

    # A few rows spanning several shards (incl. the first/last shard edges).
    probes = [
        (0, 0),
        (0, shard_rows - 1),
        (1, 0),
        (split // 2, 12345),
        (split - 1, shard_rows - 1),
    ]
    ids = torch.tensor([s * shard_rows + r for s, r in probes])
    got = method.gather_rows(ids)
    for k, (s, r) in enumerate(probes):
        expected = torch.from_numpy(read_shard_row(s, r)).view(1, -1)
        assert torch.equal(got[k : k + 1], expected), (s, r)
    # Lazy transport only opened the shards actually touched (not all 128).
    assert len(method._shard_memmaps) <= len(probes)
