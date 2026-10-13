# SPDX-License-Identifier: Apache-2.0
"""Sparse pool causality, incomplete tails, and stage control regressions."""

import socket
import sys
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.layer_actor import CONTROL, receive_control
from tools.sidecard.layer_indexer import PoolIndexer, cuda_rotation_enabled, hadamard_reference, select_tokens
from tools.sidecard.stage_wire import metadata, validate_metadata


def test_rotation_matches_independent_sylvester_matrix():
    torch.manual_seed(101)
    values = torch.randn(3, 128)
    matrix = torch.ones((1, 1), dtype=torch.float64)
    for _ in range(7):
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    expected = (values.double() @ matrix / 128**0.5).float()
    torch.testing.assert_close(hadamard_reference(values, torch), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("selection", [None, 0, 1, "true", [], {}])
def test_cuda_rotation_rejects_non_boolean_configuration(selection):
    with pytest.raises(ValueError, match="boolean"):
        cuda_rotation_enabled({"cuda_indexer_hadamard": selection})


def test_cuda_rotation_defaults_off_and_accepts_explicit_selection():
    assert not cuda_rotation_enabled({})
    assert cuda_rotation_enabled({"cuda_indexer_hadamard": True})
    assert not cuda_rotation_enabled({"cuda_indexer_hadamard": False})


@pytest.mark.parametrize("device", ["cpu", "npu:0", "npu:1"])
def test_fused_rotation_rejects_non_cuda_owner_before_allocations(device):
    with pytest.raises(ValueError, match="CUDA owner"):
        PoolIndexer({}, {}, device, None, 64, fused_cuda_rotation=True)


def test_fused_rotation_dispatch_keeps_reference_fallback_and_rounding(monkeypatch):
    values = torch.randn(2, 32, 128).bfloat16().float()
    expected = hadamard_reference(values, torch).bfloat16().half()
    calls = []

    def fused(x):
        calls.append(x)
        return expected

    monkeypatch.setitem(sys.modules, "tools.sidecard.cuda_layer_ops", SimpleNamespace(hadamard=fused))
    owner = object.__new__(PoolIndexer)
    owner.device, owner.torch, owner.fused_cuda_rotation = "cuda", torch, True
    assert owner.rotate(values) is expected
    assert calls == [values]
    owner.fused_cuda_rotation = False
    torch.testing.assert_close(owner.rotate(values), expected, rtol=0, atol=0)
    assert len(calls) == 1


def test_future_high_scores_cannot_displace_causal_pools_and_tail():
    scores = torch.tensor([[1, 9, 3, 1000], [1, 9, 3, 1000]]).float()
    positions = torch.tensor([10, 12])
    selected = select_tokens(scores, positions, 8, torch)
    assert set(selected[0][selected[0] >= 0].tolist()) == set(range(11))
    assert set(selected[1][selected[1] >= 0].tolist()) == set(range(4, 13))
    assert (selected[0] <= 10).all()
    assert (selected[1] <= 12).all()


def indexer():
    torch.manual_seed(132)
    weights = {
        "self_attn.indexer.wk.weight": torch.randn(128, 8).half(),
        "self_attn.indexer.weights_proj.weight": torch.randn(2, 8).half(),
        "self_attn.indexer.index_kpool_compress_gate": torch.randn(128, 8).half(),
        "self_attn.indexer.index_kpool_compress_ape": torch.randn(4, 128).half(),
        "self_attn.indexer.k_norm.weight": torch.ones(128).half(),
        "self_attn.indexer.k_norm.bias": torch.zeros(128).half(),
        "self_attn.indexer.wq_b.weight": torch.randn(256, 4).half(),
    }
    return PoolIndexer(weights, {"index_topk": 8, "index_n_heads": 2}, "cpu", torch, 64)


def test_pool_tail_survives_arbitrary_chunk_boundaries():
    complete, chunked = indexer(), indexer()
    torch.manual_seed(44)
    hidden = torch.randn(19, 8).half()
    complete.append(hidden)
    start = 0
    for end in (1, 4, 6, 11, 19):
        chunked.append(hidden[start:end])
        start = end
    assert complete.pool_count == chunked.pool_count == 4
    assert complete.tail_length == chunked.tail_length == 3
    torch.testing.assert_close(complete.pools[:4], chunked.pools[:4], rtol=0, atol=0)
    torch.testing.assert_close(complete.tail_keys[:3], chunked.tail_keys[:3], rtol=0, atol=0)
    torch.testing.assert_close(complete.tail_gates[:3], chunked.tail_gates[:3], rtol=0, atol=0)
    chunked.reset()
    assert chunked.pool_count == chunked.tail_length == 0


def test_sparse_selection_respects_every_query_position_in_crossing_chunk():
    cache = indexer()
    torch.manual_seed(21)
    hidden, qr = torch.randn(15, 8).half(), torch.randn(15, 4).half()
    assert cache.select(hidden[:6], qr[:6], 0) is None
    selected = cache.select(hidden[6:], qr[6:], 6)
    for offset, row in enumerate(selected):
        valid = row[row >= 0]
        assert len(valid) == len(set(valid.tolist()))
        assert (valid <= offset + 6).all()
        assert len(valid) <= 11


def test_request_context_reservation_is_bound_into_stage_identity():
    packet = metadata(32, 9, "a" * 64, request_context_limit=65536)
    assert packet["schema_version"] == 2
    assert validate_metadata(packet) == packet
    packet["schema_version"] = 1
    with pytest.raises(ValueError):
        validate_metadata(packet)
    with pytest.raises(ValueError):
        metadata(32, 9, "a" * 64, request_context_limit=16)


def test_control_plane_reassembles_fragments_without_npu_collectives():
    left, right = socket.socketpair()
    try:
        body = CONTROL.pack(32, 64, 65536)
        left.sendall(body[:3])
        left.sendall(body[3:])
        assert receive_control(right) == (32, 64, 65536)
        left.close()
        with pytest.raises(EOFError):
            receive_control(right)
    finally:
        left.close()
        right.close()
