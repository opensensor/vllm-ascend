# SPDX-License-Identifier: Apache-2.0
"""Sparse pool causality, incomplete tails, and stage control regressions."""

import socket

import pytest
import torch

from tools.sidecard.layer_actor import CONTROL, receive_control
from tools.sidecard.layer_indexer import PoolIndexer, hadamard_reference, select_tokens
from tools.sidecard.stage_wire import metadata, validate_metadata


def test_rotation_matches_independent_sylvester_matrix():
    torch.manual_seed(101)
    values = torch.randn(3, 128)
    matrix = torch.ones((1, 1), dtype=torch.float64)
    for _ in range(7):
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    expected = (values.double() @ matrix / 128**0.5).float()
    torch.testing.assert_close(hadamard_reference(values, torch), expected, rtol=1e-5, atol=1e-6)


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
