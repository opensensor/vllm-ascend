# SPDX-License-Identifier: Apache-2.0
"""Captured indexer math preserves causal pools, tails and owner contracts."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.indexer_projection_benchmark import graph_mode, run
from tools.sidecard.indexer_projection_graph import IndexerProjectionGraph, enabled, project
from tools.sidecard.layer_indexer import PoolIndexer


def owner():
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


def test_projection_replay_keeps_every_pool_tail_and_sparse_selection_byte_exact():
    reference, captured = owner(), owner()
    calls = []

    def replay(hidden, qr):
        calls.append(hidden.shape[0])
        return project(captured, hidden, qr)

    captured.projection_graph_enabled = True
    captured.projection_graph = replay
    torch.manual_seed(51)
    hidden, qr = torch.randn(25, 8).half(), torch.randn(25, 4).half()
    for position in range(25):
        inputs = hidden[position : position + 1], qr[position : position + 1], position
        expected, actual = reference.select(*inputs), captured.select(*inputs)
        assert (expected is None and actual is None) or torch.equal(expected, actual)
        assert reference.tail_length == captured.tail_length
        assert reference.pool_count == captured.pool_count
        for name, count in (
            ("pools", reference.pool_count),
            ("tail_keys", reference.tail_length),
            ("tail_gates", reference.tail_length),
        ):
            assert (
                getattr(reference, name)[:count].numpy().tobytes() == getattr(captured, name)[:count].numpy().tobytes()
            )
    assert calls == [1] * 17
    reference.reset()
    captured.reset()
    assert torch.equal(reference.select(hidden[:12], qr[:12], 0), captured.select(hidden[:12], qr[:12], 0))
    assert len(calls) == 17


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_graph_flag_rejects_non_boolean_values(value):
    with pytest.raises(ValueError):
        enabled(dict(cuda_indexer_projection_graphs=value, cuda_indexer_hadamard=True))


def test_graph_flag_defaults_off_and_requires_fused_rotation():
    assert not enabled({})
    with pytest.raises(ValueError):
        enabled(dict(cuda_indexer_projection_graphs=True))
    assert enabled(dict(cuda_indexer_projection_graphs=True, cuda_indexer_hadamard=True))
    with pytest.raises(ValueError, match="qualified CUDA"):
        IndexerProjectionGraph(owner())


def test_replay_rejects_geometry_and_dtype_before_copying_then_closes():
    graph = object.__new__(IndexerProjectionGraph)
    graph.inputs = torch.zeros(1, 8).half(), torch.zeros(1, 4).half()
    graph.outputs = object()
    calls = []
    graph.graph = SimpleNamespace(replay=lambda: calls.append(True))
    assert graph(*graph.inputs) is graph.outputs
    with pytest.raises(ValueError, match="decode row"):
        graph(torch.zeros(2, 8).half(), torch.zeros(2, 4).half())
    with pytest.raises(ValueError, match="dtype"):
        graph(torch.zeros(1, 8), torch.zeros(1, 4).half())
    assert calls == [True]
    graph.close()
    assert graph.graph is graph.inputs is graph.outputs is None


@pytest.mark.parametrize("failure", [False, True])
def test_benchmark_mode_restores_all_owners_after_success_or_failure(failure):
    indexers = [SimpleNamespace(projection_graph=object(), projection_graph_enabled=value) for value in (True, False)]
    engine = SimpleNamespace(layers={i: SimpleNamespace(linear=False, indexer=x) for i, x in enumerate(indexers)})

    def execute():
        with graph_mode(engine, False):
            assert not any(x.projection_graph_enabled for x in indexers)
            if failure:
                raise RuntimeError("failed decode")

    if failure:
        with pytest.raises(RuntimeError):
            execute()
    else:
        execute()
    assert [x.projection_graph_enabled for x in indexers] == [True, False]
    indexers[0].projection_graph = None
    with pytest.raises(ValueError, match="captured"), graph_mode(engine, True):
        pass


@pytest.mark.parametrize("error", ["signature", "reference", "disabled", "native_changed"])
def test_benchmark_rejects_unqualified_reference_or_modes_before_allocation(tmp_path, monkeypatch, error):
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=object()))
    config, reference = tmp_path / "runtime.json", tmp_path / "reference.json"
    config.write_text(
        json.dumps(
            dict(
                cuda_indexer_hadamard=True,
                cuda_indexer_projection_graphs=error != "disabled",
                direct_dma=dict(npu_single_route_graphs=error == "native_changed"),
            )
        )
    )
    reference.write_text(json.dumps(dict(passed=error != "reference")))
    signature = hashlib.sha256(reference.read_bytes()).hexdigest() if error != "signature" else "0" * 64
    with pytest.raises(ValueError):
        run(config, tmp_path / "private.json", reference, signature)
