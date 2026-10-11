# SPDX-License-Identifier: Apache-2.0
"""Head sum happens after per-head ReLU and before full-column top-k."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf.indexer_heads import bind_indexer, memory_plan, read_options, reduced_score


def kpool_module(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/glm5next/kpool_ops.py"
    spec = importlib.util.spec_from_file_location("head_test_kpool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "vllm_ascend.models.glm5next", SimpleNamespace(kpool_ops=module))
    return module


def test_full_score_reduction_precedes_topk_and_keeps_negative_head_weights():
    # Pool 2 is not either partition's top-1, but wins after head summation.
    partials = [torch.tensor([[10.0, 0.0, 6.0]]), torch.tensor([[0.0, 10.0, 6.0]])]
    expected = sum(partials)
    backend = SimpleNamespace(all_reduce=lambda value, group: value.add_(partials[1]))
    score = reduced_score(lambda *args: partials[0].clone(), group="tp-stage", backend=backend)
    result = score(None, None, None)
    assert torch.equal(result, expected)
    assert result.argmax().item() == 2
    negative = torch.tensor([[3.0, -7.0, 2.0]])
    score = reduced_score(
        lambda *args: negative.clone(),
        group="tp-stage",
        backend=SimpleNamespace(all_reduce=lambda value, group: value.add_(1)),
    )
    assert torch.equal(score(None, None, None), negative + 1)


@pytest.mark.parametrize("size", [2, 4])
def test_score_budget_covers_complete_columns_and_prepared_weights(size):
    plan = memory_plan(size=size, max_tokens=640, max_model_len=8192, score_elements=1 << 20, indexers=4)
    assert plan["local_heads"] == 32 // size
    assert plan["score_rows"] < 640
    assert plan["required_bytes"] > plan["prepared_weight_bytes"]
    with pytest.raises(ValueError, match="complete"):
        memory_plan(size=size, max_tokens=640, max_model_len=8192, score_elements=1)


def test_modes_are_mutually_exclusive_and_budgets_explicit():
    assert read_options({}) is None
    with pytest.raises(ValueError, match="explicit"):
        read_options(dict(indexer_head_sharding=True))
    with pytest.raises(ValueError, match="alternative"):
        read_options(
            dict(
                indexer_head_sharding=True,
                indexer_head_budget_bytes=10**8,
                indexer_score_elements=1 << 20,
                indexer_owner_rank=0,
            )
        )


class Select:
    def __init__(self):
        self.calls = []

    def _select_tokens(self, queries, weights, keys):
        return score_kpool(queries, weights, keys)  # noqa: F821 - production namespace is replaced

    def _select_tokens_fixed(self, queries, weights, keys):
        return score_kpool(queries, weights, keys)  # noqa: F821

    def forward_oot(self, hidden, queries, keys, weights, **kwargs):
        self.calls.append((queries.clone(), keys.clone(), weights.clone(), kwargs["gate_score"].clone()))
        return queries


class Indexer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_head, self.head_dim, self.rope_dim = 32, 128, 0
        self.wq_b = torch.nn.Linear(7, 4096, bias=False)
        self.wk_weights_proj = torch.nn.Linear(5, 160, bias=False)
        self.index_kpool_compress_gate = torch.nn.Parameter(torch.randn(128, 5))
        self.index_kpool_compress_ape = torch.randn(4, 128)
        self.k_norm = torch.nn.LayerNorm(128)
        self.softmax_scale, self.index_kpool = 128**-0.5, 4
        self.indexer_op = Select()
        self.decode_calls = 0

    def forward(self, hidden, qr, positions, rotary):
        self.decode_calls += 1
        return hidden


@pytest.mark.parametrize("size,rank", [(2, 1), (4, 3)])
def test_shards_loaded_projection_keeps_global_normalization_and_decode_unchanged(monkeypatch, size, rank):
    kpool_module(monkeypatch)
    torch.manual_seed(51)
    indexer = Indexer()
    hidden, qr = torch.randn(37, 5), torch.randn(37, 7)
    start, stop = rank * (32 // size), (rank + 1) * (32 // size)
    reference_q = indexer.wq_b(qr).reshape(37, 32, 128)[:, start:stop].bfloat16()
    reference_kw = indexer.wk_weights_proj(hidden)
    factor = indexer.softmax_scale * 32**-0.5
    bind_indexer(indexer, rank=rank, size=size, group="tp", score_elements=1 << 20)
    assert torch.equal(indexer(hidden[:4], qr[:4], torch.arange(4), None), hidden[:4])
    assert indexer.decode_calls == 1
    result = indexer(hidden, qr, torch.arange(37), None)
    torch.testing.assert_close(result, reference_q, rtol=0, atol=0)
    q, keys, weights, gates = indexer.indexer_op.calls[-1]
    torch.testing.assert_close(weights, reference_kw[:, 128 + start : 128 + stop] * factor, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(keys, indexer.k_norm(reference_kw[:, :128]))
    torch.testing.assert_close(gates, hidden @ indexer.index_kpool_compress_gate.t())
    assert indexer.n_head == 32 and indexer.wq_b.weight.shape == (4096, 7)
    with pytest.raises(ValueError, match="cold-load"):
        bind_indexer(indexer, rank=rank, size=size, group="tp", score_elements=1 << 20)
