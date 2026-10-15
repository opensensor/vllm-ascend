# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact CPU semantics and mocked NPU index dispatch; no hardware claims."""

from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp.resident_candidates import qfn_compact_topk_probe as probe
from tools.qwen4exp.resident_candidates import qsa_compact_topk as candidate


def qualified_original(scores, k):
    if k <= 0 or k > scores.shape[1]:
        raise ValueError("k must be in [1, scores.shape[1]]")
    selected = torch.topk(scores, k, dim=1, sorted=False).indices
    index_order = torch.argsort(selected.to(torch.float32), dim=1, stable=True)
    selected = selected.gather(1, index_order)
    selected_values = scores.gather(1, selected)
    value_order = torch.argsort(selected_values, dim=1, descending=True, stable=True)
    return selected.gather(1, value_order)


@pytest.mark.parametrize("rows,width,k", [(1, 8, 1), (3, 128, 32), (64, 1024, 512), (64, 2048, 512)])
@pytest.mark.parametrize("pattern", ["random", "ties", "infinities"])
def test_cpu_keeps_exact_int64_values_order_and_input(rows, width, k, pattern):
    scores = probe.score_case(rows, width, pattern, generator=torch.Generator().manual_seed(42))
    before = scores.clone()
    expected = qualified_original(scores, k)
    actual = candidate.compact_topk_indices(scores, k)
    assert actual.dtype == torch.int64
    assert torch.equal(actual, expected)
    assert torch.equal(scores, before)


class FakeNPUTensor:
    """Run math on CPU while recording the dtype passed to NPU gathers."""

    def __init__(self, value, calls):
        self.value = value
        self.calls = calls
        self.shape = value.shape
        self.device = SimpleNamespace(type="npu")

    def to(self, dtype):
        self.calls.append(("cast", self.value.dtype, dtype))
        return FakeNPUTensor(self.value.to(dtype), self.calls)

    def gather(self, dim, indices):
        self.calls.append(("gather", indices.value.dtype))
        # CPU gather requires INT64; record the actual supplied type first.
        return FakeNPUTensor(self.value.gather(dim, indices.value.long()), self.calls)


class FakeNPUTorch:
    int32 = torch.int32
    float32 = torch.float32

    def __init__(self, calls, int32_max):
        self.calls = calls
        self.int32_max = int32_max

    def iinfo(self, dtype):
        assert dtype == torch.int32
        return SimpleNamespace(max=self.int32_max)

    def topk(self, scores, k, **kwargs):
        self.calls.append(("topk", scores.shape, k, kwargs))
        return SimpleNamespace(indices=FakeNPUTensor(torch.topk(scores.value, k, **kwargs).indices, self.calls))

    def argsort(self, value, **kwargs):
        self.calls.append(("argsort", value.value.dtype, kwargs))
        return FakeNPUTensor(torch.argsort(value.value, **kwargs), self.calls)


@pytest.mark.parametrize("pattern", ["random", "ties", "infinities"])
@pytest.mark.parametrize("compact", [True, False])
def test_npu_only_index_casts_preserve_original_calls_and_exact_order(monkeypatch, pattern, compact):
    scores = probe.score_case(6, 128, pattern, generator=torch.Generator().manual_seed(99))
    expected = qualified_original(scores, 32)
    calls = []
    # A synthetic smaller index limit exercises the enormous-width fallback
    # without allocating a tensor with billions of columns.
    shim = FakeNPUTorch(calls, torch.iinfo(torch.int32).max if compact else 63)
    monkeypatch.setattr(candidate, "torch", shim)
    actual = candidate.compact_topk_indices(FakeNPUTensor(scores, calls), 32)
    expected_dtype = torch.int32 if compact else torch.int64
    assert actual.value.dtype == expected_dtype
    assert torch.equal(actual.value.long(), expected)
    assert [call for call in calls if call[0] == "topk"] == [("topk", scores.shape, 32, {"dim": 1, "sorted": False})]
    assert [call for call in calls if call[0] == "gather"] == [("gather", expected_dtype)] * 3
    assert [call for call in calls if call[0] == "argsort"] == [
        ("argsort", torch.float32, {"dim": 1, "stable": True}),
        ("argsort", torch.float32, {"dim": 1, "descending": True, "stable": True}),
    ]
    casts = [call for call in calls if call[0] == "cast"]
    if compact:
        assert casts == [
            ("cast", torch.int64, torch.int32),
            ("cast", torch.int32, torch.float32),
            ("cast", torch.int64, torch.int32),
            ("cast", torch.int64, torch.int32),
        ]
    else:
        assert casts == [("cast", torch.int64, torch.float32)]


@pytest.mark.parametrize("k", [-1, 0, 9])
def test_invalid_topk_rejected_before_dispatch(k):
    with pytest.raises(ValueError, match="k must be"):
        candidate.compact_topk_indices(torch.empty(2, 8), k)


def test_factory_only_replaces_fast_policy_helper():
    assert candidate.replacements() == {
        "vllm_ascend.models.qwen4_exp.ops.qsa_indexer:_fast_topk_indices": candidate.compact_topk_indices
    }


def test_exact_gate_accepts_int32_and_rejects_changed_order_and_float_indices():
    expected = torch.tensor([[7, 2, 3]])
    probe.assert_exact_indices(expected.int(), expected)
    with pytest.raises(AssertionError, match="values or order"):
        probe.assert_exact_indices(torch.tensor([[2, 7, 3]], dtype=torch.int32), expected)
    with pytest.raises(AssertionError, match="shape or dtype"):
        probe.assert_exact_indices(expected.float(), expected)


def test_probe_rpc_uses_no_argument_and_delegates_output_path():
    calls = []
    worker = SimpleNamespace(rank=2)
    report = {"passed": True}
    rpc = probe.make_rpc(lambda self, output: calls.append((self, output)) or "original", probe=lambda: report)
    assert rpc(worker) == {"rank": 2, "qsa_compact_topk": report}
    assert rpc(worker, "/tmp/profile.json") == "original"
    assert calls == [(worker, "/tmp/profile.json")]


@pytest.mark.parametrize("profiler", ["_qfn_cpu_profile", "_qfn_native_profile_session"])
def test_probe_rpc_rejects_profiler_ownership(profiler):
    worker = SimpleNamespace(rank=3, **{profiler: object()})
    rpc = probe.make_rpc(lambda *args: pytest.fail("unexpected delegation"), probe=lambda: pytest.fail("ran gate"))
    result = rpc(worker)
    assert result["rank"] == 3 and not result["qsa_compact_topk"]["passed"]
    assert "profiling is active" in result["qsa_compact_topk"]["error"]


def test_probe_rpc_reports_rank_failure():
    def fail():
        raise RuntimeError("changed order")

    rpc = probe.make_rpc(lambda *args: None, probe=fail)
    assert rpc(SimpleNamespace(rank=1)) == {
        "rank": 1,
        "qsa_compact_topk": {"passed": False, "error": "RuntimeError: changed order"},
    }
