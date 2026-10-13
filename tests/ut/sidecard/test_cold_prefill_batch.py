# SPDX-License-Identifier: Apache-2.0
"""Cold batching retains causal CUDA chunks, tails and ordered owner sums."""

from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.dma_dispatch import DmaDispatch
from tools.sidecard.expert_actor import ExpertCoordinator
from tools.sidecard.layer_weights import expert_batch_size, prefill_window_size


@pytest.mark.parametrize("window", [32, 64, 128, 256, 512, 1024])
def test_window_configuration_keeps_cuda_chunk_size(window):
    spec = dict(expert_transport="direct-dma", prefill_chunk_size=32, prefill_window_size=window)
    assert prefill_window_size(spec) == window
    assert expert_batch_size(spec) == window


@pytest.mark.parametrize("window", [True, None, "64", 64.0, 0, 16, 96, 384, 2048])
def test_invalid_window_is_rejected(window):
    with pytest.raises(ValueError, match="window"):
        prefill_window_size(dict(expert_transport="direct-dma", prefill_window_size=window))


@pytest.mark.parametrize(
    "spec",
    [dict(prefill_window_size=64), dict(prefill_chunk_size=128, prefill_window_size=64, expert_transport="direct-dma")],
)
def test_window_requires_original_cuda_geometry_and_dma(spec):
    with pytest.raises(ValueError, match="32-row"):
        prefill_window_size(spec)


def test_dispatch_batches_cold_rows_but_preserves_hot_shared_geometry():
    dispatch = object.__new__(DmaDispatch)
    calls = []

    class Cuda:
        max_chunk = 32

        def __call__(self, index, x, ids, weights):
            calls.append(("hot", index, len(x), ids.clone(), weights.clone()))
            return x * 2

    dispatch.cuda, dispatch.torch, dispatch.max_chunk = Cuda(), torch, 64

    def shared(x, name):
        calls.append(("shared", name, len(x)))
        return x * 3

    layer = SimpleNamespace(index=18, dense_mlp=shared)

    def compute(index, x, ids, weights, shared_call, *, hot_call):
        calls.append(("cold", index, len(x)))
        return hot_call(ids, weights) + x * 5 + x * 7, shared_call()

    dispatch.compute = compute
    parts = [
        (
            torch.full((size, 2), value),
            None,
            None,
            None,
            torch.full((size, 8), value, dtype=torch.int32),
            torch.full((size, 8), 0.25),
        )
        for size, value in ((32, 1.0), (7, 2.0))
    ]
    results = list(dispatch.batch_with_shared(layer, parts))
    assert calls[0] == ("cold", 18, 39)
    assert [entry[2] for entry in calls[1:]] == [32, 7, 32, 7]
    for value, (routed, shared_value) in zip(parts, results, strict=True):
        assert torch.equal(routed, value[0] * 14)
        assert torch.equal(shared_value, value[0] * 3)
    with pytest.raises(ValueError, match="capacity"):
        dispatch.batch_with_shared(layer, parts * 2)
    with pytest.raises(ValueError, match="middle"):
        dispatch.batch_with_shared(SimpleNamespace(index=17), parts)


class CpuTorch:
    cuda = SimpleNamespace(synchronize=lambda: None)

    def __getattr__(self, name):
        return getattr(torch, name)

    def tensor(self, tokens, *, device):
        assert device == "cuda"
        return torch.tensor(tokens)


class CausalLayer:
    def __init__(self, index):
        self.index, self.memory, self.seen = index, torch.zeros(2), []

    def prepare_forward(self, x, residual, post, comb):
        self.seen.append(len(x))
        value = x.cumsum(0) + self.memory
        self.memory = value[-1].clone()
        return value, x.clone(), None, None

    def route(self, x):
        return torch.zeros((len(x), 8), dtype=torch.int32), torch.ones((len(x), 8))

    def finish_experts(self, x, residual, post, comb, routed, shared):
        return routed + shared + residual, None, None, None

    def forward(self, *state):
        prepared = self.prepare_forward(*state)
        return self.finish_experts(*prepared, prepared[0] * 2, prepared[0] * 3)


def make_engine():
    engine = object.__new__(ExpertCoordinator)
    engine.torch = CpuTorch()
    engine.position = engine.sequence = 0
    engine.stopping = engine.failed = False
    engine.max_context, engine.prefill_chunk_size, engine.prefill_window_size = 4096, 32, 64
    engine.config = {"rms_norm_eps": 1e-6}
    engine.globals = dict(embedding=torch.arange(4096).float().reshape(2048, 2), norm=torch.ones(2), head=torch.eye(2))
    engine.layers = {index: CausalLayer(index) for index in (0, 18, 19, 44)}
    engine.fingerprints = []
    engine.metrics = {"cuda_backbone_ms": []}
    calls = []

    def batch(layer, prepared):
        calls.append((layer.index, [len(value[0]) for value in prepared]))
        return [(value[0] * 2, value[0] * 3) for value in prepared]

    engine.experts = SimpleNamespace(batch_with_shared=batch)
    return engine, calls


@pytest.mark.parametrize("window", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize("length", [1, 32, 33, 64, 65, 99, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025])
def test_layer_major_windows_equal_serial_chunks_and_causal_states(length, window):
    candidate, calls = make_engine()
    baseline, _ = make_engine()
    candidate.prefill_window_size = window
    tokens = list(range(length))
    for offset in range(0, length, 32):
        expected = baseline.prefill_window(tokens[offset : offset + 32])
    actual = candidate.prefill(tokens)
    assert torch.equal(expected, actual)
    assert candidate.fingerprints == baseline.fingerprints
    assert candidate.position == baseline.position == length
    assert candidate.sequence == baseline.sequence
    for index, layer in candidate.layers.items():
        assert torch.equal(layer.memory, baseline.layers[index].memory)
        assert layer.seen == baseline.layers[index].seen
    assert all(sum(sizes) <= window for _, sizes in calls)


def test_failed_window_poisons_coordinator_and_cancellation_is_at_window_boundary():
    engine, calls = make_engine()
    cancel = SimpleNamespace(is_set=lambda: bool(calls))
    with pytest.raises(InterruptedError, match="windows"):
        engine.prefill(list(range(99)), cancelled=cancel)
    assert engine.position == 64 and not engine.failed

    def fail(*args):
        raise RuntimeError("native failure")

    engine.experts.batch_with_shared = fail
    with pytest.raises(RuntimeError, match="native"):
        engine.prefill(list(range(64)))
    assert engine.failed and engine.position == 64
