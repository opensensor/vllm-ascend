# SPDX-License-Identifier: Apache-2.0
"""Portable graph scheduling/state tests; no claims about Ascend arithmetic."""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard import npu_dsa_probe
from tools.sidecard.layer_math import HIDDEN, STREAMS
from tools.sidecard.npu_dsa_decode_graph import CaptureCleanupError, NpuDsaDecodeGraph
from tools.sidecard.npu_dsa_probe import DEFAULT_LENGTHS, configuration, pair_agreement, sha256, state_receipt


def fake_torch():
    def allocate(operation):
        return lambda *args, **kwargs: operation(*args, **dict(kwargs, device="cpu"))

    graphs = []

    class Graph:
        def __init__(self):
            self.resets = 0
            graphs.append(self)

        def reset(self):
            self.resets += 1

    @contextmanager
    def capture(graph):
        yield

    return SimpleNamespace(
        zeros=allocate(torch.zeros),
        ones=allocate(torch.ones),
        eye=allocate(torch.eye),
        float16=torch.float16,
        float32=torch.float32,
        npu=SimpleNamespace(synchronize=lambda: None, NPUGraph=Graph, graph=capture),
        graphs=graphs,
    )


class Layer:
    linear, device, tp_rank, index = False, "npu:0", 0, 19
    config = {"num_attention_heads": 4, "kv_lora_rank": 8}

    def __init__(self):
        self.cache = torch.zeros(64, 8).half()
        self.indexer = SimpleNamespace(
            pools=torch.zeros(16, 8).half(),
            tail_keys=torch.zeros(3, 8),
            tail_gates=torch.zeros(3, 8),
            pool_count=0,
            tail_length=0,
        )
        self.length = 0
        self.calls = []
        self.threshold = 8
        self.selected = None

    def reset(self):
        self.length = self.indexer.pool_count = self.indexer.tail_length = 0

    def prepare_attention(self, *state):
        self.calls.append("prepare")
        return state[0] + 1, *state[1:]

    def dsa_prepare(self, x):
        self.calls.append("qkv")
        return x[:, None, :8].expand(-1, 2, -1), x[:, :4], x[:, :8]

    def dsa_attend(self, x, qlatent, qr, kv):
        self.calls.append("eager-cache")
        count = len(x)
        self.cache[self.length : self.length + count].copy_(kv)
        self.length += count
        self.indexer.pool_count, self.indexer.tail_length = divmod(self.length, 4)
        pools, tail = self.indexer.pool_count, self.indexer.tail_length
        if pools:
            self.indexer.pools[:pools].copy_(self.cache[: pools * 4].reshape(pools, 4, 8).float().mean(1))
        if tail:
            self.indexer.tail_keys[:tail].copy_(self.cache[pools * 4 : self.length])
            self.indexer.tail_gates[:tail].copy_(self.cache[pools * 4 : self.length] * 2)
        self.selected = None if self.length <= self.threshold else torch.arange(self.length)[None]
        return qlatent + self.cache[: self.length].float().mean(0).half()

    def dsa_project(self, latent):
        self.calls.append("project")
        return latent.sum((1, 2))[:, None].expand(-1, HIDDEN).half().contiguous()

    def dsa(self, x):
        return self.dsa_project(self.dsa_attend(x, *self.dsa_prepare(x)))

    def finish_attention(self, *state):
        self.calls.append("attention-sum")
        return state

    def finish_forward(self, *state):
        self.calls.extend(("experts", "expert-sum"))
        return state

    def route(self, x):
        self.calls.append("route")
        return x, x


def inputs(rows=1, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(rows, HIDDEN, generator=generator).half(),
        torch.randn(rows, STREAMS, HIDDEN, generator=generator),
        torch.ones(rows, STREAMS, 1),
        torch.eye(STREAMS)[None].expand(rows, -1, -1).contiguous(),
    )


def emulated_capture(self, operation, buffers):
    computed = operation(buffers)
    outputs = tuple(value.clone() for value in computed) if isinstance(computed, tuple) else computed.clone()

    def replay():
        latest = operation(buffers)
        target_values = outputs if isinstance(outputs, tuple) else (outputs,)
        source_values = latest if isinstance(latest, tuple) else (latest,)
        for target, source in zip(target_values, source_values, strict=True):
            target.copy_(source)

    return SimpleNamespace(replay=replay, reset=lambda: None), buffers, outputs


@pytest.mark.parametrize("cold", [False, True])
def test_capture_keeps_cache_collectives_routing_and_experts_eager(cold):
    layer, backend, agrees = Layer(), fake_torch(), []
    graph = NpuDsaDecodeGraph(
        layer, backend, agree=lambda index, ready: agrees.append((index, ready)) or ready, cold=cold
    )
    assert layer.calls == ["prepare", "qkv"] * 3 + ["project"] * 3
    assert layer.length == layer.indexer.pool_count == layer.indexer.tail_length == 0
    assert agrees == [(19, True)] and graph.enabled
    graph.close()
    graph.close()
    assert [g.resets for g in backend.graphs] == [1, 1]


@pytest.mark.parametrize("length", [0, 6, 7, 8, 9, 10, 11, 12])
def test_replay_matches_eager_for_changed_positions_all_tails_and_reset(monkeypatch, length):
    monkeypatch.setattr(NpuDsaDecodeGraph, "capture", emulated_capture)
    reference, candidate = Layer(), Layer()
    graph = NpuDsaDecodeGraph(candidate, fake_torch(), agree=lambda *_: True)
    for repetition in range(2):
        reference.reset()
        candidate.reset()
        for owner in (reference, candidate):
            if length:
                state = inputs(length, seed=3)
                x, residual, post, comb = owner.prepare_attention(*state)
                owner.finish_attention(owner.dsa(x), residual, post, comb)
        for step, count in enumerate((1, 1, 1, 1, 2)):
            state = inputs(count, seed=step + repetition + 11)
            x, residual, post, comb = reference.prepare_attention(*state)
            expected = reference.finish_attention(reference.dsa(x), residual, post, comb)
            actual = graph.attention(state)
            assert state_receipt(reference, expected, reference.selected, torch) == state_receipt(
                candidate, actual, candidate.selected, torch
            )
    assert candidate.length == length + 6


@pytest.mark.parametrize("cold", [False, True])
def test_forward_preserves_collective_schedule_and_cold_contract(monkeypatch, cold):
    monkeypatch.setattr(NpuDsaDecodeGraph, "capture", emulated_capture)
    layer = Layer()
    graph = NpuDsaDecodeGraph(layer, fake_torch(), agree=lambda *_: True, cold=cold)
    layer.calls.clear()
    result = graph.forward(inputs())
    assert layer.calls == [
        "prepare",
        "qkv",
        "eager-cache",
        "project",
        "attention-sum",
        *(["route"] if cold else ["experts", "expert-sum"]),
    ]
    assert len(result) == (6 if cold else 4)


def test_peer_rejection_releases_both_local_graphs_and_retains_eager():
    layer, backend = Layer(), fake_torch()
    graph = NpuDsaDecodeGraph(layer, backend, agree=lambda *_: False)
    assert not graph.enabled and graph.before is graph.projected is None
    assert [g.resets for g in backend.graphs] == [1, 1]
    graph.forward(inputs())
    assert layer.length == 1


@pytest.mark.parametrize("failed_capture", [False, True])
def test_rejected_captures_discard_only_their_diagnostic_references(failed_capture):
    layer, backend, keys = Layer(), fake_torch(), {"valid-earlier-graph"}
    layer.execution_diagnostic = SimpleNamespace(
        capture_begin=lambda key: keys.add(key),
        capture_end=lambda key: None,
        discard_capture=lambda key: keys.discard(key),
    )
    if failed_capture:

        @contextmanager
        def reject(graph):
            yield
            raise RuntimeError("capture rejected")

        backend.npu.graph = reject
    graph = NpuDsaDecodeGraph(layer, backend, agree=lambda *_: False)
    assert not graph.enabled
    assert keys == {"valid-earlier-graph"}
    assert graph.diagnostic_graph_keys == {}


def test_second_capture_failure_releases_first_and_coordinates_fallback(monkeypatch):
    original = NpuDsaDecodeGraph.capture

    def fail(self, operation, buffers):
        if self.before is not None:
            raise RuntimeError("second capture failed")
        return original(self, operation, buffers)

    monkeypatch.setattr(NpuDsaDecodeGraph, "capture", fail)
    backend, ready = fake_torch(), []
    graph = NpuDsaDecodeGraph(Layer(), backend, agree=lambda _, value: ready.append(value) or value)
    assert ready == [False] and not graph.enabled
    assert [g.resets for g in backend.graphs] == [1]
    graph.forward(inputs())


def test_mutated_cache_during_capture_is_fatal(monkeypatch):
    def unsafe(self, *_):
        self.layer.length += 1
        raise RuntimeError("unsafe")

    monkeypatch.setattr(NpuDsaDecodeGraph, "capture", unsafe)
    with pytest.raises(RuntimeError, match="mutated attention state"):
        NpuDsaDecodeGraph(Layer(), fake_torch(), agree=lambda *_: False)


def test_failed_capture_cleanup_is_fatal(monkeypatch):
    def unsafe(self, *_):
        raise CaptureCleanupError("owner uncertain")

    monkeypatch.setattr(NpuDsaDecodeGraph, "capture", unsafe)
    with pytest.raises(CaptureCleanupError, match="owner uncertain"):
        NpuDsaDecodeGraph(Layer(), fake_torch(), agree=lambda *_: False)


def test_capture_context_failure_drains_and_resets_failed_graph():
    backend, ready = fake_torch(), []

    @contextmanager
    def failed_context(graph):
        yield
        raise RuntimeError("capture rejected")

    backend.npu.graph = failed_context
    graph = NpuDsaDecodeGraph(Layer(), backend, agree=lambda _, local: ready.append(local) or local)
    assert not graph.enabled and ready == [False]
    assert [g.resets for g in backend.graphs] == [1]


def test_failed_warmup_drain_refuses_fallback():
    backend = fake_torch()

    def failure():
        raise RuntimeError("stream cannot drain")

    backend.npu.synchronize = failure
    with pytest.raises(CaptureCleanupError, match="failed DSA warmup"):
        NpuDsaDecodeGraph(Layer(), backend, agree=lambda *_: False)


def test_input_mismatch_is_rejected_before_cache_append(monkeypatch):
    monkeypatch.setattr(NpuDsaDecodeGraph, "capture", emulated_capture)
    layer = Layer()
    graph = NpuDsaDecodeGraph(layer, fake_torch(), agree=lambda *_: True)
    state = inputs()
    with pytest.raises(ValueError, match="complete owner-local"):
        graph.forward((state[0].float(), *state[1:]))
    assert layer.length == 0


def test_replay_failure_never_retries_cache_append(monkeypatch):
    monkeypatch.setattr(NpuDsaDecodeGraph, "capture", emulated_capture)
    layer = Layer()
    graph = NpuDsaDecodeGraph(layer, fake_torch(), agree=lambda *_: True)

    def fail():
        raise RuntimeError("device replay failed")

    graph.projected[0].replay = fail
    with pytest.raises(RuntimeError, match="device replay failed"):
        graph.forward(inputs())
    assert layer.length == 1 and layer.calls.count("eager-cache") == 1


def test_replaced_cache_and_closed_owner_reject_before_side_effects(monkeypatch):
    monkeypatch.setattr(NpuDsaDecodeGraph, "capture", emulated_capture)
    layer = Layer()
    graph = NpuDsaDecodeGraph(layer, fake_torch(), agree=lambda *_: True)
    layer.cache = layer.cache.clone()
    with pytest.raises(RuntimeError, match="storage changed"):
        graph.forward(inputs())
    assert layer.length == 0
    graph.close()
    with pytest.raises(RuntimeError, match="closed"):
        graph.forward(inputs())


@pytest.mark.parametrize(
    "changes", [{"linear": True}, {"device": "cuda"}, {"tp_rank": None}, {"tp_rank": True}, {"index": 18}]
)
def test_reject_incompatible_owners_before_allocating(changes):
    layer = Layer()
    for name, value in changes.items():
        setattr(layer, name, value)
    with pytest.raises(ValueError, match="assigned TP2"):
        NpuDsaDecodeGraph(layer, None, agree=lambda *_: True)


@pytest.mark.parametrize("decision", [None, 1, "true"])
def test_pair_admission_requires_exact_boolean(decision):
    with pytest.raises(ValueError, match="invalid pair"):
        NpuDsaDecodeGraph(Layer(), fake_torch(), agree=lambda *_: decision)


def test_pair_schedule_rejects_mismatched_layers_and_combines_readiness():
    class Distributed:
        def all_reduce(self, value):
            value.add_(torch.tensor([0, 19, 19 * 19]))

    fake = SimpleNamespace(tensor=torch.tensor, float32=torch.float32, distributed=Distributed())
    agree = pair_agreement(fake, "cpu")
    assert not agree(19, True)
    with pytest.raises(RuntimeError, match="different layer schedules"):
        agree(23, True)


@pytest.fixture
def config(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    layout = root / "native-layout.json"
    layout.write_text("{}")
    library = tmp_path / "ops.so"
    library.write_bytes(b"frozen")
    return {
        "checkpoint": str(root),
        "checkpoint_sha256": sha256(layout),
        "ops_library": str(library),
        "ops_library_sha256": sha256(library),
        "physical_chips": [2, 3],
    }


def test_probe_validates_frozen_inputs_and_complete_threshold_schedule(config):
    result = configuration(config)
    assert result["lengths"] == list(DEFAULT_LENGTHS)
    assert result["max_context"] == 8192


@pytest.mark.parametrize(
    "changes",
    [
        {"physical_chips": [0, 1]},
        {"ops_library_sha256": "0" * 64},
        {"checkpoint_sha256": "0" * 64},
        {"lengths": [2048]},
        {"lengths": [True]},
        {"lengths": [{}]},
        {"steps": 1},
        {"repeats": 2},
        {"max_context": 2056},
        {"unknown": True},
    ],
)
def test_probe_rejects_unqualified_inputs_and_incomplete_coverage(config, changes):
    with pytest.raises(ValueError):
        configuration(dict(config, **changes))


@pytest.mark.parametrize("missing_rank", [None, 1])
def test_parent_launches_pair_before_waiting_and_requires_both_receipts(monkeypatch, config, tmp_path, missing_rank):
    source = tmp_path / "config.json"
    source.write_text(json.dumps(config))
    processes = []

    class Process:
        def __init__(self, args, **kwargs):
            self.rank = int(args[args.index("--worker-rank") + 1])
            self.output = Path(args[args.index("--output") + 1])
            processes.append(self)

        def wait(self):
            assert len(processes) == 2
            if self.rank != missing_rank:
                self.output.write_text(json.dumps({"ok": True, "rank": self.rank}))
            return 0

    monkeypatch.setattr(npu_dsa_probe.subprocess, "Popen", Process)
    destination = tmp_path / "result.json"
    report = npu_dsa_probe.run(source, destination)
    assert [process.rank for process in processes] == [0, 1]
    assert report["ok"] is (missing_rank is None)
    assert report["coordinated_tp2"] and not report["full_model_qualified"]
    assert report["same_math_comparison"] == "bitwise"
    with pytest.raises(FileExistsError):
        npu_dsa_probe.run(source, destination)
