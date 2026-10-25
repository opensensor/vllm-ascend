# SPDX-License-Identifier: Apache-2.0
"""Dense conversion recovery and selection tests, independent of vLLM/NPU."""

import types

import pytest
import torch

from tools.glm_perf.resident_control import Control
from tools.minimax_m3 import dense_int8, resident_dense_int8
from tools.minimax_m3.dense_int8 import DenseInt8Method, convert, quantize_weight, restore, selected_projection


class ReferenceMethod:
    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.weight, bias)


def model_fixture():
    modules = []
    for suffix in ("self_attn.o_proj", "block_sparse_moe.shared_experts.gate_up_proj"):
        layer = torch.nn.Linear(128, 256, bias=False, dtype=torch.float16)
        layer.requires_grad_(False)
        layer.quant_method = ReferenceMethod()
        modules.append(("language_model.model.layers.3." + suffix, layer))
    return types.SimpleNamespace(named_modules=lambda: iter(modules)), modules


def test_convert_and_restore_exact_weights_and_methods():
    model, modules = model_fixture()
    originals = [(layer.weight.clone(), layer.quant_method) for _, layer in modules]
    state = {}
    reports = convert(model, state)
    assert len(reports) == 2
    assert all(r["nrmse"] < 0.03 and r["saved_bytes"] > 0 for r in reports)
    assert all(layer.weight.dtype == torch.int8 for _, layer in modules)
    with pytest.raises(ValueError, match="already active"):
        convert(model, state)
    restore(state)
    assert not state
    for (_, layer), (weight, method) in zip(modules, originals):
        assert torch.equal(layer.weight, weight)
        assert layer.weight.dtype == torch.float16
        assert layer.quant_method is method


def test_conversion_failure_restores_completed_and_current_layers(monkeypatch):
    model, modules = model_fixture()
    originals = [layer.weight.clone() for _, layer in modules]
    apply = DenseInt8Method.apply
    calls = []

    def fail_second(self, layer, x, bias=None):
        calls.append(layer)
        if len(calls) == 2:
            raise RuntimeError("injected projection failure")
        return apply(self, layer, x, bias)

    monkeypatch.setattr(DenseInt8Method, "apply", fail_second)
    state = {}
    with pytest.raises(RuntimeError, match="injected"):
        convert(model, state)
    assert not state
    assert all(torch.equal(layer.weight, original) for (_, layer), original in zip(modules, originals))
    assert all(isinstance(layer.quant_method, ReferenceMethod) for _, layer in modules)


def test_restore_allocation_failure_keeps_remaining_recovery_records(monkeypatch):
    model, modules = model_fixture()
    originals = [layer.weight.clone() for _, layer in modules]
    state = {}
    convert(model, state)
    device_weight = dense_int8.device_weight
    calls = 0

    def fail_second(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected allocation failure")
        return device_weight(*args)

    monkeypatch.setattr(dense_int8, "device_weight", fail_second)
    with pytest.raises(RuntimeError, match="allocation failure"):
        restore(state)
    assert list(state) == [modules[1][0]]
    assert torch.equal(modules[0][1].weight, originals[0])
    assert modules[1][1].weight.dtype == torch.int8
    restore(state)
    assert not state
    assert all(torch.equal(layer.weight, original) for (_, layer), original in zip(modules, originals))


def test_conversion_releases_metadata_before_restoration(monkeypatch):
    events = []
    codes = torch.ones(4, dtype=torch.int8)
    kernel = types.SimpleNamespace(requires_int8_layout=True, banks={1: (codes,)}, scales={2: torch.ones(1)})
    unrelated = types.SimpleNamespace(banks={1: "keep"})
    monkeypatch.setattr(
        torch,
        "npu",
        types.SimpleNamespace(
            synchronize=lambda: events.append("synchronize"), empty_cache=lambda: events.append("empty_cache")
        ),
        raising=False,
    )

    def restore_with_headroom(state):
        assert not kernel.banks and not kernel.scales
        assert events == ["idle", "synchronize", "empty_cache"]
        events.append("restore")

    monkeypatch.setattr(dense_int8, "restore", restore_with_headroom)
    replacements = resident_dense_int8.conversion_replacements(False, {"int8": kernel, "other": unrelated})
    worker = types.SimpleNamespace(_m3_idle=lambda: events.append("idle"), resident_status=lambda: {"rank": 0})
    result = next(iter(replacements.values()))(worker)
    assert result["dense_conversion"]["passed"]
    assert events[-1] == "restore"
    assert torch.equal(codes, torch.ones(4, dtype=torch.int8))
    assert unrelated.banks == {1: "keep"}


@pytest.mark.parametrize("suffix", ["self_attn.qkv_proj", "block_sparse_moe.gate", "self_attn.index_q_proj"])
def test_selection_excludes_sensitive_projections(suffix):
    assert not selected_projection("language_model.model.layers.3." + suffix)
    assert not selected_projection("vision_tower.encoder.layers.3.self_attn.o_proj")


def test_zero_weights_and_inputs_stay_finite():
    packed, scale = quantize_weight(torch.zeros(16, 32))
    layer = types.SimpleNamespace(weight=packed)
    method = DenseInt8Method(scale)
    output = method.apply(layer, torch.zeros(4, 32).half())
    assert torch.equal(output, torch.zeros(4, 16).half())
    with pytest.raises(ValueError, match="bias-free"):
        method.apply(layer, torch.zeros(4, 32).half(), torch.ones(16))


class Client:
    def __init__(self):
        self.paused = False
        self.resumed = False
        self.source = "def replacements(native_resources): return {}\n"
        control = Control("original", "direct-target", "m3_candidate", self.source)
        self.status = [
            {
                "rank": 0,
                "pid": 123,
                "generation": "original",
                "candidate": "m3_candidate",
                "digest": control.digest,
                "native_failed": False,
            }
        ]

    def request(self, path, **kwargs):
        if path == "/is_paused":
            return {"is_paused": self.paused}
        assert path.startswith("/pause?")
        self.paused = True

    def rpc(self, method):
        assert method == "resident_status"
        return self.status

    def switch(self, control):
        assert self.paused
        assert control.source == self.source
        return self.status

    def resume(self):
        self.paused, self.resumed = False, True


def test_transition_recovers_all_workers_after_one_conversion_failure(monkeypatch):
    client = Client()
    calls = []

    def mutate(client, enable):
        assert client.paused
        calls.append(enable)
        return [{"dense_conversion": {"passed": not enable}}]

    monkeypatch.setattr(resident_dense_int8, "mutate", mutate)
    with pytest.raises(RuntimeError, match="failed and recovered"):
        resident_dense_int8.transition(client, client.source, True)
    assert calls == [True, False]
    assert client.resumed


def test_transition_refuses_wrong_source_without_pausing():
    client = Client()
    with pytest.raises(ValueError, match="source must match"):
        resident_dense_int8.transition(client, client.source + "\n", True)
    assert not client.paused and not client.resumed


def test_uncertain_mutation_leaves_pause_in_place(monkeypatch):
    client = Client()

    def mutate(client, enable):
        raise RuntimeError("lost RPC acknowledgment")

    monkeypatch.setattr(resident_dense_int8, "mutate", mutate)
    with pytest.raises(RuntimeError, match="lost RPC"):
        resident_dense_int8.transition(client, client.source, True)
    assert client.paused and not client.resumed
