# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Private cold-variant probe with actual metadata inspector and observer."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tools.qwen4exp.resident_candidates import qfn_cold_variants_probe as probe
from tools.qwen4exp.resident_candidates import qsa_cold_variants as variants
from tools.qwen4exp.resident_candidates.qsa_graph_dispatch import PrefixGeometry, inspect_admission
from tools.sidecard.npu_collective_graph import CollectiveGraphObserver


def setup_probe(*, double_replay=False, mutate_parameter=False):
    metadata = probe.private_metadata(torch, "cpu")
    context = SimpleNamespace(
        batch_descriptor=probe.ProbeDescriptor(3), attn_metadata={"probe.qsa": metadata}, capturing=False
    )
    wrapper = SimpleNamespace(entries={})
    facade = SimpleNamespace(state="active", owners={}, fallbacks={}, _context=lambda: None)
    observer = CollectiveGraphObserver(facade, track_replays=False)
    controller = variants.ColdVariants({"probe.qsa": PrefixGeometry()}, observer)
    inputs, parameter, counter = torch.zeros(3), torch.tensor([2.0, 4.0, 8.0]), torch.zeros(1)
    captures = []
    replayed = []

    class Graph:
        capture_token = None
        num_graphs = 1
        num_eager_breaks = 0

        def __init__(self, variant):
            self.variant = variant
            self.output = torch.empty_like(inputs)
            self.resets = 0
            capture_metadata = context.attn_metadata["probe.qsa"]
            self.records_producer = probe.DERIVED_CACHE_KEY not in getattr(
                capture_metadata, "_gdn_recurrent_step_meta", {}
            )
            self.derived = probe.cached_product(capture_metadata, inputs, parameter)

        def replay(self):
            replayed.append(self.variant)
            counter.add_(2.0 if double_replay else 1.0)
            offset = probe.VARIANT_OFFSETS[0 if self.variant == "prefix" else 1]
            if self.records_producer:
                self.derived.copy_(inputs * parameter)
            self.output.copy_(self.derived + offset)
            if mutate_parameter:
                parameter.add_(1)

        def reset(self):
            self.resets += 1

    def capture(key, variant):
        graph = Graph(variant)
        captures.append(graph)
        with observer.capture_scope(graph):
            pass
        entry = SimpleNamespace(capture=graph, output=graph.output)
        wrapper.entries[key] = entry
        return entry, entry.output

    def replay(entry):
        entry.capture.replay()
        return entry.output

    eager = Mock(side_effect=AssertionError("private gate must not run eager"))

    def invoke(cold=False):
        return controller.call(
            wrapper, context, (inputs,), {}, capture_allowed=cold, capture=capture, replay=replay, eager=eager
        )

    invoke(True)
    return SimpleNamespace(**locals())


def test_actual_inspector_changes_mode_without_changing_binding():
    metadata = probe.private_metadata(torch, "cpu")
    context = SimpleNamespace(attn_metadata={"probe.qsa": metadata})
    geometry = {"probe.qsa": PrefixGeometry()}
    before = inspect_admission(context, geometry, (), {})
    metadata.seq_lens_cpu.fill_(2052)
    after = inspect_admission(context, geometry, (), {})
    assert before.prefix_eligible and not after.prefix_eligible
    assert before.binding == after.binding


def test_private_gate_short_long_short_changes_inputs_and_only_replays():
    gate = setup_probe()
    before = gate.observer.snapshot()
    records = probe.run_switches(torch, gate.metadata, gate.inputs, gate.parameter, gate.counter, gate.invoke)
    assert [record["variant"] for record in records] == ["prefix", "general", "prefix", "general", "prefix"]
    assert gate.replayed == [record["variant"] for record in records]
    assert [record["replay_increments"] for record in records] == [1, 2, 3, 4, 5]
    assert records[0]["values"] == [13, 19, 35]
    assert records[1]["values"] == [35, 49, 85]
    assert records[2]["values"] == [7, 27, 59]
    assert len(gate.captures) == len(gate.wrapper.entries) == 2
    assert all(graph.records_producer for graph in gate.captures)
    assert len({graph.derived.data_ptr() for graph in gate.captures}) == 2
    assert not hasattr(gate.metadata, "_gdn_recurrent_step_meta")
    assert gate.observer.snapshot() == before
    assert before["registered_graphs"] == 2 and before["native_leases_acquired"] == 0
    assert all(graph.resets == 0 for graph in gate.captures)
    gate.eager.assert_not_called()
    receipt = gate.observer.retire(lambda: None)
    assert receipt["retired"] and receipt["reset_graphs"] == 2
    assert receipt["native_leases_released"] == 0
    assert all(graph.resets == 1 for graph in gate.captures)


def test_shared_metadata_omits_prefix_producer_and_fails_changing_input_gate(monkeypatch):
    monkeypatch.setattr(variants, "isolate_capture_metadata", lambda metadata: metadata)
    gate = setup_probe()
    assert [graph.records_producer for graph in gate.captures] == [True, False]
    assert gate.captures[0].derived is gate.captures[1].derived
    with pytest.raises(AssertionError):
        probe.run_switches(torch, gate.metadata, gate.inputs, gate.parameter, gate.counter, gate.invoke)
    # General replay temporarily repairs the prefix dependency, but a new
    # prefix input still consumes the last general product without its producer.
    gate.inputs.copy_(torch.tensor([1.0, 2.0, 3.0]))
    gate.metadata.seq_lens_cpu.fill_(2052)
    gate.invoke()
    gate.metadata.seq_lens_cpu.fill_(512)
    torch.testing.assert_close(gate.invoke(), gate.inputs * gate.parameter + probe.VARIANT_OFFSETS[0])
    gate.inputs.add_(1)
    assert not torch.equal(gate.invoke(), gate.inputs * gate.parameter + probe.VARIANT_OFFSETS[0])


def test_probe_rejects_double_execution():
    gate = setup_probe(double_replay=True)
    with pytest.raises(RuntimeError, match="increments"):
        probe.run_switches(torch, gate.metadata, gate.inputs, gate.parameter, gate.counter, gate.invoke)


def test_probe_rejects_private_parameter_mutation():
    gate = setup_probe(mutate_parameter=True)
    with pytest.raises((RuntimeError, AssertionError)):
        probe.run_switches(torch, gate.metadata, gate.inputs, gate.parameter, gate.counter, gate.invoke)


def test_rpc_factory_installs_only_probe_not_qsa(monkeypatch):
    module = ModuleType("vllm_ascend._310p.worker_310p")
    module.NPUWorker310 = SimpleNamespace(qfn_cpu_profile_stop=Mock())
    monkeypatch.setitem(sys.modules, module.__name__, module)
    blocked = Mock(side_effect=AssertionError("QSA factory must not run"))
    monkeypatch.setattr(variants, "make_replacements", blocked)
    result = probe.replacements()
    assert list(result) == ["vllm_ascend._310p.worker_310p:NPUWorker310.qfn_cpu_profile_stop"]
    blocked.assert_not_called()
