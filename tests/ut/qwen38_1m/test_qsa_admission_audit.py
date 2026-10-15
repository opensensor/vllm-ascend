# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic failure isolation and bounded, pointer-free QSA audit receipts."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ut.qwen38_1m.test_qsa_graph_dispatch_candidate import metadata
from tools.qwen4exp.resident_candidates import qsa_admission_audit as candidate
from tools.qwen4exp.resident_candidates.qsa_graph_dispatch import PrefixGeometry


def setup(monkeypatch):
    boundaries = metadata()
    context = SimpleNamespace(batch_descriptor="three", attn_metadata={"qsa": boundaries})
    wrapper = SimpleNamespace(
        entries={},
        runnable=object(),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=262144)),
    )
    monkeypatch.setattr(candidate, "discover_geometries", lambda runnable, context: {"qsa": PrefixGeometry()})
    calls = []

    def invoke():
        calls.append("model")
        return "original output"

    return boundaries, context, wrapper, calls, invoke


def test_audit_records_capture_then_changed_real_metadata_without_blocking(monkeypatch):
    boundaries, context, wrapper, calls, invoke = setup(monkeypatch)
    assert candidate.audit_call(wrapper, context, (), {}, invoke) == "original output"
    wrapper.entries["three"] = SimpleNamespace(capture=object())
    boundaries.block_tables = boundaries.block_tables.clone()
    boundaries.seq_lens_cpu.fill_(2052)
    assert candidate.audit_call(wrapper, context, (), {}, invoke) == "original output"
    result = wrapper._qfn_qsa_admission_audit.snapshot()
    record = result["descriptors"][0]
    assert calls == ["model", "model"]
    assert record["capture_calls"] == record["replay_calls"] == 1
    assert record["first_success_phase"] == "capture"
    assert record["mismatches"] == {"block_tables_address": 1, "prefix_band": 1}
    assert record["first_mismatch"]["metadata"]["groups"][0]["seq_lens_cpu"]["values"] == [2052]
    serialized = json.dumps(result)
    assert str(boundaries.block_tables.data_ptr()) not in serialized
    assert '"first"' not in serialized


@pytest.mark.parametrize("field", ["seq_lens_cpu", "query_lens_cpu"])
def test_missing_cpu_lengths_record_error_and_always_delegate(monkeypatch, field):
    boundaries, context, wrapper, calls, invoke = setup(monkeypatch)
    setattr(boundaries, field, None)
    candidate.audit_call(wrapper, context, (), {}, invoke)
    record = wrapper._qfn_qsa_admission_audit.snapshot()["descriptors"][0]
    assert calls == ["model"] and record["errors"] == 1
    assert record["first_error"]["metadata"]["groups"][0][field] is None
    assert record["first_error"]["type"] == "ValueError"


def test_cpu_length_geometry_mismatch_is_observed_not_enforced(monkeypatch):
    boundaries, context, wrapper, calls, invoke = setup(monkeypatch)
    boundaries.query_lens_cpu.fill_(1)
    candidate.audit_call(wrapper, context, (), {}, invoke)
    record = wrapper._qfn_qsa_admission_audit.snapshot()["descriptors"][0]
    assert calls == ["model"]
    assert record["first_error"]["metadata"]["groups"][0]["query_lens_cpu"]["values"] == [1]
    assert "captured token geometry" in record["first_error"]["message"]


def test_discovery_and_audit_internal_failures_do_not_prevent_original_call(monkeypatch):
    _, context, wrapper, calls, invoke = setup(monkeypatch)
    monkeypatch.setattr(candidate, "discover_geometries", lambda *args: 1 / 0)
    candidate.audit_call(wrapper, context, (), {}, invoke)
    assert calls == ["model"]
    monkeypatch.setattr(candidate, "discover_geometries", lambda *args: {"qsa": PrefixGeometry()})
    candidate.audit_call(wrapper, context, (), {}, invoke)
    audit = wrapper._qfn_qsa_admission_audit
    monkeypatch.setattr(audit, "observe", lambda *args: 1 / 0)
    candidate.audit_call(wrapper, context, (), {}, invoke)
    assert calls == ["model"] * 3 and audit.internal_errors == 1


def test_original_model_error_is_preserved(monkeypatch):
    _, context, wrapper, _, _ = setup(monkeypatch)
    failure = RuntimeError("real model failure")

    def invoke():
        raise failure

    with pytest.raises(RuntimeError) as caught:
        candidate.audit_call(wrapper, context, (), {}, invoke)
    assert caught.value is failure


def test_descriptor_records_and_counters_are_bounded(monkeypatch):
    _, context, wrapper, calls, invoke = setup(monkeypatch)
    for index in range(candidate.MAX_DESCRIPTORS + 20):
        context.batch_descriptor = str(index)
        candidate.audit_call(wrapper, context, (), {}, invoke)
    result = wrapper._qfn_qsa_admission_audit.snapshot()
    assert len(result["descriptors"]) == candidate.MAX_DESCRIPTORS
    assert result["overflow_calls"] == 20 and len(calls) == candidate.MAX_DESCRIPTORS + 20
    assert candidate._increment(candidate.MAX_COUNTER) == candidate.MAX_COUNTER


def test_draft_step_model_input_pointer_and_layout_changes_are_classified(monkeypatch):
    _, context, wrapper, calls, invoke = setup(monkeypatch)
    first, second = torch.empty(3, 16), torch.empty(1, 16)
    candidate.audit_call(wrapper, context, (first,), {}, invoke)
    candidate.audit_call(wrapper, context, (second,), {}, invoke)
    categories = wrapper._qfn_qsa_admission_audit.snapshot()["descriptors"][0]["mismatches"]
    assert categories == {"model_input_address": 1, "model_input_layout": 1}
    assert len(calls) == 2


def test_device_length_metadata_never_gets_copied_to_cpu(monkeypatch):
    boundaries, context, wrapper, calls, invoke = setup(monkeypatch)
    boundaries.seq_lens_cpu = torch.empty(1, device="meta", dtype=torch.int32)
    candidate.audit_call(wrapper, context, (), {}, invoke)
    record = wrapper._qfn_qsa_admission_audit.snapshot()["descriptors"][0]
    assert calls == ["model"]
    assert record["first_error"]["metadata"]["groups"][0]["seq_lens_cpu"]["device"] == "meta"
    assert "values" not in record["first_error"]["metadata"]["groups"][0]["seq_lens_cpu"]


def test_payload_loads_through_actual_resident_source_loader(monkeypatch):
    from tools.glm_perf import resident_control

    wrapper_class = type("BreakableACLGraphWrapper", (), {"__call__": lambda self: None})
    worker_class = type("NPUWorker310", (), {"qfn_graph_inventory": lambda self: []})
    modules = {
        "vllm_ascend.compilation.breakable_aclgraph": SimpleNamespace(BreakableACLGraphWrapper=wrapper_class),
        "vllm_ascend._310p.worker_310p": SimpleNamespace(NPUWorker310=worker_class),
    }
    original_import = resident_control.importlib.import_module
    monkeypatch.setattr(
        resident_control.importlib, "import_module", lambda name: modules.get(name) or original_import(name)
    )
    control = resident_control.Control(
        generation="b" * 32,
        candidate="qsa_admission_audit",
        source=Path(candidate.__file__).read_text(),
    )
    assert {replacement.attribute for replacement in resident_control.prepare_replacements(control)} == {
        "__call__",
        "qfn_graph_inventory",
    }


def test_cold_graph_parameter_counts_do_not_read_device_values():
    assert candidate.summarize_graph_parameters(None) == {"initialized": False, "sizes": {}}
    parameters = SimpleNamespace(
        events={3: [object()]},
        handles={3: [], 6: [object(), object()]},
        attn_params={3: [], 6: []},
        workspaces={3: None, 6: torch.empty(3, device="meta")},
    )
    assert candidate.summarize_graph_parameters(parameters) == {
        "initialized": True,
        "sizes": {
            "3": {"events": 1, "handles": 0, "attn_params": 0, "workspace": False},
            "6": {"events": 0, "handles": 2, "attn_params": 0, "workspace": True},
        },
    }
