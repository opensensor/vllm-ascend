# SPDX-License-Identifier: Apache-2.0
"""Offline admission and lifetime checks for explicitly selected TP6 candidates."""

import json
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp import tp6_fusion_sources, tp6_fusions, tp6_operand_reuse
from tools.qwen4exp.build_tp6_fusions import ENTRIES


def test_failed_submission_keeps_operands_and_blocks_reuse():
    failure = RuntimeError("submission failed")

    def launch(*args):
        raise failure

    resource = tp6_fusions.EagerResource(object(), launch)
    operand = object()
    with pytest.raises(RuntimeError, match="submission failed"):
        resource.submit([operand])
    assert resource.retained == (operand,)
    assert resource.failure is failure
    with pytest.raises(RuntimeError, match="cannot be reused"):
        resource.preflight(torch.device("cpu"))


def test_graph_capture_refuses_candidate_before_submission(monkeypatch):
    resource = tp6_fusions.EagerResource(object(), lambda *args: pytest.fail("must not submit"))
    monkeypatch.setattr(tp6_fusions, "_capturing", lambda device: True)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="eager inference"):
        resource.preflight(torch.device("cpu"))


def test_build_identity_rejects_modified_binary(tmp_path):
    binaries = {}
    for name in ("native_tp6_gate_pack.bin", "native_tp6_finalize.bin"):
        data = name.encode()
        (tmp_path / name).write_bytes(data)
        binaries[name] = tp6_fusion_sources.digest(data)
    names = (
        "native_tp6_gate_pack.cpp",
        "native_tp6_finalize.cpp",
        "native_int4_schedule.h",
        "native_int4_route_groups.h",
        "qwen_w4_a8_int4_matmul_v310_tiling_data.h",
        "tp6_local_schedule.h",
        "tp6_local_pack.h",
    )
    sources = {}
    for name in names:
        (tmp_path / name).write_bytes(b"identity-test-fixture")
        sources[name] = tp6_fusion_sources.digest(b"identity-test-fixture")
    receipt = {"schema_version": 1, "sources": sources, "binaries": binaries, "entries": ENTRIES}
    (tmp_path / "provenance.json").write_text(json.dumps(receipt))
    assert tp6_fusions.verify_build(tmp_path) == receipt
    (tmp_path / "native_tp6_finalize.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity changed"):
        tp6_fusions.verify_build(tmp_path)


def test_unqualified_sources_cannot_generate_candidates():
    with pytest.raises(ValueError, match="unqualified INT4 schedule"):
        tp6_fusion_sources.private_schedule(b"changed schedule")
    with pytest.raises(ValueError, match="unqualified SwiGLU"):
        tp6_fusion_sources.local_pack(b"changed pack")


def test_cpu_operand_reuse_delegates_without_packing():
    value = torch.ones(3, 4)
    weight = torch.ones(1)
    module = SimpleNamespace(input_mix_weight_down=weight, input_mix_weight_up=weight, block_inject_weight=weight)
    expected = object()

    def fallback(instance, actual):
        assert instance is module and actual is value
        return expected

    mix = tp6_operand_reuse.make_mix(fallback, pack=lambda _: pytest.fail("CPU must not use NPU packing"))
    with torch.inference_mode():
        assert mix(module, value) is expected
