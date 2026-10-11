# SPDX-License-Identifier: Apache-2.0
"""Full-stage coverage and memory admission before any device allocation."""

import json
from types import SimpleNamespace

import pytest
import torch
from test_channel_scale_checkpoint import original_source
from test_channel_scale_runtime import CpuLaunch, bundle

from tools.glm_perf import channel_scale_runtime, channel_scale_stage
from tools.glm_perf.channel_scale_checkpoint import export
from tools.glm_perf.channel_scale_layout import logical_bank
from tools.glm_perf.channel_scale_math import fused_moe_reference
from tools.glm_perf.channel_scale_stage import plan_stage, prepare_stage


class Stage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(1))
        self.moe = torch.nn.Module()
        self.moe.w2_experts = SimpleNamespace(
            layer_key="layers.3", local_expert_offset=0, num_local_experts=2, grouped_ready=False
        )
        self.moe.routed_experts_forward = lambda: None


def overlay(tmp_path):
    source = tmp_path / "source"
    original_source(source)
    root = tmp_path / "overlay"
    export(source, root, [3], [0, 1], 256, 256)
    return root


def test_stage_rejects_incomplete_coverage_and_live_banks_before_allocation(tmp_path):
    root = overlay(tmp_path)
    model = Stage()
    owners, plan = plan_stage(model, root, max_tokens=640, top_k=8)
    assert len(owners) == 1 and plan["candidate_peak_bytes"] > plan["resident_expert_bytes"]
    build = tmp_path / "build"
    build.mkdir()
    bundle(build)
    with pytest.raises(ValueError, match="reservation"):
        prepare_stage(model, root, build, max_tokens=640, top_k=8, host_budget_bytes=1, device_budget_bytes=1)
    model.moe.w2_experts.grouped_ready = True
    with pytest.raises(ValueError, match="cold loading"):
        plan_stage(model, root, max_tokens=640, top_k=8)
    model.moe.w2_experts.grouped_ready = False
    manifest = json.loads((root / "manifest.json").read_text())
    del manifest["experts"]["3:1"]
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="every local"):
        plan_stage(model, root, max_tokens=640, top_k=8)


def test_cold_stage_binds_new_pipeline_and_retains_shared_expert_outer_contract(tmp_path, monkeypatch):
    root = overlay(tmp_path)
    kernels = tmp_path / "kernels"
    kernels.mkdir()
    bundle(kernels)
    simulator = CpuLaunch()

    class CpuNative(channel_scale_runtime.NativeChannelScaleMoE):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, kernel_factory=lambda path, name: name, launch=simulator)

    helper = SimpleNamespace(
        ScratchArena=channel_scale_runtime.ScratchArena,
        NativeChannelScaleMoE=CpuNative,
        ChannelScaleMoEMethod=channel_scale_runtime.ChannelScaleMoEMethod,
    )
    monkeypatch.setattr(channel_scale_stage, "frozen_runtime", lambda root: helper)
    model = Stage()
    report = prepare_stage(
        model, root, kernels, max_tokens=640, top_k=8, host_budget_bytes=16 * 1024**2, device_budget_bytes=128 * 1024**2
    )
    assert report["legacy_expert_tensors_loaded"] == report["runtime_weight_transforms"] == 0
    owner = model.moe.w2_experts
    owner.finalize_grouped_storage()  # original GLM loader's finalization seam
    assert not owner.grouped_ready and owner.layer_key == "layers.3"
    x = torch.randn(4, 256).half()
    ids = torch.tensor([1, 0, 1, 0, 999, 999, 0, 1]).expand(4, 8).clone()
    weights = torch.full((4, 8), 0.125)
    layer = SimpleNamespace(w2_experts=owner)
    actual = model.moe._method.apply(layer, x, weights, ids, None, None)
    banks = channel_scale_stage.load_partition(root, 3, 0, 2, host_budget_bytes=16 * 1024**2)
    expected = fused_moe_reference(x, *[logical_bank(b) for b in banks], weights, ids)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("capacity", [16, 640, 1280])
def test_stage_reserves_all_decode_slots_and_full_glm_scratch_growth(tmp_path, capacity):
    root = overlay(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    for entry in manifest["experts"].values():
        entry.update(gate_shape=[1, 4096, 4096], down_shape=[1, 4096, 2048], tensor_bytes=13393920)
    (root / "manifest.json").write_text(json.dumps(manifest))
    _, ledger = plan_stage(
        Stage(),
        root,
        max_tokens=capacity,
        top_k=8,
        decode_bits=8,
        prefill_bits=4,
        fused_gate_up=True,
        fused_decode=True,
        fused_prefill=True,
        identity_smoothing=True,
    )
    geometry = channel_scale_runtime.Geometry(2, 4096, 2048, 256, 256, True, True, True)
    arena = channel_scale_runtime.ScratchArena(ledger["scratch_bytes"])
    for count in range(1, 17):
        arena.get(count, 8, geometry, 8, "cpu")
    if capacity > 16:
        arena.get(capacity, 8, geometry, 4, "cpu")
        arena.get(37, 8, geometry, 4, "cpu")
    assert arena.allocated_bytes == ledger["scratch_bytes"]
    expected_growth = channel_scale_runtime.shape_bytes(
        channel_scale_runtime.scratch_shapes(capacity, 8, geometry, 8 if capacity <= 16 else 4)
    )
    assert ledger["scratch_growth_reserve_bytes"] == expected_growth
    assert ledger["disk_expert_bytes"] - ledger["resident_expert_bytes"] == 2 * (4096 + 2048) * 4


def test_operator_plan_admits_separate_decode_and_prefill_precision_without_backend(tmp_path):
    from tools.glm_perf.probe_channel_scale import plan

    root = overlay(tmp_path)
    kernels = tmp_path / "kernels"
    kernels.mkdir()
    bundle(kernels, fused=True, decode=True, prefill=True)
    _, _, all_a8 = plan(root, kernels, 3, 0, 2, (1, 37), 8, 8)
    _, _, mixed = plan(root, kernels, 3, 0, 2, (1, 37), 4, 8)
    assert mixed["decode_bits"] == 4 and mixed["prefill_bits"] == 8
    assert mixed["scratch_bytes"]["1"] < all_a8["scratch_bytes"]["1"]
    assert mixed["scratch_bytes"]["37"] == all_a8["scratch_bytes"]["37"]
    assert mixed["status"] == "CPU admission only"
