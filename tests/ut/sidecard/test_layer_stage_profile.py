# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch

from tools.sidecard.layer_math import TextLayer
from tools.sidecard.layer_stage_profile import StageProfile


@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("linear", [True, False])
@pytest.mark.parametrize("reduce", [True, False])
def test_profile_preserves_complete_middle_layer_math(tmp_path, monkeypatch, rows, linear, reduce):
    torch.manual_seed(319)
    width = 8
    weights = {}
    for kind in ("attn", "ffn"):
        weights[f"hc_{kind}_fn"] = torch.randn(24, width * 4)
        weights[f"hc_{kind}_scale"] = torch.tensor([0.1, 0.2, 0.3])
        weights[f"hc_{kind}_base"] = torch.randn(24)
    for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        weights[name] = torch.randn(width).half()
    weights["mlp.gate.weight"] = torch.randn(288, width)
    weights["mlp.gate.e_score_correction_bias"] = torch.randn(288)
    layer = object.__new__(TextLayer)
    layer.torch, layer.weights, layer.device = torch, weights, "npu:0"
    layer.index, layer.linear, layer.tp_rank = 18 if linear else 19, linear, 0 if reduce else None
    layer.native_sinkhorn = None
    layer.config = dict(
        num_hidden_layers=45, rms_norm_eps=1e-5, hc_eps=1e-6, hc_sinkhorn_iters=20, routed_scaling_factor=2.5
    )
    layer.kda = lambda x: (x.float() * 0.6).half()
    layer.dsa = lambda x: (x.float() * 0.7).half()
    layer.dense_mlp = lambda x, prefix: (x.float() * 0.3).half()
    routes = []

    def experts(index, x, ids, route_weights):
        routes.append((ids.tolist(), route_weights.tolist()))
        return x.float() * route_weights.sum(-1, keepdim=True)

    layer.experts = experts
    sums = []

    def all_reduce(value):
        sums.append(value.dtype)
        value.mul_(2)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    state = (
        torch.randn(rows, width).half(),
        (torch.randn(rows, 4, width) * 70000).bfloat16().float(),
        torch.rand(rows, 4, 1),
        torch.rand(rows, 4, 4),
    )
    expected = layer.forward(*state)
    reference_sums = list(sums)
    reference_routes = list(routes)
    sums.clear()
    routes.clear()
    syncs = []
    profile = StageProfile(tmp_path / "profile.jsonl", 0, lambda: syncs.append(True))
    profile.begin(5, 31, rows)
    actual = profile.forward(layer, state)
    profile.finish()
    for value, golden in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, golden, rtol=0, atol=0)
    assert sums == reference_sums
    assert routes == reference_routes
    record = json.loads(profile.output.read_text())
    assert record["layers"][0]["expert_ids"] == reference_routes[0][0]
    assert all(value >= 0 for value in record["layers"][0]["components_ms"].values())
    assert len(syncs) == len(record["layers"][0]["components_ms"]) * 2
    assert profile.packet is None


def test_profile_rejects_incomplete_or_nonmiddle_state(tmp_path):
    profile = StageProfile(tmp_path / "trace.jsonl", 0, lambda: None)
    layer = SimpleNamespace(index=0)
    with pytest.raises(ValueError, match="middle"):
        profile.forward(layer, (None,) * 4)
    layer = SimpleNamespace(index=18, torch=None, weights={}, config={})
    with pytest.raises(ValueError, match="complete"):
        profile.forward(layer, (None,) * 4)
