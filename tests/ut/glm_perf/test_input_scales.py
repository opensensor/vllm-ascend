# SPDX-License-Identifier: Apache-2.0
"""Scalar input scales and their private producer/consumer descriptor contract."""

import json

import pytest
import torch

from tools.glm_perf.build_reconstruction import build
from tools.glm_perf.glm_fused_moe import FusedGeometry, NativeFusedMoE


def test_input_scale_dependency_rejected_before_build(tmp_path):
    with pytest.raises(ValueError, match="raw input scales require routed"):
        build(tmp_path / "build", tmp_path, tmp_path, raw_input_scales=True)
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize(
    "options,message", [({"raw_input_scales": 1}, "must be boolean"), ({"raw_input_scales": True}, "require routed")]
)
def test_input_scale_provenance_rejected_before_kernel_load(tmp_path, options, message):
    (tmp_path / "provenance.json").write_text(json.dumps({"_build": options}))
    with pytest.raises(ValueError, match=message):
        NativeFusedMoE(
            tmp_path, activation_bits=4, namespace="unused", kernel_factory=lambda *args: pytest.fail("partial load")
        )


@pytest.mark.parametrize("tokens,bits,scalar", [(16, 4, False), (17, 4, True), (640, 4, True), (640, 8, False)])
def test_input_scale_launch_bypass_and_token_descriptor(tokens, bits, scalar):
    n = NativeFusedMoE.__new__(NativeFusedMoE)
    n.device, n.activation_bits = torch.device("cpu"), bits
    n.raw_input_scales, n.fp16_route_workspace = True, True
    n.weight_lookup, n.configs, n.scratch = None, {}, {}
    n.pack_kernel, n.gate_kernel, n.down_kernel, n.reduce_kernel = "pack", "gate", "down", "reduce"
    n.route_input_kernel = "route"
    calls = []
    n.launch = lambda kernel, args, blocks: calls.append((kernel, args))
    g = FusedGeometry(tokens, 2, 3, 256, 256, 3, 3, bits)
    arguments = (
        torch.zeros(tokens, 256).half(),
        None,
        torch.ones(1),
        None,
        torch.ones(1),
        torch.ones(tokens, 2),
        torch.arange(tokens * 2),
        torch.tensor([0, 0, tokens * 2]),
        g,
    )
    n.grouped(*arguments)
    input_scales, descriptor = calls[0][1][3:5]
    assert input_scales.shape == ((tokens, 8) if scalar else (tokens, 8, 8))
    assert descriptor.tolist() == [tokens * 8, bits, tokens]
    if scalar:
        assert calls[1][0] == "route" and calls[1][1][1] is input_scales
    if tokens == 640:
        calls.clear()
        n.grouped(*arguments)
        assert calls[0][1][3] is input_scales
    g_full = FusedGeometry(tokens, 8, 72, 4096, 2048, 3, 3, bits)
    if tokens == 640 and bits == 4:
        assert n.allocate_scratch(g_full)[2].numel() * 4 == 327680


@pytest.mark.parametrize("groups", [8, 128])
def test_scalar_route_rows_match_original_broadcast_gathers(groups):
    scalar = torch.randn(17, groups, generator=torch.Generator().manual_seed(919))
    scalar[0, 0] = -0.0
    old = scalar[..., None].expand(-1, -1, 8).contiguous()
    tokens = torch.tensor([16, 0, 0, 5, 2])
    old_rows = old[tokens].reshape(tokens.numel(), -1)[:, ::8]
    new_rows = scalar[tokens]
    assert torch.equal(old_rows.view(torch.int32), new_rows.view(torch.int32))
