# SPDX-License-Identifier: Apache-2.0
"""Routed-input capacity, launch ABI, and replay scratch ownership."""

import json

import pytest
import torch

from tools.glm_perf.build_reconstruction import build
from tools.glm_perf.glm_fused_moe import FusedGeometry, NativeFusedMoE, route_input_shapes


def test_routed_input_requires_matching_consumer_layout_before_build(tmp_path):
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="routed input packing requires paired"):
        build(output, tmp_path, tmp_path, route_packed_input=True)
    assert not output.exists()


@pytest.mark.parametrize("counts", [(0, 0, 0), (1, 31, 32), (30, 1, 0, 62), (63, 62, 33), (5120, 0, 0)])
def test_batch_slots_are_disjoint_and_bounded_even_with_empty_experts(counts):
    # Check slot allocation without allocating memory proportional to all experts.
    routes = max(17, sum(counts))
    geometry = FusedGeometry(routes, 1, len(counts), 4096, 256, 3, 3, 4)
    shape, scales = route_input_shapes(geometry)
    first, used = 0, set()
    for expert, count in enumerate(counts):
        for batch in range((count + 30) // 31):
            slot = first // 31 + expert + batch
            assert slot not in used and slot < shape[0]
            used.add(slot)
        first += count
    assert scales == (shape[0], 32, 128)
    assert shape[1:] == (64, 2048)


@pytest.mark.parametrize("tokens,bits", [(16, 4), (17, 8)])
def test_input_shapes_reject_decode_and_a8(tokens, bits):
    with pytest.raises(ValueError, match="bulk A4"):
        route_input_shapes(FusedGeometry(tokens, 2, 3, 256, 256, 3, 3, bits))


@pytest.mark.parametrize("tokens,bits,packed", [(16, 4, False), (17, 4, True), (640, 4, True), (640, 8, False)])
def test_launch_routes_only_bulk_a4_and_keeps_shared_buffers(tokens, bits, packed):
    native = NativeFusedMoE.__new__(NativeFusedMoE)
    native.device = torch.device("cpu")
    native.activation_bits = bits
    native.fp16_route_workspace = True
    native.weight_lookup = None
    native.configs, native.scratch = {}, {}
    native.pack_kernel, native.gate_kernel, native.down_kernel, native.reduce_kernel = "pack", "gate", "down", "reduce"
    native.route_input_kernel = "routed-input"
    calls = []
    native.launch = lambda kernel, args, blocks: calls.append((kernel, args))
    geometry = FusedGeometry(tokens, 2, 3, 256, 256, 3, 3, bits)
    args = (
        torch.zeros(tokens, 256, dtype=torch.float16),
        None,
        None,
        None,
        None,
        torch.ones(tokens, 2),
        torch.arange(tokens * 2),
        torch.tensor([0, 0, tokens * 2]),
        geometry,
    )
    output = native.grouped(*args)
    assert output.dtype == torch.float32 and output.shape == (tokens, 256)
    assert [k for k, _ in calls] == ["pack"] + (["routed-input"] if packed else []) + ["gate", "down"] + (
        ["reduce"] if tokens > 16 else []
    )
    if packed:
        pack_args, gate_args = calls[1][1], calls[2][1]
        assert gate_args[0] is pack_args[0] and gate_args[1] is pack_args[4] and gate_args[2] is pack_args[5]
        if tokens == 640:
            calls.clear()
            native.grouped(*args)
            assert calls[1][1][4] is pack_args[4] and calls[1][1][5] is pack_args[5]
            native.scratch.clear()
            calls.clear()
            native.grouped(*args)
            assert calls[1][1][4] is not pack_args[4]


def test_missing_route_producer_rejected_before_any_partial_kernel_load(tmp_path):
    (tmp_path / "provenance.json").write_text(json.dumps({"_build": {"route_packed_input": True}}))
    with pytest.raises(ValueError, match="compiled producer"):
        NativeFusedMoE(
            tmp_path,
            namespace="unused",
            activation_bits=4,
            kernel_factory=lambda *a: pytest.fail("partial kernel load"),
        )
