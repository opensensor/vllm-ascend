# SPDX-License-Identifier: Apache-2.0
"""Precision boundaries, routing ABI and frozen build options for prefill reuse."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf import build_reconstruction as builder
from tools.glm_perf.glm_fused_moe import (
    FusedGeometry,
    NativeFusedMoE,
    fused_route_metadata,
    sorted_token_route_ranks,
)


@pytest.mark.parametrize("tokens", [16, 17, 31, 640])
@pytest.mark.parametrize("half_routes", [False, True])
def test_half_workspace_dispatch_and_shared_scratch_keep_output_fp32(tokens, half_routes):
    native = NativeFusedMoE.__new__(NativeFusedMoE)
    native.device = torch.device("cpu")
    native.activation_bits = 4
    native.fp16_route_workspace = half_routes
    native.weight_lookup = None
    native.configs, native.scratch = {}, {}
    native.pack_kernel, native.gate_kernel, native.down_kernel, native.reduce_kernel = "pack", "gate", "down", "reduce"
    calls = []
    native.launch = lambda kernel, args, blocks: calls.append((kernel, args))
    geometry = FusedGeometry(tokens, 8, 3, 4096, 256, 3, 4, 4)
    order, ends = torch.arange(tokens * 8), torch.tensor([0, 0, tokens * 8])
    weights = torch.ones(tokens, 8)
    args = (torch.zeros(tokens, 4096, dtype=torch.float16), None, None, None, None, weights, order, ends, geometry)
    output = native.grouped(*args)
    assert output.dtype == torch.float32 and output.shape == (tokens, 4096)
    workspace = calls[2][1][8]
    if tokens == 16:
        assert workspace is output and len(calls) == 3
    else:
        assert workspace.dtype == (torch.float16 if half_routes else torch.float32)
        assert workspace.numel() * workspace.element_size() == tokens * 8 * 4096 * (2 if half_routes else 4)
        assert calls[3][1][0] is workspace
        assert calls[3][1][5:] == ([order, weights] if half_routes else [])
        if tokens == 640:
            calls.clear()
            next_output = native.grouped(*args)
            assert calls[2][1][8] is workspace and next_output.data_ptr() != output.data_ptr()


@pytest.mark.parametrize("empty", [False, True])
def test_half_route_storage_preserves_rounding_multiply_and_stable_reduction(empty):
    # Duplicate experts, peer routes, zero weights and a half-way rounding value.
    ids = torch.tensor([[9, 8, 9, 99], [8, 10, 9, 8]])
    weights = torch.tensor([[0.1234567, 0.75, 0.375, 1], [0, -0.125, 0.812345, 0.375]])
    if empty:
        weights.zero_()
    order, ends = fused_route_metadata(weights, ids, 3, 8)
    ranks = sorted_token_route_ranks(order, 2, 4)
    projected = torch.tensor([1.00048828125, -65504, 0.00000006, 123.456]).repeat(8, 1)
    projected += torch.arange(8)[:, None] * 0.015
    # Both producers round to half at the same existing boundary. The new
    # reducer must not round after weighting or use original-slot summation.
    half_workspace = projected.half()
    weighted_workspace = half_workspace.float() * weights.flatten()[order, None]
    expected, actual = torch.zeros(2, 4), torch.zeros(2, 4)
    for token in range(2):
        for row in ranks[token].tolist():
            if row >= int(ends[-1]):
                continue
            expected[token] += weighted_workspace[row]
            actual[token] += half_workspace[row].float() * weights.flatten()[order[row]]
    assert torch.equal(actual, expected)
    assert not torch.equal(weighted_workspace[0], (projected[0] * weights.flatten()[order[0]]).half().float()) or empty


@pytest.mark.parametrize("half_routes", [False, True])
def test_mismatched_route_dtype_rejected_before_any_kernel_load(tmp_path, half_routes):
    (tmp_path / "provenance.json").write_text(json.dumps({"_build": {"fp16_route_workspace": not half_routes}}))
    with pytest.raises(ValueError, match="differs from compiled"):
        NativeFusedMoE(
            tmp_path,
            namespace="unused",
            activation_bits=4,
            fp16_route_workspace=half_routes,
            launch=lambda *args: None,
            kernel_factory=lambda *args: pytest.fail("unsafe partial load"),
        )


@pytest.mark.parametrize(
    "options",
    [
        {"prefill_weight_cache": True},
        {"fp16_route_workspace": True},
        {"prefill_weight_cache": 1},
        {"fp16_route_workspace": "true"},
        {"share_gate_up_input": True},
        {"share_gate_up_input": 1},
        {"cache_gate_up_activations": True},
        {"cache_gate_up_activations": 1},
        {"vector_scale_products": True},
        {"vector_scale_products": 1},
        {"gather_product_matrix": True},
        {"gather_product_matrix": 1},
        {
            "fused_moe": True,
            "all_bits": True,
            "tile_pipeline": True,
            "output_columns": 128,
            "cache_gate_up_activations": True,
        },
        {
            "fused_moe": True,
            "all_bits": True,
            "tile_pipeline": True,
            "output_columns": 128,
            "prepared_weight_layout": True,
            "prefill_weight_cache": True,
            "weight_decode_lut": True,
        },
    ],
)
def test_bad_prefill_flags_fail_before_creating_build(tmp_path, options):
    with pytest.raises(ValueError):
        builder.build(tmp_path / "output", tmp_path, tmp_path, **options)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("cache,half_routes", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("share_input,cache_activations", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("vector_scales", [False, True])
@pytest.mark.parametrize("gather_products", [False, True])
def test_builder_freezes_matching_projection_reducer_flags(
    tmp_path, monkeypatch, cache, half_routes, share_input, cache_activations, vector_scales, gather_products
):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        target = command[command.index("-o") + 1] if "-o" in command else command[2]
        Path(target).write_bytes(b"mock compilation; not an NPU binary")

    monkeypatch.setattr(builder.subprocess, "run", fake_run)
    monkeypatch.setattr(
        builder.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(tmp_path / "__init__.py"))
    )
    monkeypatch.setattr(builder, "include_paths", lambda: [])
    monkeypatch.setattr(builder, "library_paths", lambda: [])
    output = builder.build(
        tmp_path / "build",
        tmp_path,
        tmp_path,
        version=993,
        output_columns=128,
        tile_pipeline=True,
        all_bits=True,
        fused_moe=True,
        prepared_weight_layout=True,
        prefill_weight_cache=cache,
        fp16_route_workspace=half_routes,
        share_gate_up_input=share_input,
        cache_gate_up_activations=cache_activations,
        vector_scale_products=vector_scales,
        gather_product_matrix=gather_products,
    )
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["_build"]["prefill_weight_cache"] is cache
    assert provenance["_build"]["fp16_route_workspace"] is half_routes
    assert provenance["_build"]["share_gate_up_input"] is share_input
    assert provenance["_build"]["cache_gate_up_activations"] is cache_activations
    assert provenance["_build"]["vector_scale_products"] is vector_scales
    assert provenance["_build"]["gather_product_matrix"] is gather_products
    for command in commands:
        if len(command) > 1 and Path(command[1]).name == "glm_fused_moe.cpp":
            assert ("-DGLM_PREFILL_WEIGHT_CACHE" in command) is cache
            assert ("-DGLM_VECTOR_SCALE_PRODUCTS" in command) is vector_scales
            assert ("-DGLM_GATHER_PRODUCT_MATRIX" in command) is gather_products
            assert ("-DGLM_FP16_ROUTE_WORKSPACE" in command) is half_routes
            assert ("-DGLM_SHARE_GATE_UP_INPUT" in command) is (share_input and "gate_up" in Path(command[2]).name)
            assert ("-DGLM_CACHE_GATE_UP_ACTIVATIONS" in command) is (
                cache_activations and "gate_up" in Path(command[2]).name
            )
        if len(command) > 1 and Path(command[1]).name == "glm_fused_reduce.cpp":
            assert ("-DGLM_FP16_ROUTE_WORKSPACE" in command) is half_routes
    assert "glm_fused_reduce.bin" in provenance


@pytest.mark.parametrize("conflict", ["weight_decode_lut", "strided_product_copy", "repeat_product_cast"])
def test_matrix_gather_rejects_overlapping_scratch_and_readback_schedules(tmp_path, conflict):
    options = {
        "fused_moe": True,
        "all_bits": True,
        "tile_pipeline": True,
        "output_columns": 128,
        "prepared_weight_layout": True,
        "gather_product_matrix": True,
        conflict: True,
    }
    with pytest.raises(ValueError, match="matrix product gather needs"):
        builder.build(tmp_path / "output", tmp_path, tmp_path, **options)
    assert not (tmp_path / "output").exists()


def test_cli_passes_w3_and_prefill_flags_to_builder(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build",
            "--build-dir",
            str(tmp_path),
            "--w3-float-fragments",
            "--specialize-w3",
            "--prefill-weight-cache",
            "--fp16-route-workspace",
            "--share-gate-up-input",
            "--cache-gate-up-activations",
            "--vector-scale-products",
            "--gather-product-matrix",
            "--prefill-rows-32",
            "--quad-hidden-quant",
            "--direct-hidden-gather",
            "--route-packed-input",
            "--route-packed-down",
            "--route-compact-down-scales",
            "--raw-hidden-scales",
            "--raw-input-scales",
            "--prefill-product-cast",
        ],
    )
    calls = []
    monkeypatch.setattr(builder, "build", lambda *args: calls.append(args))
    builder.main()
    assert calls[0][-20:] == (True,) * 16 + (False, True, False, False)
