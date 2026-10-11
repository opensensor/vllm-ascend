# SPDX-License-Identifier: Apache-2.0
"""CPU launch simulation checks the complete native ABI, routing and lifetimes.

This does not execute Ascend instructions or establish native numerical parity.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from tools.glm_perf.channel_scale_layout import NativeWeightBank, native_weight_parameters, product_offsets
from tools.glm_perf.channel_scale_math import LAYOUT, fused_moe_reference, quantize_weights
from tools.glm_perf.channel_scale_runtime import (
    KERNEL_NAMES,
    Geometry,
    NativeChannelScaleMoE,
    ScratchArena,
    scratch_shapes,
    shape_bytes,
    verify_bundle,
)
from tools.glm_perf.fused_weight_layout import unpack_cube


def bundle(tmp_path, fused=False, decode=False, identity=False, prefill=False, reciprocal=False, token=False):
    files = {}
    kernels = (*KERNEL_NAMES, "glm_channel_gate_up") if fused else KERNEL_NAMES
    if decode:
        kernels = (*kernels, "glm_channel_down_decode")
    if prefill:
        kernels = (*kernels, "glm_channel_down_prefill")
    for name in (*[n + ".bin" for n in kernels], "bridge.so"):
        payload = b"CPU ABI simulation only: " + name.encode()
        (tmp_path / name).write_bytes(payload)
        files[name] = hashlib.sha256(payload).hexdigest()
    (tmp_path / "provenance.json").write_text(
        json.dumps(
            {
                "token_prefill": token,
                "kernel_sources": {"glm_channel_down_prefill": "glm_channel_down_token.cpp"} if token else {},
                "reciprocal_quant": reciprocal,
                "fused_prefill": prefill,
                "identity_smoothing": identity,
                "abi_version": 4 if token else 3,
                "fused_decode": decode,
                "fused_gate_up": fused,
                "cube_rows": 32,
                "compiler_options": ["-DGLM_CHANNEL_ROWS=32"]
                + (["-DGLM_CHANNEL_IDENTITY_SMOOTHING"] if identity else [])
                + (["-DGLM_CHANNEL_RECIPROCAL_QUANT"] if reciprocal else []),
                "complete": True,
                "layout": LAYOUT,
                "namespace": "glm_channel_scale_v1",
                "files": files,
            }
        )
    )
    return tmp_path


def unpack_rows(packed, k):
    byte = packed.view(torch.uint8).int()
    pairs = torch.stack((byte & 15, byte >> 4), dim=-1).reshape(-1, k)
    return (pairs + 8) % 16 - 8


def pack_rows(values):
    pairs = values.to(torch.int8).view(torch.uint8).reshape(values.shape[0], values.shape[1] // 2, 2)
    return ((pairs[..., 0] & 15) | ((pairs[..., 1] & 15) << 4)).view(torch.int8)


class CpuLaunch:
    """Interpret explicit buffers, not the runtime's Python reference bank."""

    def __init__(self, identity=False, reciprocal=False, token=False):
        self.names = []
        self.identity = identity
        self.reciprocal = reciprocal
        self.token = token

    def __call__(self, kernel, arguments, blocks):
        assert blocks == 8
        self.names.append(kernel)
        if kernel.startswith("glm_channel_gate_up"):
            (
                low,
                high,
                xs,
                packed,
                ws,
                sums,
                ends,
                order,
                sorted_keys,
                offsets,
                diagonal,
                hidden_low,
                hidden_high,
                hidden_scales,
                quant_offsets,
                descriptor,
            ) = arguments
            tokens, top_k, experts, twice_i, k, group, bits, _ = descriptor.tolist()
            # CPU interpretation uses a temporary; the native implementation's
            # buffer stays in UB and the serving frontend allocates none.
            temporary = torch.empty(tokens * top_k, twice_i, dtype=torch.float16)
            original_count = len(self.names)
            self(
                "glm_channel_projection_v1",
                [low, high, xs, packed, ws, sums, ends, order, sorted_keys, offsets, temporary, descriptor],
                blocks,
            )
            ids = torch.full((tokens * top_k,), -1, dtype=torch.int64)
            first = 0
            for expert in range(experts):
                ids[order[first : int(ends[expert])]] = expert
                first = int(ends[expert])
            config = torch.tensor((tokens * top_k, twice_i // 2, 256, bits, 1, top_k, experts, 0))
            self(
                "glm_channel_quantize_v1",
                [temporary, diagonal, ends, order, ids, hidden_low, hidden_high, hidden_scales, quant_offsets, config],
                blocks,
            )
            del self.names[original_count:]
        elif kernel.startswith("glm_channel_down_prefill"):
            (
                low,
                high,
                xs,
                packed,
                ws,
                sums,
                ends,
                order,
                sorted_keys,
                offsets,
                tiles,
                inverse,
                weights,
                output,
                descriptor,
            ) = arguments
            tokens, top_k, experts, hidden = descriptor[:4].tolist()
            if self.token:
                token_inverse = torch.empty(tokens * top_k, dtype=torch.int32)
                for window in range(order.shape[0]):
                    for i in range(order.shape[1]):
                        position = window * 128 + int(inverse[window, i])
                        if position < token_inverse.numel():
                            token_inverse[position] = order[window, i]
                inverse = token_inverse
                order = torch.argsort(inverse)
            temporary = torch.empty(tokens * top_k, hidden, dtype=torch.float16)
            original_count = len(self.names)
            self(
                "glm_channel_projection_v1",
                [low, high, xs, packed, ws, sums, ends, order, sorted_keys, offsets, temporary, descriptor],
                blocks,
            )
            self("glm_channel_combine_v1", [temporary, inverse, ends, weights, output, descriptor], blocks)
            del self.names[original_count:]
        elif kernel.startswith("glm_channel_down_decode"):
            low, high, xs, packed, ws, sums, ends, order, sorted_keys, offsets, weights, output, descriptor = arguments
            tokens, top_k, experts, hidden = descriptor[:4].tolist()
            temporary = torch.empty(tokens * top_k, hidden, dtype=torch.float16)
            original_count = len(self.names)
            self(
                "glm_channel_projection_v1",
                [low, high, xs, packed, ws, sums, ends, order, sorted_keys, offsets, temporary, descriptor],
                blocks,
            )
            inverse = torch.argsort(order).int()
            self("glm_channel_combine_v1", [temporary, inverse, ends, weights, output, descriptor], blocks)
            del self.names[original_count:]
        elif kernel.startswith("glm_channel_quantize"):
            x, diagonal, ends, order, ids, low, high, scales, offsets, descriptor = arguments
            rows, k, group, bits, hidden, _, experts, offset = descriptor.tolist()
            count = int(ends[-1]) if hidden else rows
            value = x[:count].float()
            if hidden:
                gate, up = value[:, :k], value[:, k:]
                value = (torch.nn.functional.silu(gate) * up).clamp(-65504, 65504).half().float()
                if not self.identity:
                    value = value / diagonal[ids.reshape(-1)[order[:count]] - offset]
            elif not self.identity:
                value = value / diagonal
            grouped = value.reshape(count, k // group, group)
            scale = grouped.abs().amax(-1) / ((1 << (bits - 1)) - 1)
            scale[scale == 0] = 1
            normalized = grouped * scale.reciprocal()[..., None] if self.reciprocal else grouped / scale[..., None]
            q = normalized.round().int().reshape(count, k)
            if bits == 8:
                upper = q.div(16, rounding_mode="floor")
                high[:count].copy_(pack_rows(upper))
                q = q - 16 * upper - 8
            low[:count].copy_(pack_rows(q))
            scales[:count].copy_(scale[..., None].expand(-1, -1, 8))
        elif kernel.startswith("glm_channel_projection"):
            low, high, xs, packed, ws, sums, ends, order, sorted_keys, offsets, output, descriptor = arguments
            tokens, top_k, experts, n, k, group, bits, original = descriptor.tolist()
            weights = unpack_cube(packed, k).long()
            scales = ws.permute(0, 1, 3, 2).reshape(experts, n, k // group)
            corrections = sums.permute(0, 1, 3, 2).reshape(experts, n, k // group)
            first = 0
            for expert in range(experts):
                end = int(ends[expert])
                row_ids = order[first:end].div(top_k, rounding_mode="floor") if original else torch.arange(first, end)
                lo = unpack_rows(low[row_ids], k).long()
                hi = unpack_rows(high[row_ids], k).long() if bits == 8 else None
                result = torch.zeros(end - first, n)
                for g in range(k // group):
                    columns = slice(g * group, (g + 1) * group)
                    product = lo[:, columns] @ weights[expert, :, columns].T
                    if bits == 8:
                        product += 16 * (hi[:, columns] @ weights[expert, :, columns].T)
                        product += 8 * corrections[expert, :, g]
                    result += product.float() * scales[expert, :, g] * xs[row_ids, g, 0, None]
                output[first:end].copy_(result.half())
                first = end
        else:
            routed, inverse, ends, weights, output, descriptor = arguments
            tokens, top_k = descriptor[:2].tolist()
            output.zero_()
            for token in range(tokens):
                for slot in range(top_k):
                    row = int(inverse[token * top_k + slot])
                    if row < int(ends[-1]):
                        output[token].add_(routed[row].float() * weights[token, slot])


@pytest.mark.parametrize("fused,decode", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("tokens", [1, 4, 17, 37])
@pytest.mark.parametrize("bits", [4, 8])
def test_complete_five_launch_pipeline_matches_independent_moe_reference(tmp_path, tokens, bits, fused, decode):
    torch.manual_seed(33)
    gate = quantize_weights(torch.randn(3, 512, 256) / 64, 256)
    down = quantize_weights(torch.randn(3, 256, 256) / 64, 256)
    simulator = CpuLaunch()
    arena = ScratchArena()
    method = NativeChannelScaleMoE(
        bundle(tmp_path, fused, decode),
        gate,
        down,
        device="cpu",
        arena=arena,
        top_k=4,
        expert_offset=5,
        max_tokens=37,
        decode_bits=bits,
        prefill_bits=bits,
        kernel_factory=lambda path, name: name,
        launch=simulator,
    )
    x = torch.randn(tokens, 256).half()
    ids = torch.tensor([7, 5, 5, 999]).expand(tokens, 4).clone()
    weights = torch.tensor([0.2, 0.3, 0.4, 0.1]).expand(tokens, 4).clone()
    weights[::3, 1] = 0
    expected = fused_moe_reference(x, gate, down, weights, ids, expert_offset=5, bits=bits)
    actual = method(x, weights, ids)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_names = (
        (KERNEL_NAMES[0], "glm_channel_gate_up", KERNEL_NAMES[1], KERNEL_NAMES[2])
        if fused
        else (KERNEL_NAMES[0], KERNEL_NAMES[1], KERNEL_NAMES[0], KERNEL_NAMES[1], KERNEL_NAMES[2])
    )
    if decode and tokens <= 16:
        expected_names = (KERNEL_NAMES[0], "glm_channel_gate_up", "glm_channel_down_decode")
    assert simulator.names == [n + "_v1" for n in expected_names]
    assert ("gate_up" not in arena.get(tokens, 4, method.geometry, bits, "cpu")) == fused
    retained = actual.clone()
    weights.zero_()
    assert torch.count_nonzero(method(x * 2, weights, ids)) == 0
    assert torch.equal(actual, retained)  # scratch reuse cannot mutate a returned model output


def test_scratch_grows_with_incoming_rows_and_preserves_decode_addresses():
    geometry = Geometry(3, 256, 256, 256, 256)
    arena = ScratchArena()
    decode = arena.get(1, 8, geometry, 8, "cpu")
    pointers = {key: value.data_ptr() for key, value in decode.items()}
    arena.get(37, 8, geometry, 4, "cpu")
    bigger = arena.get(640, 8, geometry, 4, "cpu")
    smaller = arena.get(38, 8, geometry, 4, "cpu")
    assert smaller["gate_up"].data_ptr() == bigger["gate_up"].data_ptr()
    assert pointers == {key: value.data_ptr() for key, value in arena.get(1, 8, geometry, 8, "cpu").items()}
    assert arena.allocated_bytes == shape_bytes(scratch_shapes(1, 8, geometry, 8)) + shape_bytes(
        scratch_shapes(640, 8, geometry, 4)
    )


def test_full_glm_scratch_budget_rejects_before_allocation(monkeypatch):
    geometry = Geometry(144, 4096, 2048, 4096, 2048)
    needed = shape_bytes(scratch_shapes(1280, 8, geometry, 4))
    assert 170 * 1024**2 < needed < 180 * 1024**2
    arena = ScratchArena(needed - 1)
    real_empty = torch.empty

    def no_device_allocation(*args, **kwargs):
        assert not args or args[0] == (), "budget must reject before allocating tensors"
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", no_device_allocation)
    with pytest.raises(ValueError, match="budget"):
        arena.get(1280, 8, geometry, 4, "cpu")
    assert arena.allocated_bytes == 0


def test_bundle_requires_matching_binary_hashes(tmp_path):
    root = bundle(tmp_path)
    verify_bundle(root)
    (root / "glm_channel_projection.bin").write_bytes(b"different build")
    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(root)


@pytest.mark.parametrize("maximum_rows", [32, 64])
def test_product_offset_tables_match_independent_co1_fractal_construction(maximum_rows):
    tables = product_offsets(maximum_rows).long() // 4
    for block, rows in enumerate(range(16, maximum_rows + 1, 16)):
        logical = torch.arange(rows * 128).reshape(rows, 128)
        physical = torch.cat((logical[:, ::2], logical[:, 1::2]), dim=1)
        nz = physical.reshape(rows, 8, 16).permute(1, 0, 2).contiguous().flatten()
        for mode, limbs in ((0, 1), (2, 2)):
            for limb in range(limbs):
                for row in range(rows // limbs):
                    assert torch.equal(nz[tables[block, mode + limb, row]], logical[row * limbs + limb])


def test_weight_parameter_tiles_preserve_channel_scale_and_integer_sum_correspondence():
    bank = quantize_weights(torch.randn(2, 256, 1024), 256)
    packed, scales, sums, smoothing = native_weight_parameters(bank)
    assert packed is bank.codes and smoothing is bank.smoothing
    for expert in range(2):
        for tile in range(2):
            for group in range(4):
                assert torch.equal(
                    scales[expert, tile, group], bank.scales[expert, tile * 128 : (tile + 1) * 128, group]
                )
                assert torch.equal(sums[expert, tile, group], bank.sums[expert, tile * 128 : (tile + 1) * 128, group])


def test_native_factor_protocol_survives_frozen_helper_class_identity(monkeypatch):
    name = "tools.glm_perf.alternate_channel_scale_layout"
    path = Path(__file__).resolve().parents[3] / "tools/glm_perf/channel_scale_layout.py"
    spec = importlib.util.spec_from_file_location(name, path)
    alternate = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, alternate)
    spec.loader.exec_module(alternate)
    logical = quantize_weights(torch.randn(2, 256, 1024), 256)
    fields = native_weight_parameters(logical)
    stored = alternate.NativeWeightBank(*fields, logical.group_size).validate()
    assert not isinstance(stored, NativeWeightBank)
    parameters = native_weight_parameters(stored)
    assert all(actual is expected for actual, expected in zip(parameters, fields))


def test_bundle_rejects_declared_rows_that_were_not_compiled(tmp_path):
    root = bundle(tmp_path)
    manifest = json.loads((root / "provenance.json").read_text())
    manifest["cube_rows"] = 64
    (root / "provenance.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="compiler options"):
        verify_bundle(root)


def test_identity_variant_eliminates_diagonal_uploads_and_route_weight_copies(tmp_path):
    gate = quantize_weights(torch.randn(2, 512, 256) / 32, 256)
    down = quantize_weights(torch.randn(2, 256, 256) / 32, 256)
    arena = ScratchArena()
    simulator = CpuLaunch(identity=True)
    native = NativeChannelScaleMoE(
        bundle(tmp_path, True, True, True),
        gate,
        down,
        device="cpu",
        arena=arena,
        max_tokens=37,
        kernel_factory=lambda path, name: name,
        launch=simulator,
    )
    assert native.gate[3] is native.down[3] and native.gate[3].numel() == 1
    x = torch.randn(4, 256).half()
    ids = torch.tensor([1, 0, 0, 1, 0, 1, 99, 99]).expand(4, 8).clone()
    weights = torch.full((4, 8), 0.125)
    result = native(x, weights, ids)
    torch.testing.assert_close(result, fused_moe_reference(x, gate, down, weights, ids), rtol=0, atol=0)
    scratch = arena.get(4, 8, native.geometry, 8, "cpu")
    assert "weight_lanes" not in scratch and "gate_up" not in scratch and "routed" not in scratch
    nonidentity = quantize_weights(torch.randn(2, 512, 256), 256, torch.full((256,), 2.0))
    with pytest.raises(ValueError, match="unit stored"):
        NativeChannelScaleMoE(
            tmp_path,
            nonidentity,
            down,
            device="cpu",
            arena=arena,
            kernel_factory=lambda *args: pytest.fail("admission must precede kernel construction"),
        )


@pytest.mark.parametrize("reciprocal", [False, True])
def test_complete_fused_prefill_with_core_owned_route_tiles(tmp_path, reciprocal):
    gate = quantize_weights(torch.randn(2, 512, 1024) / 64, 256)
    down = quantize_weights(torch.randn(2, 1024, 256) / 64, 256)
    arena = ScratchArena()
    simulator = CpuLaunch(identity=True, reciprocal=reciprocal)
    native = NativeChannelScaleMoE(
        bundle(tmp_path, True, True, True, True, reciprocal),
        gate,
        down,
        device="cpu",
        arena=arena,
        max_tokens=37,
        kernel_factory=lambda path, name: name,
        launch=simulator,
    )
    x = torch.randn(37, 1024).half()
    ids = torch.tensor([1, 0, 0, 1, 0, 1, 99, 99]).expand(37, 8).clone()
    weights = torch.full((37, 8), 0.125)
    result = native(x, weights, ids)
    expected = fused_moe_reference(x, gate, down, weights, ids, reciprocal=reciprocal)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert simulator.names == [
        n + "_v1" for n in ("glm_channel_quantize", "glm_channel_gate_up", "glm_channel_down_prefill")
    ]
    scratch = arena.get(37, 8, native.geometry, 8, "cpu")
    assert "routed" not in scratch and "route_tiles" in scratch and "gate_up" not in scratch


def test_integer_product_marker_requires_actual_compiler_flag(tmp_path):
    bundle(tmp_path)
    path = tmp_path / "provenance.json"
    manifest = json.loads(path.read_text())
    manifest["integer_reconstruct"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="integer reconstruction marker"):
        verify_bundle(tmp_path)
    manifest["compiler_options"].append("-DGLM_CHANNEL_INTEGER_RECONSTRUCT")
    path.write_text(json.dumps(manifest))
    assert verify_bundle(tmp_path)["integer_reconstruct"]


def test_source_row_cache_marker_requires_actual_compiler_flag(tmp_path):
    bundle(tmp_path)
    path = tmp_path / "provenance.json"
    manifest = json.loads(path.read_text())
    manifest["cache_source_rows"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source row cache marker"):
        verify_bundle(tmp_path)
    manifest["compiler_options"].append("-DGLM_CHANNEL_CACHE_SOURCE_ROWS")
    path.write_text(json.dumps(manifest))
    assert verify_bundle(tmp_path)["cache_source_rows"]


@pytest.mark.parametrize("tokens", [17, 37])
def test_complete_token_owned_prefill_no_global_route_slab(tmp_path, tokens):
    torch.manual_seed(191)
    gate = quantize_weights(torch.randn(2, 512, 256) / 64, 256)
    down = quantize_weights(torch.randn(2, 256, 256) / 64, 256)
    arena = ScratchArena()
    simulator = CpuLaunch(identity=True, token=True)
    native = NativeChannelScaleMoE(
        bundle(tmp_path, fused=True, decode=True, identity=True, prefill=True, token=True),
        gate,
        down,
        device="cpu",
        arena=arena,
        max_tokens=37,
        kernel_factory=lambda path, name: name,
        launch=simulator,
    )
    x = torch.randn(tokens, 256).half()
    ids = torch.tensor([1, 0, 0, 1, 0, 1, 99, 99]).expand(tokens, 8).clone()
    weights = torch.full((tokens, 8), 0.125)
    result = native(x, weights, ids)
    expected = fused_moe_reference(x, gate, down, weights, ids)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    scratch = arena.get(tokens, 8, native.geometry, 8, "cpu")
    assert scratch["route_tiles"].numel() == 1
    assert "routed" not in scratch
    assert len(simulator.names) == 3
    assert scratch["token_rows"].dtype == torch.float32
    assert scratch["token_sorted_rows"].dtype == torch.int32


def test_token_metadata_cannot_be_loaded_as_bulk_abi(tmp_path):
    bundle(tmp_path, fused=True, prefill=True, token=True)
    path = tmp_path / "provenance.json"
    manifest = json.loads(path.read_text())
    manifest["token_prefill"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="distinct"):
        verify_bundle(tmp_path)
    manifest["abi_version"] = 3
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="bulk prefill"):
        verify_bundle(tmp_path)
