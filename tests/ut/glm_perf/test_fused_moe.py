# SPDX-License-Identifier: Apache-2.0
"""Fused MoE geometry, precision reference, dispatch and mandatory gates."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf.fused_moe_control import manifest
from tools.glm_perf.fused_moe_probe import projection_reference, reference
from tools.glm_perf.glm_fused_moe import FusedGeometry, NativeFusedMoE, fused_route_metadata, sorted_token_route_ranks
from tools.glm_perf.glm_int4 import MAX_GROUPED_ROUTES, activation_limbs, pack_nz_codes, signed_nibbles
from tools.glm_perf.resident_candidates.expert_reconstruction import wrap_fused_moe


@pytest.mark.parametrize("bits", (2, 3, 4))
def test_contiguous_decoder_and_cube_cast_pack_preserve_every_signed_code(bits):
    generator = torch.Generator().manual_seed(901 + bits)
    signed = torch.randint(-(1 << (bits - 1)), 1 << (bits - 1), (1, 128, 256), generator=generator, dtype=torch.int8)
    fields = 8 if bits == 3 else 8 // bits
    raw = pack_nz_codes(signed, bits).view(torch.uint8).int().reshape(8, -1)
    width = 4096 // fields
    decoded = []
    for field in range(fields):
        phase, plane = field * bits % 8, field * bits // 8
        low_bits = min(bits, 8 - phase)
        values = (
            (
                (raw[:, plane * width : (plane + 1) * width] & (((1 << low_bits) - 1) << phase)).half()
                * (1.0 / (1 << phase))
            )
            .round()
            .to(torch.int16)
            .int()
        )
        if low_bits < bits:
            values |= (raw[:, (plane + 1) * width : (plane + 2) * width] & ((1 << (bits - low_bits)) - 1)) << low_bits
        sign = 1 << (bits - 1)
        decoded.append(((values + sign) & ((1 << bits) - 1)) - sign)
    logical = torch.cat(decoded, -1).reshape(8, 4, 64, 16).permute(1, 0, 3, 2).reshape(4, 128, 64)
    physical = torch.cat((torch.arange(0, 128, 2), torch.arange(1, 128, 2)))
    cube = logical[:, physical].to(torch.int8)
    pairs = cube.int().reshape(4, 128, 32, 2) & 15
    packed = (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8)
    actual = signed_nibbles(packed).flatten(-2)
    expected = signed[0, physical].reshape(128, 4, 64).permute(1, 0, 2)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("bits", (2, 3, 4))
def test_byte_mask_sign_extension_stays_exact_through_half_for_all_codes(bits):
    raw = torch.arange(256, dtype=torch.int16)
    sign = 1 << (bits - 1)
    for phase in range(8 - bits + 1):
        mask = ((1 << bits) - 1) << phase
        expected = (raw.int() >> phase) & ((1 << bits) - 1)
        expected = ((expected + sign) & ((1 << bits) - 1)) - sign
        masked = ((raw + (sign << phase)) & mask).half()
        actual = masked * (1.0 / (1 << phase)) - sign
        assert torch.equal(actual.float(), expected.float())


@pytest.mark.parametrize("activation_bits", (4, 8))
def test_adjacent_scale_groups_share_weight_tile_without_mixing_integer_products(activation_bits):
    x = torch.linspace(-1, 1, 64).reshape(1, 64).half()
    weights = torch.arange(64).remainder(16).sub(8).int()
    if activation_bits == 8:
        low, high, _, quant = activation_limbs(x)
    else:
        groups = x.float().reshape(1, 2, 32)
        scale = groups.abs().amax(-1) / 7
        quant = (groups / scale[..., None]).round().clamp(-7, 7).int()
        low, high = quant, torch.zeros_like(quant)
    for group in (0, 1):
        # Each row activates only its own half of the full K=64 weight tile.
        a, b, bias = (torch.zeros(64, dtype=torch.int32) for _ in range(3))
        selected = slice(group * 32, (group + 1) * 32)
        a[selected], b[selected] = low[0, group], high[0, group]
        bias[selected] = 1
        integer = a @ weights
        if activation_bits == 8:
            integer += 16 * (b @ weights) + 8 * (bias @ weights)
        assert integer == quant[0, group] @ weights[selected]


def test_a4_integer_to_half_shortcut_preserves_rounding_ties_and_clamping():
    # Keep the division and nearest-even decision in FP32. Only the already
    # integral code changes its intermediate storage, including boundary ties.
    ties = torch.arange(-10, 11, dtype=torch.float32) + 0.5
    values = torch.cat(
        (
            torch.nextafter(ties, torch.full_like(ties, -torch.inf)),
            ties,
            torch.nextafter(ties, torch.full_like(ties, torch.inf)),
        )
    )
    integers = values.round().to(torch.int32)
    original = integers.float().clamp(-7, 7).half()
    direct = integers.half().clamp(-7, 7)
    assert torch.equal(direct, original)
    assert torch.equal(direct.float(), values.round().clamp(-7, 7))


@pytest.mark.parametrize("dtype", (torch.int32, torch.int64))
def test_fused_routes_keep_stable_duplicates_and_exclude_peer_and_zero_weight(dtype):
    ids = torch.tensor([[9, 8, 11, 10], [8, 99, 9, 9]], dtype=dtype)
    weights = torch.tensor([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=torch.float32)
    order, ends = fused_route_metadata(weights, ids, 3, 8)
    assert order.tolist() == [1, 0, 6, 7, 2, 3, 4, 5]
    assert ends.tolist() == [1, 4, 4]
    assert order.dtype == ends.dtype == torch.int64
    assert order.is_contiguous() and ends.is_contiguous()
    _, empty = fused_route_metadata(torch.zeros_like(weights), ids, 3, 8)
    assert empty.tolist() == [0, 0, 0]


def test_fused_routes_clamp_large_peer_ids_before_int32_conversion():
    ids = torch.tensor([[1 << 32, -1, 0]], dtype=torch.int64)
    order, ends = fused_route_metadata(torch.ones(1, 3), ids, 2)
    assert order.tolist() == [2, 0, 1]
    assert ends.tolist() == [1, 1]


def test_bulk_prefill_route_ranks_preserve_expert_order_duplicates_and_peer_suffix():
    ids = torch.tensor([[9, 8, 11, 10], [8, 99, 9, 9]])
    weights = torch.tensor([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=torch.float32)
    order, ends = fused_route_metadata(weights, ids, 3, 8)
    ranks = sorted_token_route_ranks(order, 2, 4)
    assert ranks.dtype == torch.int32 and ranks.tolist() == [[0, 1, 4, 5], [2, 3, 6, 7]]
    active = ranks < ends[-1]
    assert active.tolist() == [[True, True, False, False], [True, True, False, False]]
    for token in range(2):
        expected = [position for position, route in enumerate(order.tolist()) if route // 4 == token]
        assert ranks[token].tolist() == expected


@pytest.mark.parametrize("tokens", (16, 17, 640))
def test_bulk_prefill_uses_bounded_fp32_workspace_after_complete_native_down(tokens):
    native = object.__new__(NativeFusedMoE)
    native.weight_lookup = None
    native.device, native.activation_bits, native.configs = torch.device("cpu"), 4, {}
    native.scratch = {}
    native.pack_kernel, native.gate_kernel, native.down_kernel, native.reduce_kernel = "pack", "gate", "down", "reduce"
    calls = []
    native.launch = lambda kernel, args, blocks: calls.append((kernel, args))
    geometry = FusedGeometry(tokens, 2, 3, 256, 256, 4, 4, 4)
    order = torch.arange(tokens * 2).flip(0)
    ends = torch.tensor([0, tokens, tokens * 2])
    output = native.grouped(
        torch.zeros(tokens, 256, dtype=torch.float16),
        torch.zeros(3, 512, 128, dtype=torch.int8),
        torch.ones(3, 16, 8),
        torch.zeros(3, 256, 128, dtype=torch.int8),
        torch.ones(3, 8, 8),
        torch.ones(tokens, 2),
        order,
        ends,
        geometry,
    )
    assert output.shape == (tokens, 256) and output.dtype == torch.float32
    if tokens <= 16:
        assert [call[0] for call in calls] == ["pack", "gate", "down"]
        assert calls[2][1][8] is output
    else:
        assert [call[0] for call in calls] == ["pack", "gate", "down", "reduce"]
        workspace = calls[2][1][8]
        assert workspace is calls[3][1][0] and workspace.shape == (tokens * 2, 256)
        assert workspace.dtype == torch.float32 and workspace.numel() * workspace.element_size() == tokens * 2 * 256 * 4
        assert calls[3][1][2] is ends and calls[3][1][3] is output
        assert torch.equal(calls[3][1][1], sorted_token_route_ranks(order, tokens, 2))
    first_calls = list(calls)
    calls.clear()
    second = native.grouped(
        torch.zeros(tokens, 256, dtype=torch.float16),
        torch.zeros(3, 512, 128, dtype=torch.int8),
        torch.ones(3, 16, 8),
        torch.zeros(3, 256, 128, dtype=torch.int8),
        torch.ones(3, 8, 8),
        torch.ones(tokens, 2),
        order,
        ends,
        FusedGeometry(tokens, 2, 3, 256, 256, 3, 2, 4),
    )
    assert second is not output and second.data_ptr() != output.data_ptr()
    if tokens == 640:
        assert len(native.scratch) == 1
        assert calls[0][1][1] is first_calls[0][1][1]
        assert calls[1][1][7] is first_calls[1][1][7]
        assert calls[2][1][8] is first_calls[2][1][8]
    else:
        assert not native.scratch
        assert calls[0][1][1] is not first_calls[0][1][1]


def test_fused_routes_reject_mismatched_input_shapes():
    with pytest.raises(ValueError, match="matching"):
        fused_route_metadata(torch.ones(1, 1), torch.zeros(1, 2, dtype=torch.int64), 2)


@pytest.mark.parametrize("routes", (0, MAX_GROUPED_ROUTES + 1))
def test_fused_routes_reject_counts_outside_int32_reduction_contract(routes):
    ids = torch.zeros(1, routes, dtype=torch.int64)
    with pytest.raises(ValueError, match="route count"):
        fused_route_metadata(torch.ones(1, routes), ids, 2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tokens", 0),
        ("tokens", 32769),
        ("top_k", 0),
        ("experts", 289),
        ("hidden", 128),
        ("intermediate", 255),
        ("gate_bits", 1),
        ("down_bits", 5),
        ("activation_bits", 16),
        ("tokens", True),
    ],
)
def test_invalid_fused_geometry(field, value):
    args = dict(tokens=2, top_k=2, experts=3, hidden=256, intermediate=256, gate_bits=3, down_bits=4, activation_bits=8)
    args[field] = value
    with pytest.raises(ValueError):
        FusedGeometry(**args)


@pytest.mark.parametrize("activation_bits", (4, 8))
def test_reference_quantizes_to_requested_precision_and_reduces_duplicate_routes(activation_bits):
    x = torch.zeros(1, 256, dtype=torch.float16)
    x[0, 0] = 1
    x[0, 1] = 0.1
    signed = torch.ones(512, 256, dtype=torch.int8)
    scales = torch.ones(16, 8)
    projected = projection_reference(x, signed, scales, activation_bits)
    limit = (1 << (activation_bits - 1)) - 1
    expected = 1 + round(float(x[0, 1]) * limit) / limit
    torch.testing.assert_close(projected, torch.full_like(projected, expected), rtol=0, atol=0.001)
    gate = signed[None]
    down = torch.ones(1, 256, 256, dtype=torch.int8)
    ds = torch.ones(1, 8, 8) * 0.001
    gs = scales[None] * 0.001
    ids = torch.tensor([[5, 5]])
    weights = torch.tensor([[0.25, 0.75]])
    result = reference(x, gate, gs, down, ds, weights, ids, activation_bits, offset=5)
    single = reference(x, gate, gs, down, ds, torch.tensor([[1.0, 0.0]]), ids, activation_bits, offset=5)
    torch.testing.assert_close(result, single)
    assert torch.equal(
        reference(x, gate, gs, down, ds, weights, ids + 1, activation_bits, offset=5), torch.zeros_like(result)
    )


@pytest.mark.parametrize("dtype", (torch.int8, torch.uint8))
def test_fused_geometry_accepts_both_byte_containers_and_rejects_scale_mismatch(dtype):
    native = object.__new__(NativeFusedMoE)
    native.weight_lookup = None
    native.device = torch.device("cpu")
    native.activation_bits = 4
    args = [
        torch.zeros(2, 256, dtype=torch.float16),
        torch.zeros(3, 512, 96, dtype=dtype),
        torch.ones(3, 16, 8),
        torch.zeros(3, 256, 128, dtype=dtype),
        torch.ones(3, 8, 8),
        torch.ones(2, 2),
        torch.zeros(2, 2, dtype=torch.int64),
    ]
    assert native.geometry(*args).gate_bits == 3
    args[2] = torch.ones(3, 15, 8)
    with pytest.raises(ValueError, match="scale geometry"):
        native.geometry(*args)


def test_fused_wrapper_bypasses_original_pipeline_and_preserves_shared_branch():
    calls = []
    geometry = FusedGeometry(1, 2, 3, 256, 256, 3, 4, 8)

    class Native:
        input_dtype = torch.float16

        def geometry(self, *args):
            assert args[0].dtype == torch.float16
            return geometry

        def __call__(self, *args):
            calls.append(args)
            return torch.ones(1, 256)

    def original(*args):
        raise AssertionError("separate-kernel pipeline was invoked")

    experts = SimpleNamespace(
        gate_up_packed_bank=object(),
        gate_up_scale_bank=object(),
        down_packed_bank=object(),
        down_scale_bank=object(),
        local_expert_offset=5,
    )
    audit = {"native_dispatches": 0, "fallback_dispatches": 0, "bank_coverage": {}}
    shared = SimpleNamespace(forward=lambda x: torch.full_like(x, 2))
    result = wrap_fused_moe(original, Native(), audit)(
        None, None, experts, torch.ones(1, 256), torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.int64), shared
    )
    assert torch.equal(result, torch.full_like(result, 3))
    assert audit["native_dispatches"] == 2 and audit["fallback_dispatches"] == 0 and audit["kernel_fused_calls"] == 1
    assert len(audit["bank_coverage"]) == 2 and len(calls) == 1
    assert not hasattr(experts, "decode_swiglu")


def test_fused_wrapper_counts_rejected_bank_as_fallback():
    native = SimpleNamespace(input_dtype=torch.float16)

    def reject(*args):
        raise ValueError("bad bank")

    native.geometry = reject
    experts = SimpleNamespace(
        gate_up_packed_bank=None, gate_up_scale_bank=None, down_packed_bank=None, down_scale_bank=None
    )
    audit = {"native_dispatches": 0, "fallback_dispatches": 0, "bank_coverage": {}}
    wrapped = wrap_fused_moe(lambda *args: "original", native, audit)
    assert (
        wrapped(None, None, experts, torch.ones(1, 256), torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.int64), None)
        == "original"
    )
    assert audit["fallback_dispatches"] == 2 and audit["native_dispatches"] == 0
    native.prepared_weight_layout = True
    with pytest.raises(ValueError, match="complete native geometry"):
        wrapped(None, None, experts, torch.ones(1, 256), torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.int64), None)
    assert audit["fallback_dispatches"] == 2


@pytest.mark.parametrize("specialize_w3", [False, True])
def test_fused_manifest_requires_real_weights_prefill_and_exact_binaries(tmp_path, specialize_w3):
    namespace = "glm_reconstruction_v997"
    package = namespace + "_helpers"
    helper = tmp_path / package
    helper.mkdir()
    (helper / "__init__.py").write_text("")
    helpers = {"__init__.py": hashlib.sha256(b"").hexdigest()}
    options = {
        "namespace": namespace,
        "version": 997,
        "helper_package": package,
        "fused_moe": True,
        "specialize_w3": specialize_w3,
    }
    (tmp_path / "provenance.json").write_text(json.dumps({"_build": options, "_helpers": helpers}))
    names = ("glm_reconstruction_bridge_v997.so", "glm_fused_gate_up.bin", "glm_fused_down.bin", "glm_fused_pack.bin")
    if specialize_w3:
        names += ("glm_fused_gate_up_w3.bin", "glm_fused_down_w3.bin")
    for name in names:
        (tmp_path / name).write_bytes(name.encode())
    row = {"passed": True, "graph_changed_inputs_routes_weights": True, "fp16_intermediate_gm_bytes": 0, "tokens": 128}
    records = [dict(row, weight_bits=w, activation_bits=a) for w in (2, 3, 4) for a in (4, 8)]
    gates = {
        "complete": True,
        "build_options": options,
        "records": records,
        "real_weight_records": records,
        "binaries": {n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest() for n in names},
    }
    report = tmp_path / "gates.json"

    def write():
        report.write_text(json.dumps(gates))

    write()
    payload = manifest(tmp_path, report).value
    compile(payload["validation_source"], "fused manifest", "exec")
    assert payload["operators"] == [namespace + "::launch"]
    if specialize_w3:
        specialized = tmp_path / "glm_fused_down_w3.bin"
        assert str(specialized) in {entry["path"] for entry in payload["assets"]}
        original = specialized.read_bytes()
        specialized.write_bytes(b"tampered specialized stage")
        with pytest.raises(ValueError, match="exact binaries"):
            manifest(tmp_path, report)
        specialized.write_bytes(original)
    gates["real_weight_records"] = []
    write()
    with pytest.raises(ValueError, match="real-weight gates"):
        manifest(tmp_path, report)
    gates["real_weight_records"] = records
    gates["records"] = [dict(r, tokens=2) for r in records]
    write()
    with pytest.raises(ValueError, match="prefill gates"):
        manifest(tmp_path, report)
    gates["records"] = records
    write()
    (tmp_path / names[1]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="exact binaries"):
        manifest(tmp_path, report)


def test_input_is_quantized_once_per_token_and_shared_by_all_expert_tiles():
    native = object.__new__(NativeFusedMoE)
    native.weight_lookup = None
    native.device = torch.device("cpu")
    native.activation_bits = 8
    native.configs = {}
    native.scratch = {}
    native.pack_kernel, native.gate_kernel, native.down_kernel = "pack", "gate", "down"
    calls = []
    native.launch = lambda kernel, args, blocks: calls.append((kernel, args))
    geometry = FusedGeometry(2, 8, 3, 256, 256, 4, 4, 8)
    x = torch.ones(2, 256, dtype=torch.float16)
    gate = torch.zeros(3, 512, 128, dtype=torch.int8)
    down = torch.zeros(3, 256, 128, dtype=torch.int8)
    output = native.grouped(
        x,
        gate,
        torch.ones(3, 16, 8),
        down,
        torch.ones(3, 8, 8),
        torch.ones(2, 8),
        torch.arange(16),
        torch.tensor([4, 8, 16]),
        geometry,
    )
    assert [call[0] for call in calls] == ["pack", "gate", "down"]
    assert calls[0][1][0] is x
    assert calls[0][1][1].shape == (2, 8, 32)
    assert calls[0][1][1] is calls[1][1][0]
    assert calls[1][1][7].shape == (16, 8, 32)
    assert calls[1][1][7] is calls[2][1][0]
    assert output.shape == (2, 256) and output.dtype == torch.float32


@pytest.mark.parametrize("groups", (8, 64, 128))
def test_weight_scale_broadcast_matches_block32_in_native_channel_order(groups):
    # The native Cube tile places even channels before odd channels. A scale
    # belongs to the logical block32 channel, independent of this permutation.
    logical = torch.cat((torch.arange(0, 128, 2), torch.arange(1, 128, 2)))
    scales = torch.arange(4 * groups, dtype=torch.float32).reshape(4, groups)
    offsets = (torch.arange(128) % 64 // 16) * groups
    for group in (0, groups // 2, groups - 1):
        broadcast = scales.flatten()[offsets + group]
        assert torch.equal(broadcast, scales[logical // 32, group])
    # Persistent gather indices must follow, rather than overlap, FP32 cached
    # scales and their intermediate FP16 rounding storage.
    scale_bytes = 4 * 128 * (4 + 2)
    assert scale_bytes + 128 * 4 <= 4096


@pytest.mark.parametrize("bits", [2, 3])
def test_weight_lookup_reconstructs_every_two_byte_pattern(bits):
    from tools.glm_perf.glm_fused_moe import weight_decode_table

    tables = weight_decode_table().reshape(6, 4096).to(torch.int16)
    words = torch.arange(65536)

    def lookup(source, phase, width, table):
        byte_mask = ((1 << width) - 1) << phase
        pair = source & (byte_mask | (byte_mask << 8))
        signed = torch.where(pair >= 32768, pair - 65536, pair)
        return tables[table, signed // (1 << phase) + 2048]

    for field in range(8 if bits == 3 else 4):
        phase = field * bits % 8
        low = min(bits, 8 - phase)
        upper = words.roll(257)
        first = lookup(words, phase, low, (0 if bits == 2 else 1) if low == bits else (2 if low == 2 else 3))
        if low != bits:
            first += lookup(upper, 0, bits - low, 4 if low == 2 else 5)
        codes = []
        for shift in (0, 8):
            code = (words >> (phase + shift)) & ((1 << low) - 1)
            if low != bits:
                code |= ((upper >> shift) & ((1 << (bits - low)) - 1)) << low
            signed = torch.where(code >= (1 << (bits - 1)), code - (1 << bits), code)
            codes.append(signed & 15)
        assert torch.equal(first, codes[0] + 16 * codes[1])


@pytest.mark.parametrize("stage", ["gate", "down"])
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_specialized_stage_selected_only_for_w3(stage, bits):
    native = NativeFusedMoE.__new__(NativeFusedMoE)
    setattr(native, stage + "_kernel", "generic")
    setattr(native, stage + "_w3_kernel", "specialized")
    assert native.stage_kernel(stage, bits) == ("specialized" if bits == 3 else "generic")


def test_existing_bundle_without_specialized_files_retains_generic_dispatch():
    native = NativeFusedMoE.__new__(NativeFusedMoE)
    native.gate_kernel, native.down_kernel = "gate", "down"
    assert native.stage_kernel("gate", 3) == "gate"
    assert native.stage_kernel("down", 3) == "down"


def test_partial_w3_bundle_rejected_before_loading_a_kernel(tmp_path):
    (tmp_path / "glm_fused_gate_up_w3.bin").write_bytes(b"partial")
    with pytest.raises(ValueError, match="both fused stage binaries"):
        NativeFusedMoE(
            tmp_path,
            namespace="unused",
            activation_bits=4,
            launch=lambda *args: None,
            kernel_factory=lambda *args: pytest.fail("partial load"),
        )
