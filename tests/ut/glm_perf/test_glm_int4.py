# SPDX-License-Identifier: Apache-2.0
"""Independent integer arithmetic and layout gates for GLM native W4 trials."""

import pytest
import torch

from tools.glm_perf.glm_int4 import (
    GroupedProjectionGeometry,
    NativeW4Projection,
    activation_limbs,
    block_reference,
    fp16_weight_reference,
    pack_nz_codes,
    packed_weight_bits,
    pipeline_gather_offsets,
    signed_nibbles,
    unpack_canonical_codes,
    unpack_nz_w4,
)
from tools.glm_perf.reconstruction_native import ProjectionGeometry, geometry_for
from tools.glm_perf.resident_candidates.expert_reconstruction import wrap_projection


def test_all_int8_values_reconstruct_from_signed_int4_limbs():
    values = torch.arange(-127, 128, dtype=torch.float16)
    # Scale=1 in every group: include a magnitude-127 anchor.
    inputs = values[:, None].expand(-1, 32).clone()
    inputs[:, -1] = 127
    low, high, scale, quant = activation_limbs(inputs)
    assert low.min() >= -8 and low.max() <= 7
    assert high.min() >= -8 and high.max() <= 7
    assert torch.equal(low + 16 * high + 8, quant)
    assert torch.equal(quant[:, 0, 0], values.to(torch.int32))
    assert torch.equal(scale, torch.ones_like(scale))


def test_zero_activation_groups_have_finite_unit_scale():
    low, high, scale, quant = activation_limbs(torch.zeros(2, 256, dtype=torch.float16))
    assert torch.equal(scale, torch.ones_like(scale))
    assert torch.count_nonzero(quant) == 0
    assert torch.count_nonzero(low + 16 * high + 8) == 0


def test_nz_inverse_preserves_every_signed_nibble_and_tile_boundary():
    generator = torch.Generator().manual_seed(310)
    signed = torch.randint(-8, 8, (3, 128, 512), generator=generator, dtype=torch.int8)
    packed = pack_nz_codes(signed)
    assert torch.equal(unpack_nz_w4(packed, 512), signed)
    assert torch.equal(unpack_nz_w4(packed.view(torch.uint8), 512), signed)
    # Direct field ownership: one byte holds codes 128 K positions apart,
    # rather than two adjacent K values as in canonical row storage.
    byte = int(packed[0].view(torch.uint8).flatten()[0])
    assert byte & 15 == int(signed[0, 0, 0]) & 15
    assert byte >> 4 == int(signed[0, 0, 128]) & 15


def test_nibble_sign_extension_covers_all_bytes():
    packed = torch.arange(256, dtype=torch.uint8)
    actual = signed_nibbles(packed)
    for byte in range(256):
        fields = [byte % 16, byte // 16]
        assert actual[byte].tolist() == [value if value < 8 else value - 16 for value in fields]


def test_block32_scaling_does_not_collapse_four_distinct_groups():
    generator = torch.Generator().manual_seed(19)
    inputs = torch.randint(-127, 128, (3, 256), generator=generator).half()
    inputs[:, 31::32] = 127
    signed = torch.randint(-8, 8, (64, 256), generator=generator, dtype=torch.int8)
    scales = torch.tensor([[1, 2, 4, 8, 16, 8, 4, 2], [2, 4, 8, 16, 8, 4, 2, 1]]).float() / 1024
    actual = block_reference(inputs, signed, scales)
    weights = signed.float() * scales.repeat_interleave(32, 0).repeat_interleave(32, 1)
    expected = (inputs.float() @ weights.T).half()
    assert torch.equal(actual, expected)
    assert torch.equal(actual, fp16_weight_reference(inputs, signed, scales))


def test_activation_quantization_is_a_distinct_approximation():
    inputs = torch.full((1, 256), 0.01, dtype=torch.float16)
    inputs[:, 31::32] = 127
    signed = torch.ones((32, 256), dtype=torch.int8)
    scale = torch.full((1, 8), 1 / 32)
    assert not torch.equal(block_reference(inputs, signed, scale), fp16_weight_reference(inputs, signed, scale))


@pytest.mark.parametrize("byte_dtype", (torch.int8, torch.uint8))
def test_native_support_rejects_strided_activation_before_dispatch(byte_dtype):
    from types import SimpleNamespace

    device = SimpleNamespace(type="npu")

    def prepared(tensor):
        return SimpleNamespace(
            ndim=tensor.ndim,
            shape=tensor.shape,
            dtype=tensor.dtype,
            device=device,
            is_contiguous=tensor.is_contiguous,
        )

    projection = NativeW4Projection.__new__(NativeW4Projection)
    geometry = ProjectionGeometry(8, 3, 128, 256)
    projection.configs = {geometry: SimpleNamespace(device=device)}
    strided = torch.empty(256, 8, dtype=torch.float16).T
    tensors = tuple(
        prepared(tensor)
        for tensor in (
            torch.empty(3, 128, 128, dtype=byte_dtype),
            torch.empty(3, 4, 8),
            torch.empty(3, dtype=torch.int64),
        )
    )
    assert not projection.supports(prepared(strided), *tensors)
    assert projection.supports(prepared(strided.contiguous()), *tensors)


@pytest.mark.parametrize(
    "values",
    [
        (0, 72, 4096, 4096),
        (65, 72, 4096, 4096),
        (8, 0, 4096, 4096),
        (8, 72, 4095, 4096),
        (8, 72, 4096, 4097),
        (True, 72, 4096, 4096),
    ],
)
def test_projection_geometry_rejects_unsupported_dimensions(values):
    with pytest.raises(ValueError):
        ProjectionGeometry(*values)


@pytest.mark.parametrize("rows", (3, 65, 128, 6144, 65536))
def test_grouped_geometry_covers_prefill_rows_without_changing_decode_contract(rows):
    assert GroupedProjectionGeometry(rows, 72, 4096, 4096).rows == rows
    for bad in (0, 65537, True):
        with pytest.raises(ValueError):
            GroupedProjectionGeometry(bad, 72, 4096, 4096)


def test_fused_moe_selection_restores_bank_flags_on_failure():
    from types import SimpleNamespace

    bank = SimpleNamespace(decode_swiglu=False, decode_combine=True)

    def original(self, op, experts, *args):
        assert experts.decode_swiglu and experts.prefill_swiglu and experts.fp32_route_combine
        raise RuntimeError("down projection failed")

    def previous_selector(*args):
        raise AssertionError("previous selector must be unwrapped")

    previous_selector.__decode_flags_original__ = original
    wrapped = wrap_projection(previous_selector, None, fused_pipeline=True)
    with pytest.raises(RuntimeError, match="down projection failed"):
        wrapped(None, None, bank, None, None, None, None)
    assert vars(bank) == {"decode_swiglu": False, "decode_combine": True}


def test_w3_geometry_uses_storage_width_not_signed_byte_values():
    inputs = torch.empty(8, 256, dtype=torch.float16)
    codes = torch.empty(3, 128, 96, dtype=torch.int8)
    scales = torch.empty(3, 4, 8, dtype=torch.float32)
    ends = torch.empty(3, dtype=torch.int64)
    geometry = geometry_for(inputs, codes, scales, ends)
    assert geometry == ProjectionGeometry(8, 3, 128, 256)
    assert geometry.tiling("gm")[-1] == 0
    assert geometry.tiling("l1_singleton")[-1] == 1
    assert geometry.workspace_elements == 128 * 256
    with pytest.raises(ValueError, match="W3"):
        geometry_for(inputs, torch.empty(3, 128, 128, dtype=torch.int8), scales, ends)


def test_resident_wrapper_preserves_existing_dispatch_and_fallback():
    calls = []

    class Projection:
        def supports(self, inputs, *args):
            return inputs == "supported"

        def __call__(self, *args):
            calls.append(("native", args))
            return 19

    def baseline(*args):
        calls.append(("baseline", args))
        return 7

    def original(self, op, experts, x, weights, ids, shared):
        calls.append(("original", (self, experts, weights, ids, shared)))
        return op(x, "codes", "scales", "ends", False)

    selected = wrap_projection(original, Projection())
    assert selected("self", baseline, "experts", "supported", "weights", "ids", "shared") == 19
    assert calls[-1] == ("native", ("supported", "codes", "scales", "ends", False))
    assert selected("self", baseline, "experts", "prefill", "weights", "ids", "shared") == 7
    again = wrap_projection(selected, Projection())
    assert again.__glm_reconstruction_original__ is original
    assert again("self", baseline, "experts", "supported", "weights", "ids", "shared") == 19
    assert len([name for name, _ in calls if name == "original"]) == 3


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_nz_reference_packer_has_independent_field_ownership(bits):
    values = torch.arange(4096).remainder(1 << bits) - (1 << (bits - 1))
    signed = values.to(torch.int8).reshape(1, 16, 256)
    packed = pack_nz_codes(signed, bits).view(torch.uint8).flatten()
    logical = signed[0].T.contiguous().flatten().int() & ((1 << bits) - 1)
    count = 8 if bits == 3 else 8 // bits
    width = logical.numel() // count
    for offset in (0, 1, width - 1):
        word = sum(int(logical[field * width + offset]) << (field * bits) for field in range(count))
        for byte in range(3 if bits == 3 else 1):
            assert int(packed[byte * width + offset]) == (word >> (byte * 8)) & 255


@pytest.mark.parametrize(
    "bad", [torch.empty(1, 0, 256, dtype=torch.int8), torch.full((1, 16, 256), 8, dtype=torch.int8)]
)
def test_nz_packer_rejects_empty_and_out_of_range_codes(bad):
    with pytest.raises(ValueError):
        pack_nz_codes(bad)


def test_dispatch_audit_distinguishes_native_and_fallback_calls():
    class Projection:
        def supports(self, inputs, *args):
            return inputs == "decode"

        def __call__(self, *args):
            return "native"

    def original(self, op, experts, inputs, weights, ids, shared):
        return op(inputs, "codes", "scales", "ends")

    audit = {"native_dispatches": 0, "fallback_dispatches": 0}
    wrapped = wrap_projection(original, Projection(), audit)
    assert wrapped(None, lambda *args: "fallback", None, "decode", None, None, None) == "native"
    assert wrapped(None, lambda *args: "fallback", None, "prefill", None, None, None) == "fallback"
    assert audit == {"native_dispatches": 1, "fallback_dispatches": 1}


def test_next_selector_factory_unwraps_native_wrapper_to_the_original():
    def original(self, op, experts, inputs, weights, ids, shared):
        return op(inputs, "codes", "scales", "ends")

    def selector_factory(current):
        current = getattr(current, "__selector_original__", current)

        def selected(*args):
            return current(*args)

        selected.__selector_original__ = current
        return selected

    class Projection:
        def supports(self, *args):
            return True

        def __call__(self, *args):
            return "native"

    wrapped = wrap_projection(selector_factory(original), Projection())
    assert wrapped(None, lambda *args: "baseline", None, "decode", None, None, None) == "native"
    restored = selector_factory(wrapped)
    assert restored(None, lambda *args: "baseline", None, "decode", None, None, None) == "baseline"


@pytest.mark.parametrize("n", (32, 64, 128))
@pytest.mark.parametrize("bits", (2, 3, 4))
def test_integer_pipeline_repacking_preserves_every_weight_and_padding(n, bits):
    generator = torch.Generator().manual_seed(812 + n)
    signed = torch.randint(-(1 << (bits - 1)), 1 << (bits - 1), (1, n, 256), generator=generator, dtype=torch.int8)
    raw = pack_nz_codes(signed, bits).view(torch.uint8).flatten().int()
    raw = torch.cat((raw, torch.zeros(576, dtype=torch.int32)))
    channels = torch.cat((torch.arange(0, n, 2), torch.arange(1, n, 2)))
    offsets = pipeline_gather_offsets(n, bits).reshape(8, n, 16)
    output = torch.zeros_like(offsets)
    for group in range(8):
        shift = bits * group % 8 if bits == 3 else bits * (group // bits)
        low_bits = min(bits, 8 - shift)
        for field in range(4):
            for odd in (0, 1):
                part = slice(odd * n // 2, (odd + 1) * n // 2)
                index = offsets[group, part] + field * 16
                word = raw[index] | (raw[index + 1] << 8)
                fragment = word >> (shift + 8 * odd) & ((1 << low_bits) - 1)
                if bits == 3:
                    index = index + 512
                    word = raw[index] | (raw[index + 1] << 8)
                    fragment |= ((word >> (8 * odd)) & ((1 << (bits - low_bits)) - 1)) << low_bits
                output[group, part] |= fragment << (4 * field)
    if bits < 4:
        output |= (output & (0x4444 if bits == 3 else 0x2222)) * (2 if bits == 3 else 6)
    output &= 65535
    packed = torch.stack((output & 255, output >> 8), -1).to(torch.uint8)
    decoded = signed_nibbles(packed).reshape(8, n, 64)
    expected = signed[0, channels].reshape(n, 8, 32).permute(1, 0, 2)
    assert torch.equal(decoded[..., :32], expected)
    assert torch.count_nonzero(decoded[..., 32:]) == 0


def test_pipeline_metadata_rejects_unsupported_output_tile():
    with pytest.raises(ValueError, match="output tile"):
        pipeline_gather_offsets(48)


@pytest.mark.parametrize("phase", (0, 4, 8, 12))
def test_masked_nibble_normalization_is_exact_for_every_raw_word(phase):
    raw = torch.arange(1 << 16, dtype=torch.int32)
    masked = (raw & (15 << phase)).to(torch.int16)
    normalized = (masked.half() * (1.0 / (1 << phase))).round().to(torch.int16).int() & 15
    assert torch.equal(normalized, (raw >> phase) & 15)
    for field in range(4):
        native_word = (normalized.to(torch.int16) * (1 << (4 * field))).to(torch.int16).int() & 65535
        assert torch.equal(native_word, normalized << (4 * field))


@pytest.mark.parametrize("bits", (2, 3, 4))
def test_canonical_checkpoint_unpack_preserves_signed_codes(bits):
    signed = torch.arange(512).remainder(1 << bits).sub(1 << (bits - 1)).to(torch.int8).reshape(2, 256)
    fields_per_word = 8 if bits == 3 else 8 // bits
    fields = (signed.int() & ((1 << bits) - 1)).reshape(2, -1, fields_per_word)
    words = sum(fields[..., field] << (bits * field) for field in range(fields_per_word))
    packed = torch.stack([(words >> (8 * byte)) & 255 for byte in range(3 if bits == 3 else 1)], -1).to(torch.uint8)
    assert torch.equal(unpack_canonical_codes(packed.reshape(2, -1), 256), signed)
    assert packed_weight_bits(256, packed.numel() // 2) == bits
    with pytest.raises(ValueError):
        packed_weight_bits(256, packed.numel() // 2 + 1)
