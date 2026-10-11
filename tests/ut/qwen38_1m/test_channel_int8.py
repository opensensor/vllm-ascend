# SPDX-License-Identifier: Apache-2.0
"""Geometry and layout admission checks for isolated whole-K integer math."""

import pytest
import torch

from tools.qwen4exp.native_channel_int8 import NativeChannelInt8, tile_channel_weights


def test_cube_layout_preserves_known_channel_and_k_strips():
    weight = torch.zeros(2, 160, 256, dtype=torch.int8)
    weight[1, 19, 161] = -73
    tiled = tile_channel_weights(weight).flatten()
    # E / N16 / G128 / K32 / channel-in-strip / k-in-strip.
    index = ((((1 * 10 + 1) * 2 + 1) * 4 + 1) * 16 + 3) * 32 + 1
    assert tiled[index] == -73
    assert torch.count_nonzero(tiled) == 1


@pytest.mark.parametrize("shape", [(0, 160, 128), (129, 160, 128), (1, 0, 128), (1, 128, 128), (1, 160, 64)])
def test_invalid_raw_bank_geometry(shape):
    with pytest.raises(ValueError, match="geometry"):
        tile_channel_weights(torch.empty(shape, dtype=torch.int8))


def test_projection_reuses_weight_bank_and_device_group_boundaries(monkeypatch):
    monkeypatch.setattr("tools.qwen4exp.native_prefill._on_npu", lambda device: True)
    monkeypatch.setattr("tools.qwen4exp.native_prefill._capturing", lambda device: False)
    launches = []
    resource = NativeChannelInt8(None, lambda kernel, args, blocks: launches.append((args, blocks)))
    x = torch.zeros(3, 128, dtype=torch.int8)
    weights = torch.empty(2, 160, 128, dtype=torch.int8)
    scales = torch.ones(2, 160)
    input_scales = torch.ones(3, 8)
    ends = torch.tensor([0, 3], dtype=torch.int64)
    monkeypatch.setattr(torch.Tensor, "item", lambda self: pytest.fail("host count read"))
    output = resource(x, weights, scales, input_scales, ends)
    args, blocks = launches[0]
    assert args[1] is weights and args[4] is ends and args[5] is output and blocks == 8
    assert args[-1].tolist() == [3, 128, 160, 2, 1]
    assert output.shape == (3, 160)
    resource(x, weights, scales, input_scales, ends)
    assert launches[-1][0][-1] is args[-1]


def test_group_scale_recipe_is_not_accepted_as_channel_scales():
    resource = NativeChannelInt8(None, lambda *args: pytest.fail("invalid recipe launched"))
    with pytest.raises(ValueError, match="geometry/dtype"):
        resource(
            torch.zeros(3, 128, dtype=torch.int8),
            torch.empty(2, 160, 128, dtype=torch.int8),
            torch.ones(2, 160, 1),
            torch.ones(3, 8),
            torch.tensor([0, 3], dtype=torch.int64),
        )


@pytest.mark.parametrize(
    "shape,dtype", [((0, 128), torch.float16), ((3, 64), torch.float16), ((3, 128), torch.float32)]
)
def test_direct_quantizer_rejects_unsupported_inputs(shape, dtype):
    from tools.qwen4exp.native_channel_int8 import NativeDirectChannelActivation

    resource = NativeDirectChannelActivation(None, lambda *args: pytest.fail("invalid input launched"))
    with pytest.raises(ValueError, match="geometry/dtype"):
        resource(torch.empty(shape, dtype=dtype))


def test_direct_quantizer_reuses_descriptor_and_preserves_source(monkeypatch):
    from tools.qwen4exp.native_channel_int8 import NativeDirectChannelActivation

    monkeypatch.setattr("tools.qwen4exp.native_prefill._on_npu", lambda device: True)
    monkeypatch.setattr("tools.qwen4exp.native_prefill._capturing", lambda device: False)
    launches = []
    resource = NativeDirectChannelActivation(None, lambda kernel, args, blocks: launches.append((args, blocks)))
    inputs = torch.ones(3, 640, dtype=torch.float16)
    original = inputs.clone()
    output, scales = resource(inputs)
    args, blocks = launches[0]
    assert args[0] is inputs and args[1] is output and args[2] is scales and blocks == 8
    assert output.dtype == torch.int8 and scales.shape == (3, 8)
    resource(inputs)
    assert launches[-1][0][-1] is args[-1]
    assert torch.equal(inputs, original)


def test_channel_quantizer_rejects_wrong_lookup_shape():
    from tools.qwen4exp.native_channel_int8 import NativeChannelActivation

    resource = NativeChannelActivation(None, lambda *args: pytest.fail("invalid lookup launched"), torch.ones(512))
    prepared = (torch.zeros(3, 64, dtype=torch.int8), torch.zeros(3, 64, dtype=torch.int8), torch.ones(3, 1, 8), None)
    with pytest.raises(ValueError, match="geometry"):
        resource(prepared)


def test_narrow_column_tile_rejects_incompatible_output_width():
    resource = NativeChannelInt8(None, lambda *args: pytest.fail("invalid tile launched"), column_tile=64)
    with pytest.raises(ValueError, match="geometry"):
        resource(
            torch.zeros(3, 128, dtype=torch.int8),
            torch.empty(2, 160, 128, dtype=torch.int8),
            torch.ones(2, 160),
            torch.ones(3, 8),
            torch.tensor([0, 3], dtype=torch.int64),
        )


def test_packed_projection_reuses_resident_weight_and_metadata(monkeypatch):
    from types import SimpleNamespace

    from tools.qwen4exp.native_channel_int8 import NativePackedChannelInt8
    from tools.qwen4exp.native_hybrid_prefill import signed_nibble_lookup

    monkeypatch.setattr("tools.qwen4exp.native_prefill._on_npu", lambda device: True)
    monkeypatch.setattr("tools.qwen4exp.native_prefill._capturing", lambda device: False)
    launches = []
    bank = SimpleNamespace(
        weight=torch.zeros(2, 160, 64, dtype=torch.int8),
        weight_scale=torch.ones(2, 160, 1, dtype=torch.float16),
        weight_offset=torch.zeros(2, 160, 1, dtype=torch.float16),
    )
    resource = NativePackedChannelInt8(None, lambda kernel, args, blocks: launches.append(args), signed_nibble_lookup())
    x, channel_scales, input_scales = torch.zeros(3, 128, dtype=torch.int8), torch.ones(2, 160), torch.ones(3, 8)
    ends = torch.tensor([0, 3], dtype=torch.int64)
    monkeypatch.setattr(torch.Tensor, "item", lambda self: pytest.fail("host count read"))
    resource(x, bank, channel_scales, input_scales, ends)
    assert launches[0][1] is bank.weight and launches[0][2] is bank.weight_scale
    assert launches[0][3] is bank.weight_offset and launches[0][4] is channel_scales
    assert launches[0][-1].tolist() == [3, 128, 160, 2, 1]
    resource(x, bank, channel_scales, input_scales, ends)
    assert launches[1][-1] is launches[0][-1]


def test_local_quantizer_passes_device_boundaries_without_host_count(monkeypatch):
    from tools.qwen4exp.native_channel_int8 import NativeDirectChannelActivation

    monkeypatch.setattr("tools.qwen4exp.native_prefill._on_npu", lambda device: True)
    monkeypatch.setattr("tools.qwen4exp.native_prefill._capturing", lambda device: False)
    launches = []
    local = object()
    resource = NativeDirectChannelActivation(None, lambda kernel, args, blocks: launches.append((kernel, args)), local)
    inputs = torch.ones(640, 640, dtype=torch.float16)
    ends = torch.tensor([127, 256], dtype=torch.int64)
    monkeypatch.setattr(torch.Tensor, "item", lambda self: pytest.fail("host count read"))
    output, scales = resource(inputs, ends)
    kernel, args = launches[0]
    assert kernel is local and args[1] is ends and args[2] is output and args[3] is scales
    assert args[-1].tolist() == [640, 640, 2]
    resource(inputs, ends)
    assert launches[-1][1][-1] is args[-1]


def test_local_quantizer_rejects_missing_kernel(monkeypatch):
    from tools.qwen4exp.native_channel_int8 import NativeDirectChannelActivation

    monkeypatch.setattr("tools.qwen4exp.native_prefill._on_npu", lambda device: True)
    resource = NativeDirectChannelActivation(None, lambda *args: pytest.fail("unsupported local quantizer launched"))
    with pytest.raises(ValueError):
        resource(torch.ones(3, 128, dtype=torch.float16), torch.tensor([3], dtype=torch.int64))


def test_channel_bank_rejects_fractional_checkpoint_offsets():
    from types import SimpleNamespace

    from tools.qwen4exp.benchmark_channel_grouped_310 import channel_bank

    bank = SimpleNamespace(
        weight=torch.zeros(1, 160, 64, dtype=torch.int8),
        weight_scale=torch.ones(1, 160, 1, dtype=torch.float16),
        weight_offset=torch.full((1, 160, 1), 0.5, dtype=torch.float16),
    )
    with pytest.raises(ValueError, match="integer offsets"):
        channel_bank(bank)
