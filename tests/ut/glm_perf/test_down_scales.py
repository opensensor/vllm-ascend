# SPDX-License-Identifier: Apache-2.0
"""Capacity and transport gates for FP32 hidden scales in native prefill."""

import json

import pytest
import torch

from tools.glm_perf.build_reconstruction import build
from tools.glm_perf.glm_fused_moe import FusedGeometry, NativeFusedMoE, route_down_scale_shape


def test_scale_layout_dependency_rejected_before_build(tmp_path):
    with pytest.raises(ValueError, match="compact down scales require packed"):
        build(tmp_path / "build", tmp_path, tmp_path, route_compact_down_scales=True)
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize(
    "options,message",
    [({"route_compact_down_scales": 1}, "must be boolean"), ({"route_compact_down_scales": True}, "require packed")],
)
def test_bad_scale_provenance_rejected_before_kernel_load(tmp_path, options, message):
    (tmp_path / "provenance.json").write_text(json.dumps({"_build": options}))
    with pytest.raises(ValueError, match=message):
        NativeFusedMoE(
            tmp_path, activation_bits=4, namespace="unused", kernel_factory=lambda *args: pytest.fail("partial load")
        )


@pytest.mark.parametrize("tokens,bits,compact", [(16, 4, False), (17, 4, True), (640, 4, True), (640, 8, False)])
def test_scratch_scale_capacity_and_bypass(tokens, bits, compact):
    n = NativeFusedMoE.__new__(NativeFusedMoE)
    n.device, n.activation_bits = torch.device("cpu"), bits
    n.fp16_route_workspace, n.route_packed_down, n.route_compact_down_scales = True, True, True
    g = FusedGeometry(tokens, 8, 72, 4096, 2048, 3, 3, bits)
    scratch = n.allocate_scratch(g)
    assert scratch[5].dtype == torch.float32
    assert scratch[5].shape == (route_down_scale_shape(g) if compact else (tokens * 8, 64, 8))
    if tokens == 640 and bits == 4:
        assert tuple(scratch[5].shape) == (238, 16, 32, 8)
        assert scratch[5].numel() * scratch[5].element_size() == 3899392


@pytest.mark.parametrize("count", [1, 4, 16, 17, 31])
@pytest.mark.parametrize("groups", [8, 64, 128])
def test_tile_major_scale_transport_matches_original_active_rows(count, groups):
    generator = torch.Generator().manual_seed(917)
    original = torch.randn(count, groups, generator=generator, dtype=torch.float32)
    original[0, 0] = -0.0
    bank = torch.full((groups // 4, 32, 8), float("nan"), dtype=torch.float32)
    for tile in range(groups // 4):
        bank[tile, :count, :4] = original[:, tile * 4 : tile * 4 + 4]
        bank[tile, :count, 4:].zero_()
    # Consumer's row-relative offsets are independently derived from strides.
    flat = bank.flatten()
    recovered = torch.stack(
        [
            flat[row * 8 + torch.tensor([(group // 4) * 32 * 8 + group % 4 for group in range(groups)])]
            for row in range(count)
        ]
    )
    assert torch.equal(recovered.view(torch.int32), original.view(torch.int32))
    assert torch.equal(bank[:, :count, 4:], torch.zeros_like(bank[:, :count, 4:]))
    # Stale inactive rows are deliberately unread, including a shortened replay.
    assert torch.isnan(bank[:, count:]).all()


def test_raw_scale_dependency_rejected_before_build(tmp_path):
    with pytest.raises(ValueError, match="raw hidden scales require compact"):
        build(tmp_path / "build", tmp_path, tmp_path, raw_hidden_scales=True)
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("tokens,bits,raw", [(16, 4, False), (17, 4, True), (640, 4, True), (640, 8, False)])
def test_raw_bank_capacity_and_bypass(tokens, bits, raw):
    n = NativeFusedMoE.__new__(NativeFusedMoE)
    n.device, n.activation_bits = torch.device("cpu"), bits
    n.fp16_route_workspace, n.route_packed_down, n.route_compact_down_scales, n.raw_hidden_scales = (
        True,
        True,
        True,
        True,
    )
    g = FusedGeometry(tokens, 8, 72, 4096, 2048, 3, 3, bits)
    assert n.hidden_scale_shape(g) == (route_down_scale_shape(g, 4) if raw else (tokens * 8, 64, 8))
    if tokens == 640 and bits == 4:
        assert tuple(n.allocate_scratch(g)[5].shape) == (238, 16, 32, 4)
        assert n.allocate_scratch(g)[5].numel() * 4 == 1949696


@pytest.mark.parametrize("count", [1, 4, 16, 17, 31])
def test_aligned_four_row_scale_copy_preserves_active_rows(count):
    groups = 64
    source = torch.randn(32, groups, generator=torch.Generator().manual_seed(918), dtype=torch.float32)
    source[0, 0] = -0.0
    bank = torch.full((groups // 4, 32, 4), float("nan"))
    for tile in range(groups // 4):
        for first in range(0, count, 4):
            # Quantizer produces sixteen independent scalar FP32 scales.
            block = source[first : first + 4, tile * 4 : tile * 4 + 4].contiguous().flatten()
            destination = (tile * 32 + first) * 4
            assert destination * 4 % 32 == 0 and block.numel() * 4 == 64
            bank.flatten()[destination : destination + 16] = block
    recovered = bank[:, :count].permute(1, 0, 2).reshape(count, groups)
    assert torch.equal(recovered.view(torch.int32), source[:count].view(torch.int32))


@pytest.mark.parametrize(
    "options,message", [({"raw_hidden_scales": 1}, "must be boolean"), ({"raw_hidden_scales": True}, "require compact")]
)
def test_bad_raw_provenance_rejected_before_kernel_load(tmp_path, options, message):
    (tmp_path / "provenance.json").write_text(json.dumps({"_build": options}))
    with pytest.raises(ValueError, match=message):
        NativeFusedMoE(
            tmp_path, activation_bits=4, namespace="unused", kernel_factory=lambda *args: pytest.fail("partial load")
        )
