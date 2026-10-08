# SPDX-License-Identifier: Apache-2.0
"""Independent transport and rounding contracts for dense Cube-layout prefill."""

import pytest
import torch

from tools.glm_perf.build_reconstruction import build


@pytest.mark.parametrize("options", [{"nz_prefill_accumulator": True}, {"nz_prefill_accumulator": 1}])
def test_invalid_nz_schedule_creates_no_build(tmp_path, options):
    with pytest.raises(ValueError, match="NZ prefill accumulator requires|must be boolean"):
        build(tmp_path / "build", tmp_path, tmp_path, **options)
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("count", [17, 20, 31])
@pytest.mark.parametrize("part", [0, 1])
def test_complete_row_cast_does_not_cross_cube_strip(count, part):
    # Hardware output: eight N16 strips, two M32 blocks, sixteen columns.
    raw = torch.arange(8 * 64 * 16, dtype=torch.int32).reshape(8, 64, 16)
    base = count * part
    native = raw[:, base : base + 32].float()
    restored = native[:, :count].permute(1, 0, 2).reshape(count, 128)
    reference = torch.stack([raw[:, base + row].flatten() for row in range(count)]).float()
    assert native.shape == (8, 32, 16)
    assert torch.equal(restored.view(torch.int32), reference.view(torch.int32))


@pytest.mark.parametrize("count", [17, 20, 31])
def test_unique_scale_factors_preserve_sequential_fp32_rounding(count):
    generator = torch.Generator().manual_seed(920 + count)
    scales = torch.rand(6, 32, generator=generator) * 0.03125
    weights = (torch.rand(6, 4, generator=generator) * 0.125).half().float()
    dots = torch.randint(-2048, 2049, (6, 8, 32, 16), generator=generator).float()
    dots[0, 0, 0, 0] = -0.0
    reference = torch.zeros(count, 128)
    native = torch.zeros(8, 32, 16)
    for group in range(6):
        # Original path: expand scales per physical column, then multiply dots,
        # then add each K32 group's result independently in FP32.
        physical_weights = weights[group].repeat(2).repeat_interleave(16)
        old_factor = scales[group, :count, None] * physical_weights[None]
        old_dot = dots[group, :, :count].permute(1, 0, 2).reshape(count, 128)
        reference = reference + old_dot * old_factor
        # Candidate computes each of the four factors once per row. Its second
        # packed column half reuses those exact FP32 factors.
        factors = weights[group, :, None] * scales[group, None]
        physical_factors = factors.repeat(2, 1)[:, :, None].expand(8, 32, 16)
        native = native + dots[group] * physical_factors
    restored = native[:, :count].permute(1, 0, 2).reshape(count, 128)
    assert torch.equal(restored.view(torch.int32), reference.view(torch.int32))
