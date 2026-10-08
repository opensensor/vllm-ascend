# SPDX-License-Identifier: Apache-2.0
"""Contiguous paired-Cube casts retain both independent K32 dot planes."""

import pytest
import torch

from tools.glm_perf.build_reconstruction import build


@pytest.mark.parametrize("options", [{"prefill_product_cast": True}, {"prefill_product_cast": 1}])
def test_invalid_contiguous_cast_creates_no_build(tmp_path, options):
    with pytest.raises(ValueError, match="contiguous prefill cast requires|must be boolean"):
        build(tmp_path / "build", tmp_path, tmp_path, **options)
    assert not (tmp_path / "build").exists()


def test_alternative_readback_schedules_cannot_be_combined(tmp_path):
    with pytest.raises(ValueError, match="alternative readback schedules"):
        build(
            tmp_path / "build",
            tmp_path,
            tmp_path,
            output_columns=128,
            tile_pipeline=True,
            all_bits=True,
            fused_moe=True,
            prepared_weight_layout=True,
            pair_scale_groups=True,
            pair_prefill_scale_groups=True,
            vector_scale_products=True,
            prefill_rows_32=True,
            nz_prefill_accumulator=True,
            prefill_product_cast=True,
        )
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("count", [17, 20, 31])
def test_in_place_cast_preserves_second_group_and_row_copy_bits(count):
    generator = torch.Generator().manual_seed(921 + count)
    integer = torch.randint(-2048, 2049, (8, 64, 16), generator=generator, dtype=torch.int32)
    converted = integer.clone()
    # Simulate same-width alias conversion. All elements, including the second
    # group's plane, must convert before either group is scaled separately.
    converted.view(torch.float32).copy_(converted.float())
    accumulator = torch.zeros(count, 128)
    reference = torch.zeros_like(accumulator)
    for part in range(2):
        base = part * count
        rows = converted.view(torch.float32)[:, base : base + count].permute(1, 0, 2).reshape(count, 128)
        old = integer[:, base : base + count].permute(1, 0, 2).reshape(count, 128).float()
        assert torch.equal(rows.view(torch.int32), old.view(torch.int32))
        factors = torch.rand(count, 128, generator=generator) * 0.003
        accumulator = accumulator + rows * factors
        reference = reference + old * factors
    assert torch.equal(accumulator.view(torch.int32), reference.view(torch.int32))
