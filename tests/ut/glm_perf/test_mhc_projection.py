# SPDX-License-Identifier: Apache-2.0
"""Native projection packing and output byte indices retain the FP32 ABI."""

from pathlib import Path

import pytest
import torch

from tools.glm_perf.mhc_projection import (
    INPUT_WIDTH,
    OUTPUT_WIDTH,
    PADDED_OUTPUT_WIDTH,
    TILE_K,
    TILE_ROWS,
    UB_BYTES,
    NativeMhcProjection,
    offsets,
    pack_weight,
)


def test_fp16_weight_tiles_round_once_and_preserve_k_then_output_order():
    fn = (torch.arange(OUTPUT_WIDTH * INPUT_WIDTH).reshape(OUTPUT_WIDTH, INPUT_WIDTH) % 101).float() / 100
    original = fn.clone()
    packed = pack_weight(fn)
    recovered = packed.permute(1, 0, 2).reshape(PADDED_OUTPUT_WIDTH, INPUT_WIDTH)
    assert torch.equal(recovered[:OUTPUT_WIDTH], fn.half())
    assert torch.count_nonzero(recovered[OUTPUT_WIDTH:]) == 0
    assert torch.equal(fn, original)
    assert packed.is_contiguous()


def test_input_and_cube_gathers_use_byte_offsets_and_round_trip_layouts():
    table = offsets()
    values = torch.arange(TILE_ROWS * TILE_K).reshape(TILE_ROWS, TILE_K)
    gathered = values.reshape(-1)[table[: values.numel()].long() // 2]
    expected = values.reshape(TILE_ROWS, TILE_K // TILE_ROWS, TILE_ROWS).permute(1, 0, 2).reshape(-1)
    assert torch.equal(gathered, expected)
    output = torch.arange(TILE_ROWS * PADDED_OUTPUT_WIDTH).reshape(TILE_ROWS, PADDED_OUTPUT_WIDTH)
    raw = output.reshape(TILE_ROWS, PADDED_OUTPUT_WIDTH // TILE_ROWS, TILE_ROWS).permute(1, 0, 2).reshape(-1)
    restored = raw[table[values.numel() :].long() // 4].reshape_as(output)
    assert torch.equal(restored, output)
    assert UB_BYTES < 64 * 1024


def test_forward_requires_preparation_and_retains_float32_output():
    calls = []
    op = NativeMhcProjection(
        Path("/unused"),
        "unused",
        device=torch.device("cpu"),
        kernel_factory=lambda *args: object(),
        launch=lambda kernel, args, cores: calls.append((args, cores)),
    )
    fn = torch.zeros(OUTPUT_WIDTH, INPUT_WIDTH)
    residual = torch.zeros(33, 4, 4096)
    with pytest.raises(ValueError, match="prepared weights"):
        op(residual, fn)
    op.prepare([fn], [33])
    output = op(residual, fn)
    assert output.shape == (33, OUTPUT_WIDTH)
    assert output.dtype == torch.float32
    assert output.is_contiguous()
    assert len(calls) == 1
    assert calls[0][0][0] is residual
    fn.add_(1)
    with pytest.raises(ValueError, match="prepared weights"):
        op(residual, fn)


@pytest.mark.parametrize("rows", [0, True, 32769])
def test_invalid_row_capacity_rejected(rows):
    op = NativeMhcProjection.__new__(NativeMhcProjection)
    op.configs = {}
    with pytest.raises(ValueError, match="bounded integer"):
        op.prepare([], [rows])
