# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Catch stale packed-byte tiles in the 310P grouped W4 vector pipeline."""

import pytest
import torch
import torch_npu

from vllm_ascend.models.glm5next_w2.model import _pack_codes_nz
from vllm_ascend.utils import enable_custom_op

EXPERTS = 72
WIDTH = 4096
BYTE_COLUMNS = WIDTH // 2
PROBE_POSITIONS = (1, 33, 127, 129, 255, 257, 1023)


@pytest.mark.parametrize("repeat", range(2))
def test_grouped_w4_releases_packed_byte_after_vector_read(repeat: int) -> None:
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    op = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310

    output_index = torch.arange(WIDTH, dtype=torch.int32).view(WIDTH, 1)
    byte_index = torch.arange(BYTE_COLUMNS, dtype=torch.int32).view(1, BYTE_COLUMNS)
    canonical = ((output_index + byte_index * 37) & 255).to(torch.uint8).unsqueeze(0)
    packed = _pack_codes_nz(canonical[0], WIDTH).unsqueeze(0).view(torch.int8)
    codes = torch.zeros((EXPERTS, WIDTH, BYTE_COLUMNS), dtype=torch.int8, device="npu:0")
    codes[0].copy_(packed[0].npu())
    scales = torch.ones((EXPERTS, WIDTH // 32, WIDTH // 32), dtype=torch.float32, device="npu:0")
    ends = torch.ones(EXPERTS, dtype=torch.int64, device="npu:0")

    for position in PROBE_POSITIONS:
        activations = torch.zeros((1, WIDTH), dtype=torch.float16, device="npu:0")
        activations[0, position] = 1
        actual = op(activations, codes, scales, ends).cpu().flatten()
        packed_byte = canonical[0, :, position // 2].to(torch.int32)
        unsigned = (packed_byte >> (4 * (position % 2))) & 15
        expected = (((unsigned + 8) & 15) - 8).to(torch.float16)
        assert torch.equal(actual, expected), f"repeat={repeat}, K-column={position}"
