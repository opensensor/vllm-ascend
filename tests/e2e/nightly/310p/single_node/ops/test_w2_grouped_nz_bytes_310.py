# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exhaustive byte values across NZ planes and resident-L1 K boundaries."""

import pytest
import torch
import torch_npu

from tools.deepseek_w2.w2_format import unpack_codes
from vllm_ascend.models.glm5next_w2.model import _pack_codes_nz, _pack_codes_nz_w3
from vllm_ascend.utils import enable_custom_op


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_grouped_nz_exhaustive_bytes_across_k_tiles(bits: int):
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()

    n, k, experts = 256, 2048, 2
    # Every output column presents a different byte value. Vary K as well to
    # detect stale packed buffers and incorrect field or L1-stage addressing.
    packed_columns = torch.arange(k * bits // 8, dtype=torch.int32)
    output_rows = torch.arange(n, dtype=torch.int32)[:, None]
    codes = torch.stack([((output_rows + 17 * packed_columns + expert * 31) % 256).byte() for expert in range(experts)])
    positions = torch.tensor(
        sorted(
            {
                position
                for boundary in (0, 128, 256, 512, 1024, 1536, k)
                for position in range(boundary - 4, boundary + 4)
                if 0 <= position < k
            }
        )
    )
    routes = positions.numel()
    inputs = torch.zeros(experts * routes + 3, k, dtype=torch.float16)
    for expert in range(experts):
        inputs[expert * routes + torch.arange(routes), positions] = 1
    scales = (1 + torch.arange(experts * (n // 32) * (k // 32)) % 3).reshape(experts, n // 32, k // 32).float() / 32
    expected = torch.zeros(inputs.shape[0], n, dtype=torch.float16)
    for expert in range(experts):
        signed = unpack_codes(codes[expert], k, bits).float()
        selected_scales = scales[expert][torch.arange(n)[:, None] // 32, positions[None, :] // 32]
        expected[expert * routes : (expert + 1) * routes] = (signed[:, positions] * selected_scales).t().half()
    pack_nz = _pack_codes_nz_w3 if bits == 3 else _pack_codes_nz
    nz = torch.stack([pack_nz(codes[expert], k) for expert in range(experts)])
    ends = torch.tensor([routes, experts * routes], dtype=torch.int64, device="npu")
    actual = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310(
        inputs.npu(), nz.view(torch.int8).npu(), scales.npu(), ends
    ).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
