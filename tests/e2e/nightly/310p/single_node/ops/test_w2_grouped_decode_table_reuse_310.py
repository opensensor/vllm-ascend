# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity when grouped W2/W3/W4 experts share decode tables on 310P."""

import pytest
import torch
import torch_npu

from tools.deepseek_w2.w2_format import pack_codes
from vllm_ascend.models.glm5next_w2.model import _pack_codes_nz, _pack_codes_nz_w3
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(autouse=True, scope="module")
def require_grouped_kernel():
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    assert hasattr(torch.ops._C_ascend, "npu_w2_grouped_blocked_dequant_matmul_310")


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("nz_packed", [False, True])
@pytest.mark.parametrize("group_ends", [(0, 1, 1, 3), (1, 1, 2, 3)])
def test_grouped_decode_tables_survive_empty_experts(bits: int, nz_packed: bool, group_ends: tuple[int, ...]):
    """Compare reused tables to standalone preparation for every active expert."""
    torch.manual_seed(3100 + bits)
    experts, rows, n, k = 4, 4, 256, 256
    signed = torch.randint(-(1 << (bits - 1)), 1 << (bits - 1), (experts, n, k), dtype=torch.int8)
    canonical = pack_codes(signed, bits).contiguous()
    if nz_packed:
        pack_nz = _pack_codes_nz_w3 if bits == 3 else _pack_codes_nz
        codes = torch.stack([pack_nz(canonical[expert], k) for expert in range(experts)]).view(torch.int8)
    else:
        codes = canonical
    codes = codes.npu()
    canonical = canonical.npu()
    scales = (torch.rand(experts, n // 32, k // 32) * 0.02 + 0.005).float().npu()
    inputs = torch.randn(rows, k).half().npu()
    ends = torch.tensor(group_ends, dtype=torch.int64, device="npu")

    grouped = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310(inputs, codes, scales, ends).cpu()
    start = 0
    for expert, end in enumerate(group_ends):
        if end > start:
            standalone = torch.ops._C_ascend.npu_w2_blocked_dequant_matmul_310(
                inputs[start:end], canonical[expert], scales[expert]
            ).cpu()
            assert torch.equal(grouped[start:end], standalone)
        start = end
    assert torch.count_nonzero(grouped[start:]) == 0


@pytest.mark.parametrize("nz_packed", [False, True])
def test_w3_glm_width_decode_tables_survive_wide_cube_epilogue(nz_packed: bool):
    """Exercise W3 table reuse at a real gate width with more than 32 rows."""
    torch.manual_seed(3103)
    experts, rows, n, k = 3, 67, 2048, 4096
    canonical = torch.randint(0, 256, (experts, n, k * 3 // 8), dtype=torch.uint8)
    if nz_packed:
        codes = torch.stack([_pack_codes_nz_w3(canonical[expert], k) for expert in range(experts)]).view(torch.int8)
    else:
        codes = canonical
    codes = codes.npu()
    canonical = canonical.npu()
    scales = (torch.rand(experts, n // 32, k // 32) * 0.02 + 0.005).float().npu()
    inputs = torch.randn(rows, k).half().npu()
    ends = torch.tensor([33, 33, 66], dtype=torch.int64, device="npu")

    grouped = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310(inputs, codes, scales, ends).cpu()
    standalone = torch.ops._C_ascend.npu_w2_blocked_dequant_matmul_310
    assert torch.isfinite(grouped).all()
    assert torch.equal(grouped[:33], standalone(inputs[:33], canonical[0], scales[0]).cpu())
    assert torch.equal(grouped[33:66], standalone(inputs[33:66], canonical[2], scales[2]).cpu())
    assert torch.count_nonzero(grouped[66:]) == 0


@pytest.mark.parametrize("counts", [(129, 0, 3), (3, 0, 129), (128, 0, 3), (257, 0, 3), (3, 0, 257)])
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_resident_and_gm_experts_share_tables(counts: tuple[int, ...], bits: int):
    """Cross the resident row limit without relocating or corrupting tables."""
    torch.manual_seed(310128)
    experts, n, k = len(counts), 256, 2048
    canonical = torch.randint(0, 256, (experts, n, k * bits // 8), dtype=torch.uint8)
    pack_nz = _pack_codes_nz_w3 if bits == 3 else _pack_codes_nz
    nz = torch.stack([pack_nz(canonical[expert], k) for expert in range(experts)]).view(torch.int8).npu()
    canonical = canonical.npu()
    scales = (torch.rand(experts, n // 32, k // 32) * 0.02 + 0.005).float().npu()
    inputs = torch.randn(sum(counts) + 1, k).half().npu()
    ends = torch.tensor(counts, dtype=torch.int64).cumsum(0).npu()
    grouped = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310(inputs, nz, scales, ends).cpu()
    start = 0
    for expert, count in enumerate(counts):
        end = start + count
        if count:
            expected = torch.ops._C_ascend.npu_w2_blocked_dequant_matmul_310(
                inputs[start:end], canonical[expert], scales[expert]
            ).cpu()
            assert torch.equal(grouped[start:end], expected)
        start = end
    assert torch.count_nonzero(grouped[start:]) == 0


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("nz_packed", [False, True])
def test_expert_teams_cover_mixed_groups_on_all_cores(bits: int, nz_packed: bool):
    """N=1024 launches eight cores; >256 rows activates the team experiment."""
    torch.manual_seed(31072 + bits)
    counts = (129, 0, 1, 257, 18, 0, 128, 3, 0)
    experts, n, k = len(counts), 1024, 1024
    canonical = torch.randint(0, 256, (experts, n, k * bits // 8), dtype=torch.uint8)
    pack = _pack_codes_nz_w3 if bits == 3 else _pack_codes_nz
    codes = torch.stack([pack(canonical[e], k) for e in range(experts)]).view(torch.int8) if nz_packed else canonical
    codes, canonical = codes.npu(), canonical.npu()
    scales = (torch.rand(experts, n // 32, k // 32) * 0.02 + 0.005).float().npu()
    inputs = torch.randn(sum(counts) + 3, k).half().npu()
    ends = torch.tensor(counts, dtype=torch.int64).cumsum(0).npu()
    actual = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310(inputs, codes, scales, ends).cpu()
    start = 0
    for expert, count in enumerate(counts):
        end = start + count
        if count:
            expected = torch.ops._C_ascend.npu_w2_blocked_dequant_matmul_310(
                inputs[start:end], canonical[expert], scales[expert]
            ).cpu()
            assert torch.equal(actual[start:end].view(torch.uint8), expected.view(torch.uint8))
        start = end
    assert torch.count_nonzero(actual[start:]) == 0
