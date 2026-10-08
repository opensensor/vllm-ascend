# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hardware parity for grouped signed W2/W4 block-dequant Cube projections."""

import pytest
import torch
import torch_npu

from tools.deepseek_w2.w2_format import pack_codes, unpack_codes
from vllm_ascend.models.glm5next_w2.model import _pack_codes_nz, _pack_codes_nz_w3
from vllm_ascend.utils import enable_custom_op

W2_CUBE_MAX_ROWS = 128


@pytest.fixture(autouse=True, scope="module")
def require_kernel():
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    assert hasattr(torch.ops._C_ascend, "npu_w2_grouped_blocked_dequant_matmul_310")


def _reference(
    inputs: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    group_ends: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    rows, k = inputs.shape
    n = codes.shape[1]
    output = torch.zeros(rows, n, dtype=torch.float32)
    start = 0
    for expert, end in enumerate(group_ends.cpu().tolist()):
        if end > start:
            unpacked = unpack_codes(codes[expert].cpu(), k, bits).float()
            weight = (
                unpacked.view(n // 32, 32, k // 32, 32) * scales[expert].cpu().view(n // 32, 1, k // 32, 1)
            ).reshape(n, k)
            output[start:end] = inputs[start:end].cpu().float() @ weight.t()
        start = end
    return output.half()


def _standalone_projection_in_row_tiles(
    inputs: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    op = torch.ops._C_ascend.npu_w2_blocked_dequant_matmul_310
    return torch.cat(
        [
            op(inputs[start : start + W2_CUBE_MAX_ROWS], codes, scales)
            for start in range(0, inputs.shape[0], W2_CUBE_MAX_ROWS)
        ]
    )


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("initialize_output", [False, True])
def test_grouped_packed_projection_matches_reference(bits: int, initialize_output: bool):
    torch.manual_seed(7 + bits)
    experts, rows, n, k = 4, 11, 256, 256
    signed = torch.randint(-(1 << (bits - 1)), 1 << (bits - 1), (experts, n, k), dtype=torch.int8)
    codes = pack_codes(signed, bits).contiguous()
    scales = (torch.rand(experts, n // 32, k // 32) * 0.1 + 0.01).float()
    inputs = torch.randn(rows, k).half()
    group_ends = torch.tensor([3, 3, 8, 9], dtype=torch.int64)
    expected = _reference(inputs, codes, scales, group_ends, bits)

    actual = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310(
        inputs.npu(), codes.npu(), scales.npu(), group_ends.npu(), initialize_output
    ).cpu()

    # Rows after the final local group represent peer-owned routes. Their values
    # are ignored by the fused route combine when initialization is disabled.
    torch.testing.assert_close(actual[:9], expected[:9], rtol=4e-2, atol=4e-2)
    if initialize_output:
        assert torch.count_nonzero(actual[9:]) == 0


def test_grouped_command_releases_completed_queue_tensors():
    """Completed queue slots must not pin one workspace and output per call."""
    op = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310
    rows = 72
    width = 256
    inputs = torch.ones(rows, width, dtype=torch.float16, device="npu")
    codes = torch.zeros(1, width, width // 2, dtype=torch.uint8, device="npu")
    scales = torch.ones(1, width // 32, width // 32, dtype=torch.float32, device="npu")
    ends = torch.tensor([rows], dtype=torch.int64, device="npu")
    for _ in range(3):
        output = op(inputs, codes, scales, ends, False)
        del output
    torch.npu.synchronize()
    baseline = torch.npu.memory_allocated()
    output = inputs
    for _ in range(32):
        output = op(output, codes, scales, ends, False)
    assert torch.count_nonzero(output.cpu()) == 0
    del output
    torch.npu.synchronize()
    assert torch.npu.memory_allocated() <= baseline


@pytest.mark.parametrize("n,k", [(2048, 4096), (4096, 2048)])
def test_grouped_w3_glm_projection_matches_reference(n: int, k: int):
    """Exercise canonical W3 packing at the real GLM gate and down widths."""
    torch.manual_seed(n + k)
    experts, rows = 2, 4
    codes = torch.randint(0, 256, (experts, n, k * 3 // 8), dtype=torch.uint8)
    scales = (torch.rand(experts, n // 32, k // 32) * 0.02 + 0.005).float()
    inputs = torch.randn(rows, k).half()
    group_ends = torch.tensor([2, rows], dtype=torch.int64)
    expected = _reference(inputs, codes, scales, group_ends, 3)

    actual = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310(
        inputs.npu(), codes.npu(), scales.npu(), group_ends.npu()
    ).cpu()
    torch.testing.assert_close(actual, expected, rtol=4e-2, atol=4e-2)


@pytest.mark.parametrize("n,k", [(256, 256), (2048, 4096), (4096, 2048)])
def test_grouped_w3_nz_matches_canonical(n: int, k: int):
    torch.manual_seed(n + k + 3)
    experts, rows = 2, 4
    canonical = torch.randint(0, 256, (experts, n, k * 3 // 8), dtype=torch.uint8)
    nz = torch.stack([_pack_codes_nz_w3(canonical[expert], k) for expert in range(experts)])
    scales = (torch.rand(experts, n // 32, k // 32) * 0.02 + 0.005).npu()
    inputs = torch.randn(rows, k).half().npu()
    ends = torch.tensor([2, rows], dtype=torch.int64, device="npu")
    grouped = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310

    expected = grouped(inputs, canonical.npu(), scales, ends).cpu()
    actual = grouped(inputs, nz.view(torch.int8).npu(), scales, ends).cpu()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_grouped_w3_nz_admits_768_token_route_limit():
    """The 6,144-route kernel call must equal the previous two-call schedule."""
    torch.manual_seed(6144)
    experts, rows, n, k = 4, 6144, 256, 256
    canonical = torch.randint(0, 256, (experts, n, k * 3 // 8), dtype=torch.uint8)
    codes = torch.stack([_pack_codes_nz_w3(canonical[expert], k) for expert in range(experts)])
    codes_npu = codes.view(torch.int8).npu()
    scales = (torch.rand(experts, n // 32, k // 32) * 0.02 + 0.005).npu()
    inputs = torch.randn(rows, k).half().npu()
    ends = torch.tensor([1600, 3200, 4800, rows], dtype=torch.int64, device="npu")
    grouped = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310

    actual = grouped(inputs, codes_npu, scales, ends)
    first = grouped(inputs[:5120], codes_npu, scales, ends.clamp(max=5120))
    second = grouped(inputs[5120:], codes_npu, scales, (ends - 5120).clamp(min=0))
    expected = torch.cat((first, second))
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_grouped_projection_reuses_workspace_for_wide_projection():
    """Keep the real GLM expert geometry on the workspace-reuse path.

    GLM gate/up projections have sixteen 128-channel output tiles.  A grouped
    schedule assigns multiple tiles to each physical AI core.  Routes above 32
    rows make the Cube epilogue large enough to expose overlap with persistent
    dequant tables, while the smaller operator case above remains correct.
    """
    torch.manual_seed(31)
    experts, rows, n, k = 2, 34, 2048, 4096
    unsigned = torch.randint(0, 16, (experts, n, k), dtype=torch.uint8)
    codes = (unsigned[:, :, 0::2] | (unsigned[:, :, 1::2] << 4)).contiguous()
    scales = (torch.rand(experts, n // 32, k // 32) * 0.02 + 0.005).float()
    inputs = torch.randn(rows, k).half().npu()
    codes_npu = codes.npu()
    scales_npu = scales.npu()
    group_ends = torch.tensor([rows, rows], dtype=torch.int64, device="npu")

    grouped = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310(inputs, codes_npu, scales_npu, group_ends)
    single = torch.ops._C_ascend.npu_w2_blocked_dequant_matmul_310(inputs, codes_npu[0], scales_npu[0])

    torch.testing.assert_close(grouped.cpu(), single.cpu(), rtol=4e-2, atol=4e-2)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_standalone_projection_keeps_default_table_preparation(bits: int):
    """The grouped candidate must leave the shared standalone Cube path intact."""
    torch.manual_seed(310 + bits)
    rows, n, k = 3, 256, 256
    packed_width = k * bits // 8
    codes = torch.randint(0, 256, (n, packed_width), dtype=torch.uint8)
    scales = (torch.rand(n // 32, k // 32) * 0.02 + 0.005).float()
    inputs = torch.randn(rows, k).half()
    expected = _reference(inputs, codes.unsqueeze(0), scales.unsqueeze(0), torch.tensor([rows]), bits)
    actual = torch.ops._C_ascend.npu_w2_blocked_dequant_matmul_310(inputs.npu(), codes.npu(), scales.npu()).cpu()
    torch.testing.assert_close(actual, expected, rtol=4e-2, atol=4e-2)


@pytest.mark.parametrize("bits,n,k", [(4, 2048, 4096), (2, 4096, 2048)])
def test_grouped_projection_matches_sparse_glm_routes(bits: int, n: int, k: int):
    """Cover GLM-sized groups, empty experts, and peer-owned route rows.

    The grouped kernel reuses one core's dequant workspace across output
    tiles. An older package corrupted its UB tables when a group exceeded
    32 rows, producing NaNs despite correct one-row results.
    """
    torch.manual_seed(72 + bits)
    experts, rows = 4, 92
    codes_per_byte = 8 // bits
    codes = torch.randint(0, 256, (experts, n, k // codes_per_byte), dtype=torch.uint8).npu()
    scales = (torch.rand(experts, n // 32, k // 32) * 0.02 + 0.005).npu()
    inputs = torch.randn(rows, k).half().npu()
    group_ends = torch.tensor([44, 44, 89, 90], dtype=torch.int64, device="npu")

    grouped = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310(inputs, codes, scales, group_ends).cpu()
    for expert, start, end in ((0, 0, 44), (2, 44, 89), (3, 89, 90)):
        expected = torch.ops._C_ascend.npu_w2_blocked_dequant_matmul_310(
            inputs[start:end], codes[expert], scales[expert]
        ).cpu()
        assert torch.isfinite(grouped[start:end]).all()
        torch.testing.assert_close(grouped[start:end], expected, rtol=4e-2, atol=4e-2)
    assert torch.count_nonzero(grouped[90:]) == 0


@pytest.mark.parametrize("bits,n,k", [(4, 2048, 4096), (2, 4096, 2048)])
def test_grouped_projection_tiles_large_expert_groups(bits: int, n: int, k: int):
    """Keep every Cube invocation within its 128-row output allocation."""
    torch.manual_seed(128 + bits)
    rows = W2_CUBE_MAX_ROWS + 52
    codes_per_byte = 8 // bits
    canonical = torch.randint(0, 256, (1, n, k // codes_per_byte), dtype=torch.uint8)
    nz_codes = _pack_codes_nz(canonical[0], k).unsqueeze(0).view(torch.int8).npu()
    codes = canonical.npu()
    scales = (torch.rand(1, n // 32, k // 32) * 0.02 + 0.005).npu()
    inputs = torch.randn(rows, k).half().npu()
    group_ends = torch.tensor([rows], dtype=torch.int64, device="npu")

    grouped = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310(inputs, codes, scales, group_ends)
    grouped_nz = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310(inputs, nz_codes, scales, group_ends)
    expected = _standalone_projection_in_row_tiles(inputs, codes[0], scales[0])

    assert torch.isfinite(grouped).all()
    torch.testing.assert_close(grouped.cpu(), expected.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(grouped_nz.cpu(), grouped.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("bits,n,k", [(4, 2048, 4096), (2, 4096, 2048)])
@pytest.mark.parametrize("rows", [1, 16])
def test_grouped_nz_packed_projection_matches_canonical(bits: int, n: int, k: int, rows: int):
    torch.manual_seed(300 + bits)
    experts = 4
    canonical = torch.randint(0, 256, (experts, n, k // (8 // bits)), dtype=torch.uint8)
    nz_codes = torch.stack([_pack_codes_nz(canonical[expert], k) for expert in range(experts)])
    scales = (torch.rand(experts, n // 32, k // 32) * 0.02 + 0.005).npu()
    inputs = torch.randn(rows, k).half().npu()
    group_ends = torch.tensor([min(rows, 4), min(rows, 4), min(rows, 12), rows], dtype=torch.int64, device="npu")
    op = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310

    baseline = op(inputs, canonical.npu(), scales, group_ends)
    candidate = op(inputs, nz_codes.view(torch.int8).npu(), scales, group_ends)
    torch.testing.assert_close(candidate.cpu(), baseline.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("rows", [8, 32])
def test_grouped_72_experts_matches_canonical(bits: int, rows: int):
    """Cover 72 local experts, empty groups, and peer-owned trailing rows."""
    torch.manual_seed(900 + bits + rows)
    experts, n, k = 72, 256, 256
    codes_per_byte = 8 // bits
    canonical = torch.randint(0, 256, (experts, n, k // codes_per_byte), dtype=torch.uint8)
    nz_codes = torch.stack([_pack_codes_nz(canonical[expert], k) for expert in range(experts)])
    scales = (torch.rand(experts, n // 32, k // 32) * 0.02 + 0.005).npu()
    inputs = torch.randn(rows, k).half().npu()
    active_experts = (0, 17) if rows == 8 else (0, 8, 17, 26, 35, 44, 53, 71)
    rows_per_expert = 2
    local_rows = len(active_experts) * rows_per_expert
    group_ends = torch.tensor(
        [rows_per_expert * sum(active <= expert for active in active_experts) for expert in range(experts)],
        dtype=torch.int64,
        device="npu",
    )
    op = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310

    dense = op(inputs, canonical.npu(), scales, group_ends)
    sparse = op(inputs, nz_codes.view(torch.int8).npu(), scales, group_ends)

    torch.testing.assert_close(sparse.cpu(), dense.cpu(), rtol=0, atol=0)
    assert torch.count_nonzero(sparse[local_rows:]) == 0


@pytest.mark.parametrize("bits,n,k", [(4, 4096, 4096), (2, 4096, 2048)])
@pytest.mark.parametrize("rows", [8, 32, 416])
def test_glm_projection_geometry_singleton_mixed_and_peer_routes(bits: int, n: int, k: int, rows: int):
    """Exercise the opt-in singleton path at real GLM W4/W2 projection sizes.

    The single-expert operator is the unchanged numerical reference. The
    separate-process operator harness compares candidate and baseline OPP packages
    bitwise; this test also covers zero-local and peer-owned output rows.
    """
    torch.manual_seed(3100 + bits + rows)
    experts = 5
    canonical = torch.randint(0, 256, (experts, n, k // (8 // bits)), dtype=torch.uint8)
    nz_codes = torch.stack([_pack_codes_nz(canonical[expert], k) for expert in range(experts)])
    codes = nz_codes.view(torch.int8).npu()
    canonical = canonical.npu()
    scales = (torch.rand(experts, n // 32, k // 32) * 0.02 + 0.005).npu()
    inputs = torch.randn(rows, k).half().npu()
    grouped_op = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310
    for pattern in ("singleton", "repeated", "mixed", "zero_local", "peer_owned"):
        counts = {
            "singleton": [1, 1, 1, 1, 1],
            "repeated": [0, rows, 0, 0, 0],
            "mixed": [1, 0, rows - 2, 0, 1],
            "zero_local": [0, 0, 0, 0, 0],
            "peer_owned": [rows // 2, 0, 0, 0, 0],
        }[pattern]
        ends_cpu = torch.tensor(counts, dtype=torch.int64).cumsum(0)
        actual = grouped_op(inputs, codes, scales, ends_cpu.npu()).cpu()
        canonical_actual = grouped_op(inputs, canonical, scales, ends_cpu.npu()).cpu()
        assert actual.shape == (rows, n)
        assert torch.isfinite(actual).all()
        assert torch.equal(actual.view(torch.uint8), canonical_actual.view(torch.uint8))
        start = 0
        for expert, end in enumerate(ends_cpu.tolist()):
            if end > start:
                expected = _standalone_projection_in_row_tiles(
                    inputs[start:end], canonical[expert], scales[expert]
                ).cpu()
                torch.testing.assert_close(actual[start:end], expected, rtol=4e-2, atol=4e-2)
            start = end
        assert torch.count_nonzero(actual[start:]) == 0


@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("nz_packed", [False, True])
def test_grouped_scale_applies_to_both_nz_fragments(bits: int, nz_packed: bool):
    """A distinct scale per 32 input columns must cover both 16-wide NZ fragments."""
    n, k = 128, 256
    codes_per_byte = 8 // bits
    packed_one = sum(1 << (bits * field) for field in range(codes_per_byte))
    canonical = torch.full((n, k // codes_per_byte), packed_one, dtype=torch.uint8)
    codes = _pack_codes_nz(canonical, k).view(torch.int8) if nz_packed else canonical
    scales = torch.arange(1, 1 + (n // 32) * (k // 32), dtype=torch.float32).reshape(1, n // 32, k // 32) / 1024
    inputs = torch.zeros(k // 32, k, dtype=torch.float16)
    positions = torch.arange(k // 32) * 32 + 8
    inputs[torch.arange(k // 32), positions] = 1
    ends = torch.tensor([inputs.shape[0]], dtype=torch.int64, device="npu")

    actual = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310(
        inputs.npu(), codes.unsqueeze(0).npu(), scales.npu(), ends
    ).cpu()
    expected = scales[0, :, torch.arange(k // 32)].T.repeat_interleave(32, dim=1).to(torch.float16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-4)


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("large_rows", [129, 255, 256, 257, 513])
def test_resident_row_windows_match_gm_with_mixed_experts(bits, large_rows):
    """Exercise second L0C accumulator, partial windows, and expert transitions."""
    torch.manual_seed(4200 + bits + large_rows)
    experts, n, k = 5, 256, 2048
    codes = pack_codes(torch.randint(-(2 ** (bits - 1)), 2 ** (bits - 1), (experts, n, k)), bits)
    pack_nz = _pack_codes_nz_w3 if bits == 3 else _pack_codes_nz
    nz = torch.stack([pack_nz(expert, k) for expert in codes]).view(torch.int8).npu()
    counts = torch.tensor([1, 0, large_rows, 0, 17], dtype=torch.int64)
    ends = counts.cumsum(0).npu()
    rows = int(counts.sum()) + 7
    inputs = torch.randn(rows, k, dtype=torch.float16, device="npu")
    scales = (torch.rand(experts, n // 32, k // 32) * 0.02 + 0.005).npu()
    op = torch.ops._C_ascend.npu_w2_grouped_blocked_dequant_matmul_310
    expected = op(inputs, codes.npu(), scales, ends)
    actual = op(inputs, nz, scales, ends)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    assert torch.count_nonzero(actual[-7:]) == 0
