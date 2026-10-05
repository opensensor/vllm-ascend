# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host contracts for c1 native-INT4 routed weight pipelining."""

from pathlib import Path

import pytest
import torch

from vllm_ascend.models.qwen4_exp.w4a8_int4 import GROUP_SIZE, INT4_FRACTAL_K, N_TILE, pack_native_weight

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEDULE = REPO_ROOT / "csrc/gmm/qwen_w4_a8_int4_matmul_v310/op_kernel/native_int4_schedule.h"
MATMUL_KERNEL = REPO_ROOT / ("csrc/gmm/qwen_w4_a8_int4_matmul_v310/op_kernel/qwen_w4_a8_int4_matmul_v310.cpp")
DOWN_KERNEL = REPO_ROOT / ("csrc/gmm/qwen_w4_a8_int4_down_reduce_v310/op_kernel/qwen_w4_a8_int4_down_reduce_v310.cpp")
BUILD = REPO_ROOT / "csrc/build_aclnn.sh"
PIPELINE_OUTPUTS = 320
PIPELINE_ROUTE_LIMIT = 30
PIPELINE_TILE_ROWS = 16


def _reference_pipeline_eligibility(route_ids: torch.Tensor) -> bool:
    if route_ids.numel() > PIPELINE_ROUTE_LIMIT:
        return False
    valid = route_ids[route_ids >= 0]
    if valid.numel() == 0:
        return False
    counts = torch.unique(valid, return_counts=True)[1]
    return bool(torch.all(counts <= PIPELINE_TILE_ROWS))


@pytest.mark.parametrize("width", [640, 2560])
@pytest.mark.parametrize("tile", [0, 1])
def test_strided_group_copy_matches_the_resident_expert_layout(width: int, tile: int):
    """The compact GM-to-L1 copy must preserve every packed Cube B tile."""
    outputs = 2 * PIPELINE_OUTPUTS
    generator = torch.Generator().manual_seed(width + tile)
    codes = torch.randint(-128, 128, (outputs, width // 2), dtype=torch.int8, generator=generator)
    packed, _ = pack_native_weight(codes)
    groups = width // GROUP_SIZE
    flat = packed.flatten()
    tile_base = tile * PIPELINE_OUTPUTS * width // 2
    strip_bytes = N_TILE * width // 2
    group_bytes = N_TILE * GROUP_SIZE // 2

    packed_tiles = packed.reshape(outputs // N_TILE, groups, 2, N_TILE, INT4_FRACTAL_K // 2)
    first_strip = tile * PIPELINE_OUTPUTS // N_TILE
    for group in range(groups):
        copied = torch.cat(
            [
                flat[
                    tile_base + strip * strip_bytes + group * group_bytes : tile_base
                    + strip * strip_bytes
                    + (group + 1) * group_bytes
                ]
                for strip in range(PIPELINE_OUTPUTS // N_TILE)
            ]
        )
        expected = packed_tiles[first_strip : first_strip + PIPELINE_OUTPUTS // N_TILE, group].flatten()
        torch.testing.assert_close(copied, expected, rtol=0, atol=0)


def _body(source: str, function: str, following: str) -> str:
    return source.split(function, 1)[1].split(following, 1)[0]


def test_pipeline_is_bounded_to_sparse_model_c1_and_keeps_full_tile_fallback():
    source = SCHEDULE.read_text()
    assert "PIPELINE_EXPERT_WEIGHTS && M == 16 && N == 320" in source
    assert "MAX_PIPELINED_ROUTES = 30" in source
    guard = _body(source, "bool CanPipelineExpertWeights", "void StartWeightPipeline")
    assert "if (rows_ > MAX_PIPELINED_ROUTES || firstGroup >= groups) return false;" in guard
    assert "starts[group + 1] - starts[group] > M" in guard
    assert "return false" in guard
    # Dense experts and every other tile shape retain the resident full-K path.
    assert "pipe_.InitBuffer(b1_, N * MAX_K / 2);" in source
    assert source.count("N * k_ / 2);") >= 3
    assert "if (!streamWeights) LoadWeight(0);" in source


def test_pipeline_is_removed_from_the_c4_kernel_specialization():
    matmul = MATMUL_KERNEL.read_text()
    down = DOWN_KERNEL.read_text()
    assert "if (td->numRows <= MODEL_C1_ROUTE_LIMIT)" in matmul
    assert "Schedule<16, N, false, N == MODEL_DECODE_COLUMNS, false>" in matmul
    assert "Schedule<16, N> op;" in matmul
    assert "Schedule<16, 320, true, true, false> op;" in down


def test_total_route_limit_keeps_sparse_c4_on_resident_weights():
    c1_routes = torch.arange(PIPELINE_ROUTE_LIMIT, dtype=torch.int32) % 3
    # Model c4 has 120 route slots. A rank commonly owns only a sparse subset,
    # spread across many experts, while the remaining slots belong to peers.
    c4_routes = torch.cat(
        (
            torch.arange(PIPELINE_ROUTE_LIMIT, dtype=torch.int32),
            torch.full((90,), -1, dtype=torch.int32),
        )
    )

    assert _reference_pipeline_eligibility(c1_routes)
    assert not _reference_pipeline_eligibility(c4_routes)
    # The c4 fallback is caused by total routes, not an overloaded expert.
    assert torch.unique(c4_routes[c4_routes >= 0], return_counts=True)[1].max() == 1


def test_next_weight_group_moves_after_mmad_and_before_result_wait():
    source = SCHEDULE.read_text()
    product = _body(source, "void ProductPair", "template <uint32_t PRODUCT_ROWS>")
    assert product.index("Mmad(") < product.index("PrefetchWeightGroup(")
    assert product.index("StageWeightGroup(") < product.index("SetFlag<HardEvent::M_V>")

    prefetch = _body(source, "void PrefetchWeightGroup", "void StageWeightGroup")
    assert prefetch.index("WaitFlag<HardEvent::MTE1_MTE2>") < prefetch.index("DataCopy(")
    assert prefetch.index("DataCopy(") < prefetch.index("SetFlag<HardEvent::MTE2_MTE1>")
    assert "DataCopyParams copy{N / BLOCK, BLOCK_LENGTH, sourceStride, 0};" in prefetch

    stage = _body(source, "void StageWeightGroup", "void ProcessPipelinedExpertWeights")
    assert stage.index("WaitFlag<HardEvent::MTE2_MTE1>") < stage.index("LoadData(")
    assert stage.index("LoadData(") < stage.index("SetFlag<HardEvent::MTE1_MTE2>")


def test_both_consumers_rebuild_when_the_shared_schedule_changes():
    manifest = BUILD.read_text()
    marker = 'elif [[ "${op_name}" == "qwen_w4_a8_int4_down_reduce_v310" ]] &&'
    dependency = manifest.split(marker, 1)[1].split("\n        fi", 1)[0]
    assert "qwen_w4_a8_int4_matmul_v310/op_kernel" in dependency
    assert '"qwen_w4_a8_int4_matmul_v310"' in manifest
    assert '"qwen_w4_a8_int4_down_reduce_v310"' in manifest
