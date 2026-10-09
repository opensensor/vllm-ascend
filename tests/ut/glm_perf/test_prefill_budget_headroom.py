# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import pytest
import torch

from tools.glm_perf.glm_fused_moe import FusedGeometry, NativeFusedMoE, route_input_shapes
from tools.glm_perf.prefill_memory_budget import interior_prefill_capacity, moe_scratch_bytes


def test_actual_mixed_step_alignment_cliff():
    alone = interior_prefill_capacity(1280, 1280, 0)
    mixed = interior_prefill_capacity(1280, 1280, 1)
    assert alone["aligned_full_block_tokens"] == 1280
    assert mixed["available_prefill_tokens"] == 1278
    assert mixed["aligned_full_block_tokens"] == 640
    assert mixed["alignment_unused_tokens"] == 638


@pytest.mark.parametrize("decodes", range(4))
@pytest.mark.parametrize("draft_slots", range(3))
def test_small_headroom_preserves_prefill_cap_for_four_request_limit(decodes, draft_slots):
    row = interior_prefill_capacity(1296, 1280, decodes, draft_slots_per_request=draft_slots)
    assert row["aligned_full_block_tokens"] == 1280
    assert row["available_prefill_tokens"] == 1280


def test_separate_scheduling_budget_and_insufficient_physical_input_capacity():
    assert interior_prefill_capacity(1296, 1280, 3, scheduled_tokens=1280)["aligned_full_block_tokens"] == 640
    # 1288 is insufficient for three two-token queries plus four extra slots.
    row = interior_prefill_capacity(1288, 1280, 3, draft_slots_per_request=1)
    assert row["available_prefill_tokens"] == 1278
    assert row["aligned_full_block_tokens"] == 640
    assert interior_prefill_capacity(16, 1280, 9)["aligned_full_block_tokens"] == 0


@pytest.mark.parametrize("value", [True, -1, "3"])
def test_invalid_request_count_rejected(value):
    with pytest.raises(ValueError, match="nonnegative integers"):
        interior_prefill_capacity(1296, 1280, value)


def test_scratch_worksheet_matches_actual_cpu_allocation_shapes():
    native = NativeFusedMoE.__new__(NativeFusedMoE)
    native.device = torch.device("cpu")
    native.activation_bits = 4
    native.raw_input_scales = native.raw_hidden_scales = True
    native.route_packed_down = native.route_compact_down_scales = True
    native.fp16_route_workspace = True
    geometry = FusedGeometry(1280, 8, 72, 4096, 2048, 3, 3, 4)
    sizes = []
    for tokens in (1280, 1296):
        case = replace(geometry, tokens=tokens)
        buffers = native.allocate_scratch(case)
        packed, scales = route_input_shapes(case)
        routed = (torch.empty(packed, dtype=torch.int8), torch.empty(scales, dtype=torch.float32))
        actual = sum(value.numel() * value.element_size() for value in (*buffers, *routed))
        assert actual == moe_scratch_bytes(tokens)["total_bytes"]
        sizes.append(actual)
    assert 0 < sizes[1] - sizes[0] < 2 * 1024 * 1024
