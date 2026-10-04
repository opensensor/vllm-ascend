# SPDX-License-Identifier: Apache-2.0
"""Host-only checks for the Qwen grouped prefill batching experiment."""

import pytest
import torch

from tools.qwen4exp.benchmark_w4_prefill_batching_310 import plan_cases, route_geometry


def test_glm_comparable_qwen_route_geometry_and_current_cap():
    cases = plan_cases(2048, 10, [512, 1024, 2048], 20480)
    assert [case["projection_calls"] for case in cases] == [4, 2, 1]
    assert [case["largest_call_routes"] for case in cases] == [5120, 10240, 20480]
    assert all(case["within_route_cap"] for case in cases)
    longer = plan_cases(23410, 10, [2048, 2560], 20480)
    assert [case["projection_calls"] for case in longer] == [12, 10]
    assert longer[0]["within_route_cap"] and not longer[1]["within_route_cap"]


def test_route_geometry_counts_peer_rows_and_revisited_experts():
    ids = torch.tensor([[4, 7], [4, 8], [5, 7], [4, 6]], dtype=torch.int64)
    single = route_geometry(ids, chunk_tokens=4, expert_offset=4, local_experts=3)
    split = route_geometry(ids, chunk_tokens=2, expert_offset=4, local_experts=3)
    assert single == {"local_routes": 5, "active_expert_visits": 3, "max_expert_rows": 3}
    assert split == {"local_routes": 5, "active_expert_visits": 4, "max_expert_rows": 2}


@pytest.mark.parametrize(
    "tokens,top_k,chunks,route_cap",
    [
        (0, 10, [512], 20480),
        (2048, 0, [512], 20480),
        (2048, 10, [], 20480),
        (2048, 10, [512, 512], 20480),
        (2048, 10, [0], 20480),
    ],
)
def test_plan_rejects_invalid_sizes(tokens, top_k, chunks, route_cap):
    with pytest.raises(ValueError):
        plan_cases(tokens, top_k, chunks, route_cap)
