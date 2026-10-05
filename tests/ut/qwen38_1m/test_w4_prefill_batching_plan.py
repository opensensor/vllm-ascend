# SPDX-License-Identifier: Apache-2.0
"""Host-only checks for the Qwen grouped prefill batching experiment."""

import json
import sys

import pytest
import torch

from tools.qwen4exp.benchmark_w4_prefill_batching_310 import (
    DEFAULT_CHUNKS,
    DEFAULT_ROUTE_CAP,
    main,
    plan_cases,
    route_geometry,
)
from vllm_ascend.models.qwen4_exp.w4_moe import MAX_GROUPED_NATIVE_ROUTES


def test_glm_comparable_qwen_route_geometry_and_current_cap():
    cases = plan_cases(2048, 10, [512, 1024, 2048], 20480)
    assert [case["projection_calls"] for case in cases] == [4, 2, 1]
    assert [case["largest_call_routes"] for case in cases] == [5120, 10240, 20480]
    assert all(case["within_route_cap"] for case in cases)
    longer = plan_cases(23410, 10, [2048, 2560], 20480)
    assert [case["projection_calls"] for case in longer] == [12, 10]
    assert longer[0]["within_route_cap"] and not longer[1]["within_route_cap"]


def test_candidate_capacity_covers_schedule_cliff_and_full_2560_chunk():
    assert DEFAULT_ROUTE_CAP == MAX_GROUPED_NATIVE_ROUTES
    cases = plan_cases(23410, 10, list(DEFAULT_CHUNKS), DEFAULT_ROUTE_CAP)
    assert [case["projection_calls"] for case in cases] == [16, 15, 15, 12, 10]
    assert cases[-1]["largest_call_routes"] == 25600
    assert all(case["within_route_cap"] for case in cases)


def test_candidate_dry_run_records_finalizer_without_touching_npu(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["benchmark_w4_prefill_batching_310", "--dry-run"])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["grouped_finalize"] == "cann_v2"
    assert result["npu_used"] is False
    assert [case["chunk_tokens"] for case in result["cases"]] == list(DEFAULT_CHUNKS)


def test_old_opp_cap_rejects_default_2560_chunk_before_model_load(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["benchmark_w4_prefill_batching_310", "--route-cap", "20480"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "chunk exceeds the installed grouped native route cap" in capsys.readouterr().err


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
