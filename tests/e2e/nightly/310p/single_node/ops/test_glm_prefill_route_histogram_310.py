# SPDX-License-Identifier: Apache-2.0
"""310P eager and graph parity for GLM's opt-in prefill route histogram."""

from typing import Literal

import pytest
import torch
import torch_npu

from vllm_ascend.models.qwen4_exp.grouped_expert_dispatch import build_grouped_expert_dispatch

LOCAL_EXPERTS = 72
EXPERT_OFFSET = 72
TOP_K = 8


def _dispatch(weights: torch.Tensor, ids: torch.Tensor, mode: Literal["compare", "histogram"]):
    return build_grouped_expert_dispatch(
        weights,
        ids,
        num_local_experts=LOCAL_EXPERTS,
        expert_offset=EXPERT_OFFSET,
        weight_dtype=torch.float32,
        count_mode=mode,
    )


def _assert_same_routes(candidate, reference) -> None:
    for field in ("counts", "group_list", "order", "inverse_order", "token_indices", "route_weights"):
        torch.testing.assert_close(getattr(candidate, field), getattr(reference, field), atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", [9, 128, 640])
def test_prefill_histogram_matches_comparison_on_310p(num_tokens: int) -> None:
    torch_npu.npu.set_compile_mode(jit_compile=False)
    generator = torch.Generator().manual_seed(310)
    ids = torch.randint(0, LOCAL_EXPERTS * 3, (num_tokens, TOP_K), generator=generator).npu()
    weights = torch.rand((num_tokens, TOP_K), generator=generator).npu()
    weights[:, 0] = 0
    ids[:, 0] = EXPERT_OFFSET
    _assert_same_routes(_dispatch(weights, ids, "histogram"), _dispatch(weights, ids, "compare"))


def test_prefill_histogram_replays_changed_routes_on_310p() -> None:
    torch_npu.npu.set_compile_mode(jit_compile=False)
    ids = torch.full((640, TOP_K), EXPERT_OFFSET, dtype=torch.int64, device="npu")
    weights = torch.ones((640, TOP_K), dtype=torch.float32, device="npu")
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            _dispatch(weights, ids, "histogram")
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = _dispatch(weights, ids, "histogram")
    for phase in range(4):
        ids.fill_(EXPERT_OFFSET + phase if phase % 2 == 0 else LOCAL_EXPERTS * 3)
        weights[:, 0] = 0 if phase % 2 else 1
        graph.replay()
        _assert_same_routes(captured, _dispatch(weights, ids, "compare"))
