# SPDX-License-Identifier: Apache-2.0
"""Direct route IDs must capture and replay changing permutations on 310P."""

import pytest
import torch
import torch_npu

from tools.glm_perf.resident_candidates.direct_route_tokens import sorted_token_ids


@pytest.mark.parametrize("tokens", [2, 8, 640])
@pytest.mark.parametrize("top_k", [1, 3, 8])
def test_route_gather_graph_replays_changed_permutations(tokens, top_k):
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    cpu_inputs = torch.arange(tokens * 64).reshape(tokens, 64).to(torch.float16)
    inputs = cpu_inputs.npu()
    order = torch.arange(tokens * top_k, device="npu", dtype=torch.int64)

    def gather():
        ids = sorted_token_ids(order, top_k)
        return ids, inputs.index_select(0, ids)

    for _ in range(2):
        gather()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        ids, output = gather()
    generator = torch.Generator().manual_seed(3108)
    for _ in range(3):
        permutation = torch.randperm(tokens * top_k, generator=generator)
        order.copy_(permutation)
        graph.replay()
        torch.npu.synchronize()
        reference_ids = torch.arange(tokens).repeat_interleave(top_k)[permutation]
        assert torch.equal(ids.cpu().long(), reference_ids)
        assert torch.equal(output.cpu(), cpu_inputs[reference_ids])
