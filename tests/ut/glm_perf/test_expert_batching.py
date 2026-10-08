# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tools.glm_perf.benchmark_expert_batching_310 import plan_chunks


@pytest.mark.parametrize("chunk_tokens", [1, 3, 8, 16])
def test_chunks_preserve_every_route_and_expert_group(chunk_tokens):
    ids = torch.tensor([[0, 2, -1], [-1, -1, -1], [3, 0, 2], [2, 2, 0], [0, -1, 3]])
    chunks = plan_chunks(ids, 4, chunk_tokens)
    seen = []
    for chunk in chunks:
        assert chunk.begin % 3 == chunk.end % 3 == 0
        original = ids.reshape(-1)[chunk.begin : chunk.end]
        grouped = original[chunk.order]
        begin = 0
        for expert, end in enumerate(chunk.group_ends.tolist()):
            assert torch.all(grouped[begin:end] == expert)
            assert end - begin == int((original == expert).sum())
            begin = end
        assert torch.all(grouped[begin:] == -1)
        assert torch.equal(grouped[torch.argsort(chunk.order)], original)
        seen.extend((chunk.begin + chunk.order).tolist())
    assert sorted(seen) == list(range(ids.numel()))


@pytest.mark.parametrize(
    "ids,experts,chunk",
    [
        (torch.tensor([[4]]), 4, 1),
        (torch.tensor([[-2]]), 4, 1),
        (torch.tensor([[0]]), 4, 0),
        (torch.tensor([[0]]), 0, 1),
        (torch.tensor([0]), 1, 1),
    ],
)
def test_invalid_chunk_plan(ids, experts, chunk):
    with pytest.raises(ValueError):
        plan_chunks(ids, experts, chunk)
