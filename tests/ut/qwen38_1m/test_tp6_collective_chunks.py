# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp.tp6_collective_chunks import chunk_plan, reduce_chunks


def test_minimal_exact_observed_geometries():
    sizes = (1, 2, 3, 4, 5, 6, 12, 18)
    for tokens, calls in ((131, 8), (134, 9), (128, 8)):
        plan = chunk_plan(tokens, sizes)
        assert sum(plan) == tokens and len(plan) == calls
        assert set(plan) <= set(sizes)


def test_full_order_and_ownership_before_launch():
    calls = []
    live = SimpleNamespace(owners={("float32", 2560): None, ("float32", 5120): None})
    live.all_reduce = lambda value, fallback: calls.append(value.numel()) or value * 6
    value = torch.arange(3 * 2560, dtype=torch.float32).reshape(3, 2560)
    assert torch.equal(reduce_chunks(live, value, (2, 1)), value * 6)
    assert calls == [5120, 2560]
    calls.clear()
    del live.owners["float32", 2560]
    with pytest.raises(ValueError, match="existing owner"):
        reduce_chunks(live, value, (2, 1))
    assert not calls


@pytest.mark.parametrize("tokens,sizes", [(0, (1,)), (2049, (1,)), (3, (2,)), (3, ()), (3, (0,)), (3, (1, True))])
def test_invalid_plan(tokens, sizes):
    with pytest.raises(ValueError):
        chunk_plan(tokens, sizes)
