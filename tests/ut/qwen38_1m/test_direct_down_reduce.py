# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tools.qwen4exp.direct_down_reduce import DirectDownReduce, validate_geometry


def inputs(tokens=4, top_k=10):
    rows = tokens * top_k
    return [
        torch.empty(rows, 320, dtype=torch.int8),
        torch.empty(rows, 320, dtype=torch.int8),
        torch.empty(rows, 5, 8),
        torch.empty(rows, 5, 8),
        torch.empty(2, 2560, 320, dtype=torch.int8),
        *[torch.empty(2, 2560, 5, dtype=torch.float16) for _ in range(3)],
        torch.empty(rows, dtype=torch.int32),
        torch.empty(tokens, top_k),
    ]


@pytest.mark.parametrize("tokens", [1, 3, 4, 5, 6])
def test_admitted_route_shapes(tokens):
    assert validate_geometry(inputs(tokens)) == (tokens * 10, 2, 2560, 640, 8, 1, 1, 10)


@pytest.mark.parametrize("tokens,top_k", [(0, 10), (7, 10), (4, 0), (1, 61)])
def test_capacity_and_empty_input_rejected(tokens, top_k):
    with pytest.raises(ValueError, match="geometry"):
        validate_geometry(inputs(tokens, top_k))


@pytest.mark.parametrize("index,dtype", [(8, torch.int64), (9, torch.float16), (2, torch.float16), (5, torch.float32)])
def test_binary_dtype_mismatch_rejected(index, dtype):
    args = inputs()
    args[index] = args[index].to(dtype)
    with pytest.raises(ValueError, match="dtypes"):
        validate_geometry(args)


def test_metadata_and_noncontiguous_routes_rejected():
    args = inputs()
    args[2] = args[2][:, :, 0].contiguous()
    with pytest.raises(ValueError, match="metadata"):
        validate_geometry(args)
    args = inputs()
    args[8] = torch.empty(80, dtype=torch.int32)[::2]
    with pytest.raises(ValueError, match="contiguous"):
        validate_geometry(args)


def test_cpu_rejected_before_allocating_or_launching():
    resource = DirectDownReduce(None, None)
    with pytest.raises(ValueError, match="NPU"):
        resource(*inputs())
    assert not resource.geometry_cache
