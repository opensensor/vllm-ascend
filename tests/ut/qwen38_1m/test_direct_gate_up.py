# SPDX-License-Identifier: Apache-2.0
"""Host checks for direct launch geometry and early rejection."""

import pytest
import torch

from tools.qwen4exp.direct_gate_up import DirectGateUp, validate_geometry


def inputs(tokens=4, routes=40):
    # Two experts suffice to exercise the shape contract without large banks.
    return [
        torch.empty(tokens, 1280, dtype=torch.int8),
        torch.empty(tokens, 1280, dtype=torch.int8),
        torch.empty(tokens, 20, 8),
        torch.empty(tokens, 20, 8),
        torch.empty(2, 1280, 1280, dtype=torch.int8),
        torch.empty(2, 1280, 20, dtype=torch.float16),
        torch.empty(2, 1280, 20, dtype=torch.float16),
        torch.empty(2, 1280, 20, dtype=torch.float16),
        torch.empty(routes, dtype=torch.int32),
    ]


@pytest.mark.parametrize("tokens", [3, 4, 6, 9, 12])
def test_broadcast_metadata(tokens):
    assert validate_geometry(inputs(tokens, tokens * 10)) == (tokens * 10, 2, 1280, 2560, 8, 1, 10)


@pytest.mark.parametrize("rows", [30, 31, 40, 41, 128])
def test_one_to_one_boundaries(rows):
    assert validate_geometry(inputs(rows, rows))[-1] == 1


@pytest.mark.parametrize("index,dtype", [(8, torch.int64), (2, torch.float16), (5, torch.float32)])
def test_wrong_binary_dtype_rejected(index, dtype):
    arguments = inputs()
    arguments[index] = arguments[index].to(dtype)
    with pytest.raises(ValueError, match="dtypes"):
        validate_geometry(arguments)


@pytest.mark.parametrize("tokens,routes", [(0, 0), (4, 41), (13, 130), (4, 3)])
def test_invalid_broadcast_rejected(tokens, routes):
    with pytest.raises(ValueError, match="geometry"):
        validate_geometry(inputs(tokens, routes))


def test_scalar_metadata_and_down_geometry_rejected():
    arguments = inputs()
    arguments[2] = arguments[2][:, :, 0].contiguous()
    with pytest.raises(ValueError, match="metadata"):
        validate_geometry(arguments)
    arguments = inputs()
    arguments[4] = torch.empty(2, 2560, 320, dtype=torch.int8)
    with pytest.raises(ValueError, match="geometry"):
        validate_geometry(arguments)


def test_cpu_call_fails_before_allocating_or_launching():
    resource = DirectGateUp(None, None)
    with pytest.raises(ValueError, match="NPU"):
        resource(*inputs())
    assert not resource.geometry_cache
