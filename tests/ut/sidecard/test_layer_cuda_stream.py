# SPDX-License-Identifier: Apache-2.0
import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tools.sidecard.layer_weights import CudaExperts


@pytest.mark.parametrize("bits", (2, 3, 4))
def test_native_work_and_copies_use_current_torch_stream_without_sync(bits):
    def allocate(operation):
        return lambda *args, **kwargs: operation(*args, **dict(kwargs, device="cpu"))

    unexpected = Mock(side_effect=AssertionError("synchronous expert path used"))
    copy = Mock(return_value=0)
    actor = object.__new__(CudaExperts)
    actor.max_chunk = 32
    actor.use_stream = True
    actor.torch = SimpleNamespace(
        arange=allocate(torch.arange),
        empty=allocate(torch.empty),
        int32=torch.int32,
        float32=torch.float32,
        cuda=SimpleNamespace(current_stream=lambda: SimpleNamespace(cuda_stream=987), synchronize=unexpected),
    )
    pointers = tuple(ctypes.c_void_p(i) for i in range(123, 127))
    names = (
        "input",
        "ids",
        "row_map",
        "weights",
        "input_q",
        "input_scales",
        "gate_up",
        "hidden",
        "hidden_q",
        "hidden_scales",
        "output",
        "partial_output",
    )
    actor.oracle = SimpleNamespace(
        buffers={name: ctypes.c_void_p(i) for i, name in enumerate(names, 200)},
        bank_pointers={3: pointers},
        weight_bits={123: bits},
        launch=unexpected,
        reduce=unexpected,
        cuda=SimpleNamespace(
            check=lambda result: result == 0 or pytest.fail("CUDA error"),
            library=SimpleNamespace(cudaMemcpyAsync=copy, cudaMemcpy=unexpected, cudaDeviceSynchronize=unexpected),
        ),
    )
    actor.stream_packed, actor.stream_compact, actor.stream_reduce = (Mock(return_value=0) for _ in range(3))
    result = actor(3, torch.randn(1, 4096).half(), torch.arange(8)[None], torch.ones(1, 8))
    assert result.shape == (1, 4096)
    assert copy.call_count == 5
    assert all(call.args[-1].value == 987 and call.args[-2] == 3 for call in copy.call_args_list)
    launch = actor.stream_packed if bits == 4 else actor.stream_compact
    assert launch.call_count == 1
    assert launch.call_args.args[-1].value == 987
    if bits != 4:
        assert launch.call_args.args[-2] == bits
    assert actor.stream_reduce.call_args.args[-1].value == 987
    unexpected.assert_not_called()


def test_stream_abi_is_required_only_when_enabled():
    actor = object.__new__(CudaExperts)
    actor.oracle = SimpleNamespace(library=SimpleNamespace())
    actor.configure_stream(False)
    assert actor.use_stream is False
    with pytest.raises(ValueError, match="Torch-stream ABI"):
        actor.configure_stream(True)
