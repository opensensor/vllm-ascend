# SPDX-License-Identifier: Apache-2.0
"""Deferred hardware gate for the experimental fused mHC post/FP16 round."""

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op


@pytest.fixture(autouse=True, scope="module")
def require_native_mhc():
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    assert hasattr(torch.ops._C_ascend, "npu_glm_mhc_post_310")


def make_inputs(rows, width):
    x = torch.randn(rows, width, device="npu").half()
    residual = torch.randn(rows, 4, width, device="npu").half().float()
    post = torch.randn(rows, 4, 1, device="npu").sigmoid().half().float()
    comb = torch.randn(rows, 4, 4, device="npu").softmax(-1).half().float()
    return x, residual, post, comb


def reference(x, residual, post, comb):
    return (torch.einsum("nij,nih->njh", comb, residual) + post * x.float().unsqueeze(1)).half().float()


def assert_parity(actual, expected):
    assert actual.dtype == torch.float32 and actual.is_contiguous()
    # Accumulation order is different; allow one FP16 relative step and
    # near-zero cancellation drift. Serving quality remains a separate gate.
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-3, atol=2e-6, equal_nan=True)
    torch.testing.assert_close(actual, actual.half().float(), rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("rows,width", [(1, 32), (4, 4096), (640, 1056), (640, 4096), (1280, 4096), (2560, 4096)])
def test_parity_and_immutable_inputs(rows, width):
    torch.manual_seed(310)
    args = make_inputs(rows, width)
    saved = [x.clone() for x in args]
    actual = torch.ops._C_ascend.npu_glm_mhc_post_310(*args)
    assert_parity(actual, reference(*args))
    for value, before in zip(args, saved, strict=True):
        torch.testing.assert_close(value, before, rtol=0, atol=0)


def test_rounding_boundaries_and_nonfinite_values():
    # Isolate rounding from summation-order differences; test the actual
    # torch_npu cast, including its observed saturation of overflow, NaN, and Inf.
    values = torch.tensor(
        [
            0.0,
            -0.0,
            2**-25,
            3 * 2**-25,
            1 + 2**-11,
            1 + 3 * 2**-11,
            65504.0,
            65520.0,
            131008.0,
            -131008.0,
            float("inf"),
            -float("inf"),
            float("nan"),
        ]
    )
    residual = values.repeat(3)[:32].view(1, 1, 32).repeat(1, 4, 1).npu()
    x = torch.zeros(1, 32, dtype=torch.float16, device="npu")
    post = torch.zeros(1, 4, 1, device="npu")
    comb = torch.full((1, 4, 4), 0.25, device="npu")
    actual = torch.ops._C_ascend.npu_glm_mhc_post_310(x, residual, post, comb)
    torch.testing.assert_close(actual.cpu(), reference(x, residual, post, comb).cpu(), rtol=0, atol=0, equal_nan=True)


def test_graph_replay_updates_all_four_inputs():
    args = make_inputs(640, 1056)
    op = torch.ops._C_ascend.npu_glm_mhc_post_310
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            op(*args)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = op(*args)
    for _ in range(3):
        for value, updated in zip(args, make_inputs(640, 1056), strict=True):
            value.copy_(updated)
        graph.replay()
        assert_parity(output, reference(*args))


def test_queued_temporary_inputs():
    saved = [tuple(value.cpu() for value in make_inputs(32, 1056)) for _ in range(12)]
    outputs = []
    for args in saved:
        outputs.append(torch.ops._C_ascend.npu_glm_mhc_post_310(*(value.npu() for value in args)))
        for value in args:
            torch.empty_like(value, device="npu").fill_(123)
    for output, args in zip(outputs, saved, strict=True):
        assert_parity(output, reference(*(value.npu() for value in args)))


@pytest.mark.parametrize("bad", ["dtype", "shape", "layout", "cpu", "alignment", "empty"])
def test_rejects_invalid_inputs(bad):
    args = list(make_inputs(4, 64))
    if bad == "dtype":
        args[1] = args[1].half()
    elif bad == "shape":
        args[2] = args[2].squeeze(-1)
    elif bad == "layout":
        args[0] = args[0].t().contiguous().t()
    elif bad == "cpu":
        args[3] = args[3].cpu()
    elif bad == "alignment":
        args[0] = args[0][:, :63].contiguous()
        args[1] = args[1][:, :, :63].contiguous()
    else:
        args = [value[:0] for value in args]
    with pytest.raises((RuntimeError, NotImplementedError)):
        torch.ops._C_ascend.npu_glm_mhc_post_310(*args)
