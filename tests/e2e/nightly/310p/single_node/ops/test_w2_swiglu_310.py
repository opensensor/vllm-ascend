# SPDX-License-Identifier: Apache-2.0
"""Parity, invalid-input, and graph checks for GLM's FP16/FP32 SwiGLU."""

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op


@pytest.fixture(autouse=True, scope="module")
def require_swiglu():
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    assert hasattr(torch.ops._C_ascend, "npu_w2_swiglu_310")


def reference(gate_up):
    gate, up = gate_up.chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate.float()) * up.float()).half()


@pytest.mark.parametrize("rows,width", [(1, 32), (32, 1056), (72, 2048), (5120, 2048), (10240, 2048), (20480, 2048)])
def test_swiglu_parity(rows, width):
    torch.manual_seed(3102026)
    data = (torch.randn(rows, 2 * width) * 3).half().npu()
    actual = torch.ops._C_ascend.npu_w2_swiglu_310(data)
    expected = reference(data)
    assert actual.dtype == torch.float16 and actual.is_contiguous()
    # FP32 Exp/Div may round differently from the framework SiLU. Require
    # one FP16 relative step plus subnormal allowance before serving tests.
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-3, atol=2e-6)


def test_swiglu_extreme_finite_inputs():
    values = torch.tensor([-65504, -100, -20, -1, -0.0, 0, 1, 20, 100, 65504], dtype=torch.float16)
    gate = values.repeat(4)[:32].reshape(1, 32)
    up = torch.tensor([0.0, 0.5, -1.0, 2.0], dtype=torch.float16).repeat(8).reshape(1, 32)
    data = torch.cat((gate, up), dim=-1).npu()
    torch.testing.assert_close(
        torch.ops._C_ascend.npu_w2_swiglu_310(data).cpu(), reference(data).cpu(), rtol=1e-3, atol=2e-6
    )


def test_swiglu_temporary_inputs_survive_queued_launches():
    """Drop inputs immediately and reuse their allocation size before syncing."""
    torch.manual_seed(3102026)
    inputs = [(torch.randn(72, 4096) * 3).half() for _ in range(16)]
    expected = [reference(data) for data in inputs]
    outputs = []
    for data in inputs:
        outputs.append(torch.ops._C_ascend.npu_w2_swiglu_310(data.npu()))
        # If a queued command retains only an address, this same-size allocation
        # can overwrite the temporary input before its kernel consumes it.
        torch.empty_like(data, device="npu").fill_(123)
    for actual, reference_output in zip(outputs, expected):
        torch.testing.assert_close(actual.cpu(), reference_output, rtol=1e-3, atol=2e-6)


@pytest.mark.parametrize("bad", ["dtype", "rank", "alignment", "noncontiguous", "cpu"])
def test_swiglu_rejects_invalid_inputs(bad):
    data = torch.ones(8, 128, dtype=torch.float16, device="npu")
    if bad == "dtype":
        data = data.float()
    elif bad == "rank":
        data = data.unsqueeze(0)
    elif bad == "alignment":
        data = data[:, :126].contiguous()
    elif bad == "noncontiguous":
        data = data[:, ::2]
    else:
        data = data.cpu()
    with pytest.raises((RuntimeError, NotImplementedError)):
        torch.ops._C_ascend.npu_w2_swiglu_310(data)


def test_swiglu_graph_replay_changes_input():
    op = torch.ops._C_ascend.npu_w2_swiglu_310
    data = torch.ones(32, 4096, dtype=torch.float16, device="npu")
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            op(data)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = op(data)
    for value in (-3.0, 0.0, 2.0):
        data.fill_(value)
        graph.replay()
        torch.testing.assert_close(output.cpu(), reference(data).cpu(), rtol=1e-3, atol=2e-6)
