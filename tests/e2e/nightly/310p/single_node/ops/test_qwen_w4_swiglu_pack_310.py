# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity, graph replay, and model-shape timing for fused native-W4 SwiGLU packing."""

import time

import pytest
import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.models.qwen4_exp.w4a8_int4 import pack_activation_device, swiglu_pack_activation_device
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(autouse=True, scope="module")
def require_kernel():
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()


def reference(gate_up: torch.Tensor) -> tuple[torch.Tensor, ...]:
    gate, up = gate_up.float().chunk(2, -1)
    return pack_activation_device((F.silu(gate) * up).half())


@pytest.mark.parametrize("rows", [1, 10, 30, 80, 120, 128])
@pytest.mark.parametrize("width", [256, 640, 2560])
def test_swiglu_pack_exact(rows, width):
    generator = torch.Generator().manual_seed(rows * 10000 + width)
    gate_up = (torch.randn(rows, 2 * width, generator=generator) * 1.5).half().npu()
    expected = reference(gate_up)
    actual = swiglu_pack_activation_device(gate_up)
    for value, target in zip(actual, expected):
        torch.testing.assert_close(value.cpu(), target.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("rows", [5120, 15360, 20480])
def test_swiglu_pack_prefill_rows_exact(rows):
    width = 640
    generator = torch.Generator().manual_seed(rows)
    gate_up = (torch.randn(rows, 2 * width, generator=generator) * 1.5).half().npu()
    expected = reference(gate_up)
    actual = swiglu_pack_activation_device(gate_up)
    for value, target in zip(actual, expected):
        torch.testing.assert_close(value.cpu(), target.cpu(), rtol=0, atol=0)


def test_swiglu_pack_changing_input_graph():
    rows, width = 30, 640
    host = torch.randn(rows, 2 * width, generator=torch.Generator().manual_seed(91)).half()
    gate_up = host.npu()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            swiglu_pack_activation_device(gate_up)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = swiglu_pack_activation_device(gate_up)
    for phase in range(4):
        changed = host * (phase - 1) + phase * 0.03125
        gate_up.copy_(changed)
        graph.replay()
        expected = reference(gate_up)
        for value, target in zip(captured, expected):
            torch.testing.assert_close(value.cpu(), target.cpu(), rtol=0, atol=0)


def elapsed_ms(fn, *, warmup=20, repetitions=200):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(repetitions):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000 / repetitions


@pytest.mark.parametrize("rows", [10, 30, 120])
def test_swiglu_pack_model_geometry_benchmark(rows, record_property):
    gate_up = torch.randn(rows, 1280, device="npu", dtype=torch.float16)
    baseline_ms = elapsed_ms(lambda: reference(gate_up))
    fused_ms = elapsed_ms(lambda: swiglu_pack_activation_device(gate_up))
    record_property("rows", rows)
    record_property("baseline_ms", baseline_ms)
    record_property("fused_ms", fused_ms)
    record_property("speedup", baseline_ms / fused_ms)
    assert fused_ms < baseline_ms


def test_swiglu_pack_meta_and_rejects_oversized_rows():
    output = torch.ops._C_ascend.npu_qwen_w4_a8_swiglu_pack_310(
        torch.empty((30, 1280), device="meta", dtype=torch.float16)
    )
    assert [tuple(value.shape) for value in output] == [(30, 320), (30, 320), (30, 5, 8), (30, 5, 8)]
    with pytest.raises(RuntimeError, match="unsupported W4A8 SwiGLU pack dimensions"):
        swiglu_pack_activation_device(torch.empty((25601, 1280), device="npu", dtype=torch.float16))


def test_swiglu_pack_accepts_full_2560_prefill_route_capacity():
    rows = 25600
    packed = swiglu_pack_activation_device(torch.zeros((rows, 512), device="npu", dtype=torch.float16))
    assert [tuple(value.shape) for value in packed] == [
        (rows, 128),
        (rows, 128),
        (rows, 2, 8),
        (rows, 2, 8),
    ]
