# SPDX-License-Identifier: Apache-2.0
"""Bitwise BF16-rounding parity and graph replay on Ascend 310P."""

import statistics
import time

import pytest
import torch
import torch_npu

from vllm_ascend.patch.worker.patch_mhc_norm import _round_mhc_state
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(autouse=True, scope="module")
def require_kernel():
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    assert hasattr(torch.ops._C_ascend, "mhc_bf16_round_310")


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor):
    assert torch.equal(actual.cpu().view(torch.int32), expected.cpu().view(torch.int32))


@pytest.mark.parametrize("count", [1, 4, 16, 4116, 8195, 16384, 32768, 65536])
def test_round_matches_native_bf16(count: int):
    torch.manual_seed(count)
    values = torch.randn(count, dtype=torch.float32)
    if count >= 4:
        values[:4] = torch.tensor([0x3F808000, 0x3F818000, 0x3F800001, 0x7F7F8000], dtype=torch.int32).view(
            torch.float32
        )
    x = values.npu()
    actual = torch.ops._C_ascend.mhc_bf16_round_310(x)
    expected = x.to(torch.bfloat16).to(torch.float32)
    _assert_bitwise_equal(actual, expected)


def test_round_rejects_wrong_dtype():
    with pytest.raises(RuntimeError, match="contiguous float32"):
        torch.ops._C_ascend.mhc_bf16_round_310(torch.ones(16, dtype=torch.float16, device="npu"))


def test_round_matches_native_nan_and_infinity_bits():
    payloads = [1, 2, 0x7FFF, 0x8000, 0xFFFF, 0x10000, 0x20000, 0x3FFFFF, 0x400000, 0x7FFFFF]
    unsigned = [0x7F800000, 0xFF800000] + [
        sign | 0x7F800000 | payload for sign in (0, 0x80000000) for payload in payloads
    ]
    signed = [value if value < 2**31 else value - 2**32 for value in unsigned]
    x = torch.tensor(signed, dtype=torch.int32).view(torch.float32).npu()
    actual = torch.ops._C_ascend.mhc_bf16_round_310(x)
    expected = x.to(torch.bfloat16).to(torch.float32)
    _assert_bitwise_equal(actual, expected)


def test_round_matches_native_finite_boundary_bits():
    unsigned = [
        0x00000000,
        0x80000000,
        0x00000001,
        0x00007FFF,
        0x00008000,
        0x00010000,
        0x007FFFFF,
        0x00800000,
        0x807FFFFF,
        0x80800000,
        0x7F7FFFFF,
        0xFF7FFFFF,
    ]
    signed = [value if value < 2**31 else value - 2**32 for value in unsigned]
    x = torch.tensor(signed, dtype=torch.int32).view(torch.float32).npu()
    actual = torch.ops._C_ascend.mhc_bf16_round_310(x)
    expected = x.to(torch.bfloat16).to(torch.float32)
    _assert_bitwise_equal(actual, expected)


def test_mhc_opt_in_round_uses_custom_op():
    x = torch.randn(4116, dtype=torch.float32, device="npu")
    _assert_bitwise_equal(_round_mhc_state(x, use_ai_core=True), x.to(torch.bfloat16).to(torch.float32))


@pytest.mark.parametrize("count", [4116, 65536])
def test_round_changing_input_graph(count: int):
    x = torch.randn(count, dtype=torch.float32, device="npu")
    op = torch.ops._C_ascend.mhc_bf16_round_310
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            op(x)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = op(x)
    for phase in range(4):
        x.copy_(torch.randn(count, dtype=torch.float32) + phase)
        graph.replay()
        expected = x.to(torch.bfloat16).to(torch.float32)
        _assert_bitwise_equal(captured, expected)


@pytest.mark.parametrize("count", [4116, 16384, 65536])
def test_round_eager_latency_below_native_cast_pair(count: int):
    x = torch.randn(count, dtype=torch.float32, device="npu")
    paths = {
        "ai_core": lambda: torch.ops._C_ascend.mhc_bf16_round_310(x),
        "native": lambda: x.to(torch.bfloat16).to(torch.float32),
    }
    timings = {}
    for name, path in paths.items():
        for _ in range(5):
            path()
        torch.npu.synchronize()
        samples = []
        for _ in range(20):
            start = time.perf_counter()
            path()
            torch.npu.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        timings[name] = statistics.median(samples)
    print(f"mHC round elements={count} ai_core={timings['ai_core']:.4f} ms native={timings['native']:.4f} ms")
    assert timings["ai_core"] < timings["native"]
