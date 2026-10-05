# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused native W4A8 gates, including preparation inside changing-input graphs."""

import pytest
import torch
import torch_npu

from vllm_ascend.models.qwen4_exp.w4a8_int4 import (
    pack_activation_device,
    pack_native_metadata,
    pack_native_weight,
    quantize_activation_limbs,
)
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(autouse=True, scope="module")
def require_kernel():
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()


@pytest.mark.parametrize("width", [256, 640, 2560])
@pytest.mark.parametrize("kind", ["random", "zeros", "subnormal", "outlier", "ties"])
def test_fused_pack_matches_cpu(width, kind):
    x = torch.randn(17, width, generator=torch.Generator().manual_seed(12)).half()
    if kind == "zeros":
        x.zero_()
    elif kind == "subnormal":
        x *= 0.000001
    elif kind == "outlier":
        x[:, ::128] = 65504
    elif kind == "ties":
        x = (torch.arange(17 * width) % 127 - 63.5).reshape(17, width).half()
        x[:, ::128] = 127
    expected = quantize_activation_limbs(x)
    actual = pack_activation_device(x.npu())
    for i, (value, target) in enumerate(zip(actual, expected)):
        value = value.cpu()
        if i >= 2:
            target = target[..., None].expand_as(value)
        torch.testing.assert_close(value, target, rtol=1e-6 if i == 2 else 0, atol=0)


def payload(rows, width, outputs):
    gen = torch.Generator().manual_seed(43)
    x = torch.randn(rows, width, generator=gen).half() * 0.1
    codes = torch.randint(-128, 128, (3, outputs, width // 2), dtype=torch.int8, generator=gen)
    scales = (torch.rand(3, outputs, width // 128, generator=gen) * 0.02 + 0.001).half()
    offsets = torch.randint(-8, 8, scales.shape, dtype=torch.int8, generator=gen)
    weights, sums = zip(*(pack_native_weight(w) for w in codes))
    banks = [
        torch.stack(weights),
        torch.stack([pack_native_metadata(s) for s in scales]),
        torch.stack([pack_native_metadata(z).half() for z in offsets]),
        torch.stack(sums),
    ]
    return x, [t.npu() for t in banks]


@pytest.mark.parametrize("rows", [1, 3, 15, 16, 17, 79, 80, 81, 127, 128, 129, 513])
@pytest.mark.parametrize("width,outputs", [(256, 128), (256, 640), (640, 2560), (2560, 1280)])
def test_wide_schedule_matches_native_reference(rows, width, outputs):
    x, banks = payload(rows, width, outputs)
    ends = torch.tensor([0, max(1, rows - 5), max(1, rows - 3)], dtype=torch.int64, device="npu")
    op = torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310
    # CPU packs the independent reference limbs; the old native schedule
    # consumes scalar metadata and defines the exact integer-dot contract.
    expected = op(*[v.npu() for v in quantize_activation_limbs(x)], *banks, ends).cpu()
    actual = op(*pack_activation_device(x.npu()), *banks, ends).cpu()
    torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.003)
    assert torch.count_nonzero(actual[max(1, rows - 3) :]) == 0


@pytest.mark.parametrize("experts", [31, 32, 33, 127, 128, 129])
def test_boundary_cache_crosses_blocks_without_tail_overread(experts):
    x, banks = payload(17, 256, 128)
    banks = [bank[:1].repeat(experts, 1, 1) for bank in banks]
    ends = torch.zeros(experts, dtype=torch.int64)
    ends[experts // 2 :] = 15
    ends[-1] = 16
    ends = ends.npu()
    op = torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310
    expected = op(*[v.npu() for v in quantize_activation_limbs(x)], *banks, ends).cpu()
    actual = op(*pack_activation_device(x.npu()), *banks, ends).cpu()
    torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.003)
    assert torch.count_nonzero(actual[16:]) == 0


def test_pack_meta_shapes_and_invalid_inputs():
    op = torch.ops._C_ascend.npu_qwen_w4_a8_pack_310
    output = op(torch.empty((7, 640), device="meta", dtype=torch.float16))
    assert [tuple(t.shape) for t in output] == [(7, 320), (7, 320), (7, 5, 8), (7, 5, 8)]
    for value in (
        torch.empty((2, 257), device="npu", dtype=torch.float16),
        torch.empty((2, 256), device="npu", dtype=torch.float32),
        torch.empty((0, 256), device="npu", dtype=torch.float16),
    ):
        with pytest.raises(RuntimeError, match="pack"):
            op(value)


def test_pack_accepts_full_2560_prefill_route_capacity():
    rows = 25600
    low, high, scale, total = pack_activation_device(torch.zeros((rows, 256), device="npu", dtype=torch.float16))
    assert [tuple(t.shape) for t in (low, high, scale, total)] == [
        (rows, 128),
        (rows, 128),
        (rows, 2, 8),
        (rows, 2, 8),
    ]
    # q=0 decomposes to low=-8 and high=0; two packed -8 nibbles are
    # represented by the signed byte 0x88 (-120).
    assert torch.count_nonzero(low != -120).item() == 0
    assert torch.count_nonzero(high).item() == 0
    assert torch.count_nonzero(scale != 1).item() == 0
    assert torch.count_nonzero(total).item() == 0


@pytest.mark.parametrize("rows", [3, 81, 129])
@pytest.mark.parametrize("outputs", [128, 640, 1280])
def test_pack_and_matmul_changing_input_graph(rows, outputs):
    x, banks = payload(rows, 640, outputs)
    device_x = x.npu()
    ends = torch.tensor([0, rows, rows], dtype=torch.int64, device="npu")
    op = torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310

    def invoke():
        return op(*pack_activation_device(device_x), *banks, ends)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            invoke()
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = invoke()
    for phase in range(4):
        changed = x * (phase - 1) + 0.02 * phase
        device_x.copy_(changed)
        boundaries = [min(rows // 3, rows - phase), rows - phase, rows - phase]
        ends.copy_(torch.tensor(boundaries, dtype=torch.int64))
        graph.replay()
        expected = op(*[v.npu() for v in quantize_activation_limbs(changed)], *banks, ends)
        torch.testing.assert_close(captured.cpu(), expected.cpu(), rtol=0.005, atol=0.003)


def routed_reference(x, banks, ids):
    """Independent scalar-metadata schedule with CPU sorting and unsorting."""
    experts = banks[0].shape[0]
    valid = (ids >= 0) & (ids < experts)
    routing_labels = torch.where(valid, ids.long(), experts)
    order = torch.argsort(routing_labels, stable=True)
    ends = torch.bincount(routing_labels, minlength=experts + 1)[:experts].cumsum(0).npu()
    op = torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310
    sorted_output = op(*[v.npu() for v in quantize_activation_limbs(x[order])], *banks, ends).cpu()
    return sorted_output[torch.argsort(order)]


@pytest.mark.parametrize("rows", [1, 3, 16, 30, 60, 80, 90, 120, 128])
@pytest.mark.parametrize("width,outputs", [(256, 640), (640, 2560), (2560, 1280)])
def test_native_routed_decode_matches_sorted_reference(rows, width, outputs):
    x, banks = payload(rows, width, outputs)
    ids = (torch.arange(rows, dtype=torch.int32) * 7 + 1) % 5 - 1
    actual = torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310(
        *pack_activation_device(x.npu()), *banks, ids.npu()
    ).cpu()
    torch.testing.assert_close(actual, routed_reference(x, banks, ids), rtol=0.005, atol=0.003)
    assert torch.count_nonzero(actual[(ids < 0) | (ids >= 3)]) == 0


@pytest.mark.parametrize("rows", [3, 60, 80, 120, 128])
@pytest.mark.parametrize("outputs", [128, 640, 1280, 2560])
def test_native_routed_changing_input_and_ids_graph(rows, outputs):
    x, banks = payload(rows, 640, outputs)
    device_x = x.npu()
    ids = torch.zeros(rows, dtype=torch.int32, device="npu")
    op = torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310

    def invoke():
        return op(*pack_activation_device(device_x), *banks, ids)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            invoke()
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = invoke()
    for phase in range(5):
        changed = x * (phase - 1) + 0.02 * phase
        device_x.copy_(changed)
        routes = (torch.arange(rows, dtype=torch.int32) * 7 + phase) % 5 - 1
        if phase == 3:
            routes.fill_(-1)
        ids.copy_(routes)
        graph.replay()
        torch.testing.assert_close(captured.cpu(), routed_reference(changed, banks, routes), rtol=0.005, atol=0.003)


@pytest.mark.parametrize("tokens", [1, 3, 8])
@pytest.mark.parametrize("width,outputs", [(256, 640), (640, 2560)])
def test_native_routed_reuses_packed_token_across_experts(tokens, width, outputs):
    routes_per_token = 10
    x, banks = payload(tokens, width, outputs)
    ids = (torch.arange(tokens * routes_per_token, dtype=torch.int32) * 7 + 1) % 5 - 1
    op = torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310
    actual = op(*pack_activation_device(x.npu()), *banks, ids.npu()).cpu()
    expanded = x.repeat_interleave(routes_per_token, dim=0)
    expected = op(*pack_activation_device(expanded.npu()), *banks, ids.npu()).cpu()
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.count_nonzero(actual[(ids < 0) | (ids >= 3)]) == 0


@pytest.mark.parametrize("tokens", [3, 6, 12])
@pytest.mark.parametrize("experts", [128, 129])
def test_native_routed_topk10_large_expert_bank_graph_replay(tokens, experts):
    x, small_bank = payload(tokens, 256, 640)
    banks = [bank.repeat((experts + 2) // 3, 1, 1)[:experts] for bank in small_bank]
    rows = tokens * 10
    ids_cpu = (torch.arange(rows, dtype=torch.int32) * 17) % experts
    ids_cpu[:6] = torch.tensor([0, 1, 2, 127, 128, -1], dtype=torch.int32)
    ids_cpu[-1] = experts
    device_x = x.npu()
    ids = ids_cpu.npu()
    op = torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310

    def invoke():
        return op(*pack_activation_device(device_x), *banks, ids)

    expected = routed_reference(x.repeat_interleave(10, dim=0), banks, ids_cpu)
    torch.testing.assert_close(invoke().cpu(), expected, rtol=0.005, atol=0.003)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            invoke()
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = invoke()
    for phase in range(2):
        changed = x * (0.5 + phase)
        routes = (ids_cpu + phase * 13) % (experts + 1)
        routes[phase::7] = -1
        device_x.copy_(changed)
        ids.copy_(routes)
        graph.replay()
        expected = routed_reference(changed.repeat_interleave(10, dim=0), banks, routes)
        torch.testing.assert_close(captured.cpu(), expected, rtol=0.005, atol=0.003)


def test_native_routed_reused_activation_graph_replay():
    tokens, routes_per_token = 3, 10
    x, banks = payload(tokens, 640, 1280)
    device_x = x.npu()
    ids = torch.zeros(tokens * routes_per_token, dtype=torch.int32, device="npu")
    op = torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310

    def invoke():
        return op(*pack_activation_device(device_x), *banks, ids)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            invoke()
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = invoke()
    for phase in range(4):
        changed = x * (phase - 1) + 0.02 * phase
        routes = (torch.arange(tokens * routes_per_token, dtype=torch.int32) * 7 + phase) % 5 - 1
        device_x.copy_(changed)
        ids.copy_(routes)
        graph.replay()
        expected = op(*pack_activation_device(changed.repeat_interleave(routes_per_token, dim=0).npu()), *banks, ids)
        torch.testing.assert_close(captured.cpu(), expected.cpu(), rtol=0, atol=0)


def test_model_c1_gate_up_partitioned_schedule_graph_replay():
    tokens, routes_per_token = 3, 10
    x, banks = payload(tokens, 2560, 1280)
    device_x = x.npu()
    ids = torch.zeros(tokens * routes_per_token, dtype=torch.int32, device="npu")
    op = torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310

    def invoke():
        return op(*pack_activation_device(device_x), *banks, ids)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            invoke()
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = invoke()
    for phase in range(4):
        changed = x * (phase - 1) + 0.02 * phase
        routes = (torch.arange(tokens * routes_per_token, dtype=torch.int32) * 7 + phase) % 5 - 1
        device_x.copy_(changed)
        ids.copy_(routes)
        graph.replay()
        expanded = changed.repeat_interleave(routes_per_token, dim=0).npu()
        expected = op(*pack_activation_device(expanded), *banks, ids)
        torch.testing.assert_close(captured.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("width,outputs", [(640, 2560), (2560, 1280)])
def test_model_c1_weight_pipeline_and_full_tile_fallback_graph_replay(width, outputs):
    tokens, routes_per_token = 3, 10
    x, banks = payload(tokens, width, outputs)
    device_x = x.npu()
    ids = (torch.arange(tokens * routes_per_token, dtype=torch.int32) % 3).npu()
    op = torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310

    def invoke():
        return op(*pack_activation_device(device_x), *banks, ids)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            invoke()
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = invoke()

    route_phases = (
        (torch.arange(tokens * routes_per_token, dtype=torch.int32) * 7) % 5 - 1,
        torch.tensor([0] * 17 + [1] * 6 + [2] * 4 + [-1] * 3, dtype=torch.int32),
        torch.full((tokens * routes_per_token,), -1, dtype=torch.int32),
        torch.arange(tokens * routes_per_token, dtype=torch.int32) % 3,
    )
    for phase, routes in enumerate(route_phases):
        changed = x * (phase - 1) + 0.0125 * phase
        device_x.copy_(changed)
        ids.copy_(routes)
        graph.replay()
        eager = invoke()
        torch.testing.assert_close(captured.cpu(), eager.cpu(), rtol=0, atol=0)
        expanded = changed.repeat_interleave(routes_per_token, dim=0)
        torch.testing.assert_close(captured.cpu(), routed_reference(expanded, banks, routes), rtol=0.005, atol=0.003)


def test_native_routed_rejects_nonintegral_route_factor():
    x, banks = payload(3, 256, 128)
    with pytest.raises(RuntimeError, match="unsupported native INT4 dimensions"):
        torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310(
            *pack_activation_device(x.npu()), *banks, torch.zeros(29, dtype=torch.int32, device="npu")
        )


def test_native_routed_rejects_oversized_or_scalar_metadata():
    op = torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310
    x, banks = payload(129, 256, 128)
    with pytest.raises(RuntimeError, match="routed decode"):
        op(*pack_activation_device(x.npu()), *banks, torch.zeros(129, dtype=torch.int32, device="npu"))
    with pytest.raises(RuntimeError, match="routed decode"):
        op(
            *[v.npu() for v in quantize_activation_limbs(x[:3])],
            *banks,
            torch.zeros(3, dtype=torch.int32, device="npu"),
        )


@pytest.mark.parametrize("outputs", [5120, 5248])
def test_output_shape_support_boundary(outputs):
    x, banks = payload(1, 256, outputs)
    ends = torch.tensor([0, 1, 1], dtype=torch.int64, device="npu")
    op = torch.ops._C_ascend.npu_qwen_w4_a8_int4_matmul_310
    prepared = pack_activation_device(x.npu())
    if outputs > 5120:
        with pytest.raises(RuntimeError, match="unsupported native INT4 dimensions"):
            op(*prepared, *banks, ends)
    else:
        expected = op(*[v.npu() for v in quantize_activation_limbs(x)], *banks, ends).cpu()
        torch.testing.assert_close(op(*prepared, *banks, ends).cpu(), expected, rtol=0.005, atol=0.003)
