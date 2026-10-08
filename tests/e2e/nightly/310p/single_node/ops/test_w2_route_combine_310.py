# SPDX-License-Identifier: Apache-2.0
"""Deferred hardware gate for FP32 weighted route reduction on Ascend 310P."""

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op


@pytest.fixture(autouse=True, scope="module")
def require_route_combine():
    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(0):
        pytest.skip("requires Ascend 310P")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    assert hasattr(torch.ops._C_ascend, "npu_w2_route_combine_310")


@pytest.mark.parametrize(
    "tokens,top_k,hidden", [(1, 1, 32), (4, 8, 4096), (9, 4, 1056), (640, 8, 4096), (1280, 8, 4096), (3, 32, 64)]
)
@pytest.mark.parametrize("local_fraction", [0.0, 0.25, 1.0])
def test_fp32_route_combine(tokens, top_k, hidden, local_fraction):
    torch.manual_seed(3102026)
    rows = tokens * top_k
    live_rows = int(rows * local_fraction)
    routed = torch.randn(rows, hidden, dtype=torch.float16)
    inverse = torch.randperm(rows)
    weights = torch.rand(tokens, top_k)
    if top_k > 1:
        weights[:, -1] = 0
    weights /= weights.sum(1, keepdim=True).clamp_min(1)
    # Poison every peer-owned row. A zero local boundary must also work.
    routed[live_rows:] = torch.nan
    if tokens > 1:
        routed[inverse[0]] = torch.nan
        weights[0, 0] = 0
    ends = torch.tensor([0, live_rows // 2, live_rows // 2, live_rows], dtype=torch.int64)
    original = routed[inverse].reshape(tokens, top_k, hidden).double()
    active = ((inverse < live_rows).reshape(tokens, top_k) & (weights != 0)).unsqueeze(-1)
    products = torch.where(active, original, 0) * weights.double().unsqueeze(-1)
    expected = products.sum(1)
    actual = torch.ops._C_ascend.npu_w2_route_combine_310(routed.npu(), inverse.npu(), weights.npu(), ends.npu()).cpu()
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    # Separate FP32 multiply/add operations can differ from torch.sum's tree.
    # Bound against an FP64 oracle using the magnitude of all products.
    bound = (2 * top_k * torch.finfo(torch.float32).eps) * products.abs().sum(1) + 1e-7
    assert torch.all((actual.double() - expected).abs() <= bound)


@pytest.mark.parametrize("bad", ["weights_dtype", "inverse_dtype", "row_count", "hidden_alignment", "different_device"])
def test_route_combine_rejects_invalid_inputs(bad):
    routed = torch.ones(8, 32, dtype=torch.float16, device="npu")
    inverse = torch.arange(8, dtype=torch.int64, device="npu")
    weights = torch.ones(1, 8, device="npu")
    ends = torch.tensor([8], dtype=torch.int64, device="npu")
    if bad == "weights_dtype":
        weights = weights.half()
    elif bad == "inverse_dtype":
        inverse = inverse.int()
    elif bad == "row_count":
        inverse = inverse[:7]
    elif bad == "hidden_alignment":
        routed = routed[:, :31].contiguous()
    else:
        weights = weights.cpu()
    with pytest.raises(RuntimeError):
        torch.ops._C_ascend.npu_w2_route_combine_310(routed, inverse, weights, ends)


def test_route_combine_graph_replays_changed_routing():
    op = torch.ops._C_ascend.npu_w2_route_combine_310
    routed = torch.ones(32, 4096, dtype=torch.float16, device="npu")
    inverse = torch.arange(32, dtype=torch.int64, device="npu")
    weights = torch.full((4, 8), 0.125, device="npu")
    ends = torch.tensor([16, 32], dtype=torch.int64, device="npu")
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            op(routed, inverse, weights, ends)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = op(routed, inverse, weights, ends)
    for live in (32, 8, 0):
        routed.copy_(torch.arange(32).unsqueeze(1).expand(32, 4096).half())
        inverse.copy_(torch.arange(31, -1, -1))
        ends.copy_(torch.tensor([0, live]))
        graph.replay()
        original = torch.arange(31, -1, -1).float()
        expected = torch.where(original < live, original, 0).reshape(4, 8).sum(1) * 0.125
        torch.testing.assert_close(captured.cpu(), expected[:, None].expand(4, 4096), rtol=0, atol=0)
