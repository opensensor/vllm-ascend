# SPDX-License-Identifier: Apache-2.0
"""GLM 512-wide paged QSA reference and changing-input graph regression."""

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

HEAD_DIM = 512
QUERY_HEADS = 16
BLOCK_SIZE = 640
SCALE = 256**-0.5


def test_glm_qsa_512_dense_reference_and_graph_replay() -> None:
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    generator = torch.Generator().manual_seed(310512)
    cache_cpu = torch.randn((2, HEAD_DIM // 16, BLOCK_SIZE, 16), generator=generator).half() * 0.1
    cache = torch_npu.npu_format_cast(cache_cpu.npu(), 29)
    query_cpu = torch.randn((1, QUERY_HEADS, HEAD_DIM), generator=generator).half() * 0.1
    query = query_cpu.npu()
    groups = torch.zeros((1, 512), dtype=torch.int32, device="npu")
    visible = torch.tensor([35], dtype=torch.int32, device="npu")
    tail_starts = torch.zeros(1, dtype=torch.int32, device="npu")
    tail_counts = torch.full((1,), -1, dtype=torch.int32, device="npu")
    block_table = torch.tensor([[0, 1]], dtype=torch.int32, device="npu")
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device="npu")
    op = torch.ops._C_ascend.npu_qsa_sparse_attention_310

    def attention() -> torch.Tensor:
        return op(
            query,
            cache,
            cache,
            groups,
            visible,
            tail_starts,
            tail_counts,
            block_table,
            query_start_loc,
            SCALE,
            4,
            1,
        )

    actual = attention().cpu()[0]
    keys = cache_cpu[0, :, :35, :].permute(1, 0, 2).reshape(35, HEAD_DIM).float()
    expected = (torch.softmax(query_cpu[0].float() @ keys.T * SCALE, dim=-1) @ keys).half()
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=3e-3)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            attention()
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = attention()
    for phase in range(3):
        query.fill_((phase + 1) * 0.01)
        visible.fill_(35 + phase)
        graph.replay()
        torch.testing.assert_close(captured.cpu(), attention().cpu(), rtol=0, atol=0)
