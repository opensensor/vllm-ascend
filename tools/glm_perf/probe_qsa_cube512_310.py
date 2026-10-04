# SPDX-License-Identifier: Apache-2.0
"""Check 512-wide GLM QSA prefill correctness and latency on one 310P."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

HEAD_DIM = 512
QUERY_HEADS = 16
CACHE_BLOCK_SIZE = 640
GROUP_WIDTH = 512
BLOCK_TABLE_WIDTH = 308
REPEATS = 3
SCALE = 256**-0.5
SPARSE_CACHE_BLOCKS = 14
SPARSE_TAIL_START = 8000


def qsa_metadata(tokens: int, visible_lengths: torch.Tensor, *, sparse: bool = False) -> tuple[torch.Tensor, ...]:
    block_table = torch.zeros((1, BLOCK_TABLE_WIDTH), dtype=torch.int32, device="npu")
    if sparse:
        block_table[0, :SPARSE_CACHE_BLOCKS] = torch.arange(SPARSE_CACHE_BLOCKS, dtype=torch.int32, device="npu")
        groups = (torch.arange(GROUP_WIDTH, dtype=torch.int32, device="npu") * 3).expand(tokens, -1).contiguous()
        counts = torch.full((tokens,), GROUP_WIDTH, dtype=torch.int32, device="npu")
        tail_starts = torch.full((tokens,), SPARSE_TAIL_START, dtype=torch.int32, device="npu")
        tail_counts = torch.full((tokens,), 3, dtype=torch.int32, device="npu")
    else:
        block_table[0, 1] = 1
        groups = torch.zeros((tokens, GROUP_WIDTH), dtype=torch.int32, device="npu")
        counts = visible_lengths
        tail_starts = torch.zeros(tokens, dtype=torch.int32, device="npu")
        tail_counts = torch.full((tokens,), -1, dtype=torch.int32, device="npu")
    return (
        groups,
        counts,
        tail_starts,
        tail_counts,
        block_table,
        torch.tensor([0, tokens], dtype=torch.int32, device="npu"),
    )


def call_qsa(query: torch.Tensor, cache: torch.Tensor, metadata: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.ops._C_ascend.npu_qsa_sparse_attention_310(query, cache, cache, *metadata, SCALE, 4, 1)


def check_graph_replay(query: torch.Tensor, cache: torch.Tensor, metadata: tuple[torch.Tensor, ...]) -> None:
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            call_qsa(query, cache, metadata)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = call_qsa(query, cache, metadata)
    for phase in range(3):
        query.fill_((phase + 1) * 0.01)
        metadata[1].fill_(35 + phase)
        graph.replay()
        actual = captured.cpu()
        expected = call_qsa(query, cache, metadata).cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("dense", "sparse"), default="dense")
    parser.add_argument("--tokens", type=int, default=CACHE_BLOCK_SIZE)
    parser.add_argument("--graph-replay", action="store_true")
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("tokens must be positive")
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    generator = torch.Generator().manual_seed(310512)
    cache_blocks = SPARSE_CACHE_BLOCKS if args.mode == "sparse" else 4
    cache_cpu = torch.randn((cache_blocks, HEAD_DIM // 16, CACHE_BLOCK_SIZE, 16), generator=generator).half() * 0.1
    cache = torch_npu.npu_format_cast(cache_cpu.npu(), 29)
    query_cpu = torch.randn((1, QUERY_HEADS, HEAD_DIM), generator=generator).half() * 0.1
    query = query_cpu.npu()
    small_metadata = qsa_metadata(1, torch.tensor([35], dtype=torch.int32, device="npu"))
    small = call_qsa(query, cache, small_metadata).cpu()
    visible_kv = cache_cpu[0, :, :35, :].permute(1, 0, 2).reshape(35, HEAD_DIM).float()
    scores = query_cpu[0].float() @ visible_kv.T * SCALE
    expected = (torch.softmax(scores, dim=-1) @ visible_kv).half()
    torch.testing.assert_close(small[0], expected, rtol=5e-3, atol=3e-3)
    if args.graph_replay:
        check_graph_replay(query, cache, small_metadata)

    query = torch.randn((args.tokens, QUERY_HEADS, HEAD_DIM), generator=generator).half().npu() * 0.1
    lengths = torch.arange(CACHE_BLOCK_SIZE + 1, CACHE_BLOCK_SIZE + args.tokens + 1, dtype=torch.int32, device="npu")
    metadata = qsa_metadata(args.tokens, lengths, sparse=args.mode == "sparse")

    def run() -> torch.Tensor:
        return call_qsa(query, cache, metadata)

    output = run()
    torch.npu.synchronize()
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        run()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output.cpu(), args.output.with_suffix(".pt"))
    result = {
        "tokens": args.tokens,
        "mode": args.mode,
        "head_dim": HEAD_DIM,
        "small_reference": True,
        "graph_replay": args.graph_replay,
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
