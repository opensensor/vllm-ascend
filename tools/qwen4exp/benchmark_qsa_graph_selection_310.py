# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate bucketed QSA graph selection against actual eager 310P operators.

Component benchmark only: no model, collective, service, or global patches.
Requires an explicitly assigned idle NPU. Every graph is retired before the
next case. Changed positions, query values and physical block tables must
preserve all valid selected IDs in order and all causal counts exactly.
"""

import argparse
import json
import statistics
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch

from tools.qwen4exp.resident_candidates.qsa_graph_selection import make_plan, select_with_plan
from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import (
    QSAGroupSelection,
    copy_group_selection_into,
    qsa_indexer_select_groups_310,
)


def assert_same_selection(actual, expected):
    """Only count-delimited IDs are consumed; unused padding is unspecified."""
    for field in ("group_counts", "tail_starts", "tail_counts"):
        torch.testing.assert_close(getattr(actual, field).cpu(), getattr(expected, field).cpu(), rtol=0, atol=0)
    counts = expected.group_counts.cpu().tolist()
    actual_ids, expected_ids = actual.group_indices.cpu(), expected.group_indices.cpu()
    for row, count in enumerate(counts):
        if count > 0:
            torch.testing.assert_close(actual_ids[row, :count], expected_ids[row, :count], rtol=0, atol=0)


def capture(function):
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            function()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = function()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    return graph, output


def paired_timing(functions, iterations, repeats):
    """Alternate comparison order; record queued wall and device-event time."""
    wall = {name: [] for name in functions}
    events = {name: [] for name in functions}
    for repeat in range(repeats):
        names = list(functions)
        if repeat % 2:
            names.reverse()
        for name in names:
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                functions[name]()
            torch.npu.synchronize()
            wall[name].append((time.perf_counter() - start) * 1000 / iterations)
            begin, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
            begin.record()
            for _ in range(iterations):
                functions[name]()
            end.record()
            end.synchronize()
            events[name].append(begin.elapsed_time(end) / iterations)
    return {
        "wall_median_ms": {name: statistics.median(values) for name, values in wall.items()},
        "event_median_ms": {name: statistics.median(values) for name, values in events.items()},
        "wall_samples_ms": wall,
        "event_samples_ms": events,
    }


@torch.inference_mode()
def main():
    # NPU-only imports remain lazy so --help and CPU contract tests do not
    # initialize the device package or claim hardware availability.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 2048, 4096, 8192, 23410, 262144])
    parser.add_argument("--queries", type=int, nargs="+", default=[3])
    parser.add_argument("--max-matmul-decode-tokens", type=int, default=8)
    parser.add_argument("--selection-policy", choices=["stable_argsort", "fast_topk"], default="stable_argsort")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or min(args.iterations, args.repeats, *args.queries, *args.lengths) < 1:
        parser.error("use a new output path and positive iterations/repeats/queries/lengths")
    if max(args.lengths) > 262144:
        parser.error("length exceeds the qualified context")
    import torch_npu

    from vllm_ascend.utils import enable_custom_op

    if not torch_npu.npu.is_available() or "310" not in torch_npu.npu.get_device_name(args.device):
        raise RuntimeError("requires an assigned idle Ascend 310P")
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    device = f"npu:{args.device}"
    generator = torch.Generator().manual_seed(31014)
    groups_per_block, table_blocks, head_dim, heads = 16, 4096, 128, 4
    tokens, requests = sum(args.queries), len(args.queries)
    cache_cpu = torch.randn((table_blocks, groups_per_block + 3, head_dim), generator=generator).half()
    cache = cache_cpu.to(device)
    query_cpu = torch.randn((tokens, heads, head_dim), generator=generator).half()
    query = query_cpu.to(device)
    starts_cpu = torch.tensor([0, *torch.tensor(args.queries).cumsum(0).tolist()], dtype=torch.int32)
    starts = starts_cpu.to(device)
    table_cpu = torch.stack([torch.randperm(table_blocks, generator=generator) for _ in args.queries]).int()
    table = table_cpu.to(device)
    positions = torch.empty(tokens, dtype=torch.int32, device=device)
    eager_destination = QSAGroupSelection(
        torch.empty((tokens, 512), dtype=torch.int32, device=device),
        torch.empty(tokens, dtype=torch.int32, device=device),
        torch.empty(tokens, dtype=torch.int32, device=device),
        torch.empty(tokens, dtype=torch.int32, device=device),
    )
    pointers = [tensor.data_ptr() for tensor in (query, cache, table, starts, positions)]
    with args.output.open("x") as output:
        for length in args.lengths:
            plan = make_plan(
                [length] * requests,
                args.queries,
                table_blocks=table_blocks,
                groups_per_block=groups_per_block,
                max_matmul_decode_tokens=args.max_matmul_decode_tokens,
                selection_policy=args.selection_policy,
            )
            state = {"length": length}

            def stage(value, revision, plan=plan, state=state):
                if not plan.admits([value] * requests, args.queries):
                    return False
                state["length"] = value
                host_positions = torch.cat([torch.arange(value - count, value) for count in args.queries]).int()
                positions.copy_(host_positions.to(device))
                table.copy_(table_cpu.roll(revision, dims=1).to(device))
                query.copy_((query_cpu if revision % 2 == 0 else -query_cpu).to(device))
                return True

            def eager(state=state):
                current = qsa_indexer_select_groups_310(
                    query,
                    cache,
                    table,
                    starts,
                    positions,
                    compress_ratio=4,
                    token_topk=2048,
                    max_visible_groups=state["length"] // 4,
                    max_matmul_decode_tokens=args.max_matmul_decode_tokens,
                    query_lens=tuple(args.queries),
                    selection_policy=args.selection_policy,
                )
                copy_group_selection_into(eager_destination, current)
                return eager_destination

            planned = partial(select_with_plan, query, cache, table, starts, positions, plan=plan)
            stage(length, 0)
            graph, graph_output = capture(planned)
            try:
                changes = 0
                # Stay in the admitted backend/capacity band. Crossing the
                # boundary is a planner failure, not a valid graph replay.
                for value in (length, max(0, length - 1), max(0, length - 3), length):
                    if not stage(value, changes):
                        continue
                    expected = eager()
                    graph.replay()
                    assert_same_selection(graph_output, expected)
                    changes += 1
                if changes < 2:
                    raise AssertionError("insufficient changing-input replay coverage")
                torch.npu.synchronize()
                timings = paired_timing(
                    {"eager_callback_selection": eager, "captured_bucket_selection": graph.replay},
                    args.iterations,
                    args.repeats,
                )
                assert pointers == [tensor.data_ptr() for tensor in (query, cache, table, starts, positions)]
                record = {
                    "scope": "actual_qsa_selection_operators_only_not_model_throughput",
                    "plan": asdict(plan),
                    "length": state["length"],
                    "dynamic_parity_replays": changes,
                    "exact_valid_ids_and_order": True,
                    "exact_group_and_tail_counts": True,
                    "stable_input_addresses": True,
                    "timing": timings,
                    "graph_count": 1,
                    "eager_callbacks_in_candidate": 0,
                }
                line = json.dumps(record)
                print(line, flush=True)
                output.write(line + "\n")
                output.flush()
            finally:
                torch.npu.synchronize()
                graph.reset()


if __name__ == "__main__":
    main()
