# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small isolated NPU graph gate for commit decisions, not serving throughput.

Uses private tensors and no model, peer allocations, or collectives. Changing
inputs reuse their storage. Host observations occur only outside capture/replay
for the numerical gate. The allocator is capped at 64 MiB (driver context memory
is additional). No state copies or model integration are claimed by this gate.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from tools.qwen4exp.mamba_commit_plan import plan_aligned_commit

ALLOCATOR_LIMIT_BYTES = 64 * 1024 * 1024
MIN_FREE_BYTES = 768 * 1024 * 1024


def cases(block_size):
    """One stable four-row shape covering changing acceptance and boundaries."""
    for accepted in range(4):
        for offset in (-2, -1, 0, 1):
            rows = [
                (0, block_size + offset, 3, 2, accepted),
                (1, 2 * block_size + offset, 3, 2, 3 - accepted),
                (2046, 262142, 3, 2, accepted),
                (0, 0, 1, 0, accepted),
            ]
            yield tuple(torch.tensor(column, dtype=torch.int32) for column in zip(*rows))


def scalar_reference(inputs, block_size):
    result = []
    for source, computed, scheduled, draft, accepted in zip(*(value.tolist() for value in inputs)):
        running = computed + scheduled - draft
        aligned = (running + accepted - 1) // block_size * block_size
        destination = aligned // block_size - 1
        bias = aligned - running
        copy = aligned >= running and (source != destination or bias != 0)
        count = 1 if aligned >= running and source == destination else accepted
        result.append((destination if copy else 0, bias if copy else 0, copy, count))
    return tuple(torch.tensor(column) for column in zip(*result))


def assert_plan(actual, expected):
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value.cpu(), reference.to(value.dtype), rtol=0, atol=0)


@torch.inference_mode()
def gate(device, block_size, emit):
    host_cases = list(cases(block_size))
    inputs = tuple(value.to(device) for value in host_cases[0])
    addresses = tuple(value.data_ptr() for value in inputs)

    def calculate():
        return plan_aligned_commit(*inputs, block_size=block_size)

    assert_plan(calculate(), scalar_reference(host_cases[0], block_size))
    emit(dict(stage="eager_passed", block_size=block_size))
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            calculate()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = calculate()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    emit(dict(stage="captured", block_size=block_size))
    try:
        for host in host_cases:
            for destination, source in zip(inputs, host):
                destination.copy_(source)
            graph.replay()
            assert_plan(output, scalar_reference(host, block_size))
        assert addresses == tuple(value.data_ptr() for value in inputs)
        wall = {"eager": [], "graph": []}
        for repeat in range(4):
            order = (("eager", calculate), ("graph", graph.replay))
            for name, function in order if repeat % 2 == 0 else reversed(order):
                torch.npu.synchronize()
                start = time.perf_counter()
                for _ in range(20):
                    function()
                torch.npu.synchronize()
                wall[name].append((time.perf_counter() - start) * 1000 / 20)
        emit(
            dict(
                stage="passed",
                block_size=block_size,
                changing_replays=len(host_cases),
                stable_input_addresses=True,
                wall_samples_ms=wall,
                wall_median_ms={key: statistics.median(values) for key, values in wall.items()},
                scope="private_commit_decisions_only_not_model_throughput",
            )
        )
    finally:
        torch.npu.synchronize()
        graph.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output path")
    import torch_npu

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    if "310" not in torch_npu.npu.get_device_name(args.device):
        raise RuntimeError("this gate targets Ascend 310P")
    free, total = torch.npu.mem_get_info(args.device)
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"insufficient free device memory for isolated gate: {free}")
    torch.npu.set_per_process_memory_fraction(ALLOCATOR_LIMIT_BYTES / total, args.device)
    with args.output.open("x") as output:

        def emit(value):
            line = json.dumps(value)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)

        emit(dict(stage="started", device=args.device, free_bytes=free, allocator_limit_bytes=ALLOCATOR_LIMIT_BYTES))
        gate(f"npu:{args.device}", 128, emit)


if __name__ == "__main__":
    main()
