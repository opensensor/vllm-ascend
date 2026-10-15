# SPDX-License-Identifier: Apache-2.0
"""Private graph proof for rejection-safe PLE history, independent of serving."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch


@torch.inference_mode()
def gate(module, device, emit, history_length):
    initial = list(range(1, history_length + 1))
    history = torch.tensor([initial] * 4, dtype=torch.int32, device=device)
    first = torch.full((4,), history_length + 1, dtype=torch.int64, device=device)
    samples = torch.full((4, 3), -1, dtype=torch.int64, device=device)
    metadata = torch.zeros((5, 8), dtype=torch.int32, device=device)
    metadata[1].fill_(10)
    metadata[2].fill_(3)
    metadata[3].fill_(2)

    def advance():
        accepted = module.accept_step(samples, metadata, block_size=128)
        result = module.advance_ple_history(history, first, samples, accepted)
        history.copy_(result)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            advance()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        advance()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    expected = [initial.copy() for _ in range(4)]
    host_first = torch.full((4,), history_length + 1, dtype=torch.int64)
    try:
        for step in range(32):
            host_samples = torch.full((4, 3), -1, dtype=torch.int64)
            for row in range(4):
                count = (step + row) % 4
                host_samples[row, :count] = torch.arange(count) + 100 + 4 * step + row
                if count:
                    expected[row] = [*expected[row], int(host_first[row]), *host_samples[row, : count - 1].tolist()][
                        -history_length:
                    ]
            samples.copy_(host_samples)
            first.copy_(host_first)
            graph.replay()
            torch.testing.assert_close(history.cpu(), torch.tensor(expected, dtype=torch.int32), rtol=0, atol=0)
            for row in range(4):
                count = (step + row) % 4
                if count:
                    host_first[row] = host_samples[row, count - 1]
        emit(
            dict(
                stage="passed",
                device=device,
                chained_replays=32,
                request_history_checks=128,
                history_length=history_length,
                host_decisions_inside_graph=0,
                scope="private_ple_history_not_serving",
            )
        )
    finally:
        torch.npu.synchronize()
        graph.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-length", type=int, choices=(2, 3), default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output path")
    spec = importlib.util.spec_from_file_location("private_iteration_transition", args.source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import torch_npu

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    if "310" not in torch_npu.npu.get_device_name(args.device):
        raise RuntimeError("requires Ascend 310P")
    free, total = torch.npu.mem_get_info(args.device)
    if free < 768 * 1024 * 1024:
        raise RuntimeError("insufficient memory for private graph")
    torch.npu.set_per_process_memory_fraction(64 * 1024 * 1024 / total, args.device)
    with args.output.open("x") as output:

        def emit(value):
            text = json.dumps(value)
            print(text, flush=True)
            output.write(text + "\n")
            output.flush()

        gate(module, f"npu:{args.device}", emit, args.history_length)


if __name__ == "__main__":
    main()
