# SPDX-License-Identifier: Apache-2.0
"""Private grouped graph gate for TP4 and uneven TP6 state geometries.

The default uses 72 banks with four slots to bound memory. Use one bank pair
with 105/177 slots to exercise actual pool bounds. No live model banks,
collective resources or serving graphs are touched.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from tools.qwen4exp.benchmark_mamba_state_commit_310 import cases, load
from tools.qwen4exp.mamba_commit import MambaCommitGroup, reference
from tools.qwen4exp.runtime_geometry import MAX_REQUESTS, STATE_GEOMETRIES


def geometry_cases(requests, slots):
    if not 1 <= requests <= MAX_REQUESTS or slots < 4:
        raise ValueError("gate requires one through six requests and at least four slots")
    indices = torch.arange(requests) % 4
    for original, mapping in cases():
        inputs = torch.zeros_like(original)
        inputs[:, :requests] = original[:, indices]
        table = mapping[indices].clone()
        # Exercise the high end of the actual state pool. Preserve invalid IDs.
        table = torch.where(table >= 0, table % 4 + slots - 4, table)
        yield inputs, table


def host_group_copy(banks, convolution, host_inputs, host_table):
    """Match the fallback's request-first loop: decide once before all banks.

    In particular, no-copy iterations must skip every bank without tensor ops.
    Counts are already on the CPU; model metadata lookup and D2H are excluded.
    """
    for row in range(host_table.shape[0]):
        source, computed, scheduled, drafts, accepted = host_inputs[:, row].tolist()
        running = computed + scheduled - drafts
        aligned = (running + accepted - 1) // 128 * 128
        if aligned < running:
            continue
        destination, bias = aligned // 128 - 1, aligned - running
        if source == destination and bias == 0:
            continue
        for state, conv in zip(banks, convolution):
            src = int(host_table[row, source if conv else source + bias])
            dst = int(host_table[row, destination])
            selected = state[src, bias:] if conv else state[src]
            state[dst].reshape(-1)[: selected.numel()].copy_(selected.reshape(-1).clone())


@torch.inference_mode()
def gate(build, device, emit, *, requests=4, slots=4, width=2560, heads=12, pairs=36):
    if (width, heads) not in STATE_GEOMETRIES or pairs not in (1, 36):
        raise ValueError("unsupported private QFN state geometry")
    test_cases = list(geometry_cases(requests, slots))
    load(build)  # verifies frozen sources and bridge before loading either entry
    binary = Path(build) / "native_mamba_commit.bin"
    kernel = torch.classes.qwen_mamba_commit_v1.Kernel(str(binary), "qwen_mamba_commit_group_v1")
    launch = torch.ops.qwen_mamba_commit_v1.launch
    generator = torch.Generator().manual_seed(1416)
    host_banks, convolution = [], []
    for _ in range(pairs):
        host_banks.extend(
            (
                torch.randint(-32768, 32767, (slots, 5, width), dtype=torch.int16, generator=generator).view(
                    torch.float16
                ),
                torch.randint(
                    -(1 << 31), (1 << 31) - 1, (slots, heads, 128, 128), dtype=torch.int32, generator=generator
                ).view(torch.float32),
            )
        )
        convolution.extend((True, False))
    banks = [value.to(device) for value in host_banks]
    host_inputs, host_table = test_cases[3]
    inputs, table = host_inputs.to(device), host_table.to(device)
    resource = MambaCommitGroup(kernel, launch, banks, [table] * len(banks), convolution, block_size=128)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            resource(inputs)
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        resource(inputs)
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    emit(dict(stage="captured", banks=len(banks), device=str(device), launches_per_replay=1))
    try:
        checked = 0
        for revision in (0, 1, 2, 3, 16, 17, 18, 19):
            host_inputs, host_table = test_cases[revision]
            inputs.copy_(host_inputs)
            table.copy_(host_table)
            for bank, host in zip(banks, host_banks):
                bank.copy_(host)
            graph.replay()
            receipts = resource.receipt.cpu()
            for index, (bank, host, conv) in enumerate(zip(banks, host_banks, convolution)):
                expected, receipt = reference(host, host_inputs, host_table, conv=conv, block_size=128)
                torch.testing.assert_close(bank.cpu().view(torch.int32), expected.view(torch.int32), rtol=0, atol=0)
                torch.testing.assert_close(receipts[index], receipt, rtol=0, atol=0)
                checked += 1
        emit(dict(stage="parity_passed", exact_bank_checks=checked, arbitrary_fp_bits=True))
        timing = {}
        for mode in ("no_copy", "boundary_copy"):
            host_inputs, host_table = (value.clone() for value in test_cases[3])
            if mode == "no_copy":
                host_inputs[1, :requests] = 10
            inputs.copy_(host_inputs)
            table.copy_(host_table)

            def baseline(host_inputs=host_inputs, host_table=host_table):
                host_group_copy(banks, convolution, host_inputs, host_table)

            samples = {"host_copy_dispatch": [], "native_group_graph": []}
            for repeat in range(4):
                order = (("host_copy_dispatch", baseline), ("native_group_graph", graph.replay))
                for name, function in order if repeat % 2 == 0 else reversed(order):
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    for _ in range(10):
                        function()
                    torch.npu.synchronize()
                    samples[name].append((time.perf_counter() - start) * 1000 / 10)
            timing[mode] = dict(samples_ms=samples, median_ms={k: statistics.median(v) for k, v in samples.items()})
        emit(
            dict(
                stage="passed",
                exact_bank_checks=checked,
                banks=len(banks),
                slots_per_bank=slots,
                requests=requests,
                width=width,
                heads=heads,
                launches_per_replay=1,
                timing=timing,
                scope="private_group_not_model_throughput",
            )
        )
    finally:
        torch.npu.synchronize()
        graph.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--pairs", type=int, choices=(1, 36), default=36)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output path")
    import torch_npu

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    assert "310" in torch_npu.npu.get_device_name(args.device)
    free, total = torch.npu.mem_get_info(args.device)
    if free < 768 * 1024 * 1024:
        raise RuntimeError("insufficient private gate memory")
    torch.npu.set_per_process_memory_fraction(256 * 1024 * 1024 / total, args.device)
    with args.output.open("x") as output:

        def emit(value):
            line = json.dumps(value)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)

        gate(
            args.build,
            f"npu:{args.device}",
            emit,
            requests=args.requests,
            slots=args.slots,
            width=args.width,
            heads=args.heads,
            pairs=args.pairs,
        )


if __name__ == "__main__":
    main()
