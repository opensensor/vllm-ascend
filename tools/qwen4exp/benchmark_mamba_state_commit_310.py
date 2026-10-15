# SPDX-License-Identifier: Apache-2.0
"""Private native commit/copy and graph gate; no serving modifications."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from tools.qwen4exp.benchmark_mamba_commit_plan_310 import ALLOCATOR_LIMIT_BYTES, MIN_FREE_BYTES
from tools.qwen4exp.mamba_commit import MambaStateCommit, reference


def load(build):
    root = Path(build).resolve(strict=True)
    manifest = json.loads((root / "provenance.json").read_text())
    assert manifest["namespace"] == "qwen_mamba_commit_v1" and manifest["flush_host_queue"] is False
    for name, digest in manifest["sources"].items():
        assert Path(name).name == name and hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    for item in [manifest["bridge"], *manifest["binaries"].values()]:
        path = Path(item["path"]).resolve(strict=True)
        assert path.parent == root and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    torch.ops.load_library(manifest["bridge"]["path"])
    namespace = manifest["namespace"]
    kernel = getattr(torch.classes, namespace).Kernel(manifest["binaries"]["native_mamba_commit"]["path"], namespace)
    return kernel, getattr(torch.ops, namespace).launch


def cases():
    for revision in range(20):
        inputs = torch.zeros((5, 8), dtype=torch.int32)
        inputs[:, :4] = torch.tensor(
            [
                [0, 1, 2, 0],
                [126 + revision % 3, 254 + revision % 3, 382, 10],
                [3, 3, 3, 3],
                [2, 2, 2, 2],
                [revision % 4, (revision + 1) % 4, 3, 2],
            ],
            dtype=torch.int32,
        )
        # Changing physical mappings, including overlapping requests. CPU and
        # native implementation must both execute rows in the original order.
        table = ((torch.arange(4 * 8).reshape(4, 8) + revision) % 24).int()
        if revision == 16:
            table.fill_(-1)  # active rows fail bounds; inactive row must not read
        elif revision == 17:
            inputs[4, 0] = 4  # invalid acceptance count
        elif revision == 18:
            inputs[0, 2] = 8  # invalid logical source
        elif revision == 19:
            table[:, :] = torch.arange(8)  # cross-request aliases
        yield inputs, table


def host_driven_copy(state, host_inputs, host_table, *, conv):
    """Existing clone/copy semantics with counts already observed on CPU.

    Excludes the old D2H and real model metadata lookup, so this comparison is
    only the copy dispatch component, not the entire CPU fallback's latency.
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
        src = int(host_table[row, source if conv else source + bias])
        dst = int(host_table[row, destination])
        selected = state[src, bias:] if conv else state[src]
        state[dst].reshape(-1)[: selected.numel()].copy_(selected.reshape(-1).clone())


@torch.inference_mode()
def gate(kernel, launch, device, *, conv, large, emit):
    generator = torch.Generator().manual_seed(1415)
    shape = (24, 5, 2048 if large else 256) if conv else (24, 196608 if large else 8192)
    host_state = torch.randint(-(1 << 31), (1 << 31) - 1, shape, dtype=torch.int32, generator=generator)
    host_inputs, host_table = next(cases())
    state, inputs, table = (value.to(device) for value in (host_state, host_inputs, host_table))
    resource = MambaStateCommit(kernel, launch, state, table, conv=conv, block_size=128)
    pointers = [value.data_ptr() for value in (state, inputs, table)]
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
    emit(dict(stage="captured", conv=conv, shape=shape))
    checks = 0
    try:
        for host_inputs, host_table in cases():
            expected_state, expected_receipt = reference(host_state, host_inputs, host_table, conv=conv, block_size=128)
            inputs.copy_(host_inputs)
            table.copy_(host_table)
            for name, execute in (("eager", lambda: resource(inputs)), ("graph", graph.replay)):
                state.copy_(host_state)
                execute()
                torch.testing.assert_close(state.cpu(), expected_state, rtol=0, atol=0)
                torch.testing.assert_close(resource.receipt.cpu(), expected_receipt, rtol=0, atol=0)
                checks += 1
        assert pointers == [value.data_ptr() for value in (state, inputs, table)]
        # Fixed valid input for paired timings, outside host copies/assertions.
        host_inputs, host_table = list(cases())[3]
        inputs.copy_(host_inputs)
        table.copy_(host_table)
        baseline = lambda: host_driven_copy(state, host_inputs, host_table, conv=conv)
        state.copy_(host_state)
        baseline()
        expected_state, _ = reference(host_state, host_inputs, host_table, conv=conv, block_size=128)
        torch.testing.assert_close(state.cpu(), expected_state, rtol=0, atol=0)
        samples = {"host_copy_dispatch": [], "native_eager": [], "native_graph": []}
        for repeat in range(4):
            functions = (
                ("host_copy_dispatch", baseline),
                ("native_eager", lambda: resource(inputs)),
                ("native_graph", graph.replay),
            )
            for name, function in functions if repeat % 2 == 0 else reversed(functions):
                torch.npu.synchronize()
                start = time.perf_counter()
                for _ in range(20):
                    function()
                torch.npu.synchronize()
                samples[name].append((time.perf_counter() - start) * 1000 / 20)
        emit(
            dict(
                stage="passed",
                conv=conv,
                shape=shape,
                exact_checks=checks,
                stable_addresses=True,
                wall_samples_ms=samples,
                wall_median_ms={name: statistics.median(values) for name, values in samples.items()},
                scope="private_single_state_bank_not_model_throughput",
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
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output path")
    import torch_npu

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    assert "310" in torch_npu.npu.get_device_name(args.device)
    free, total = torch.npu.mem_get_info(args.device)
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"insufficient memory for private gate: {free}")
    torch.npu.set_per_process_memory_fraction(ALLOCATOR_LIMIT_BYTES / total, args.device)
    kernel, launch = load(args.build)
    with args.output.open("x") as output:

        def emit(value):
            line = json.dumps(value)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)

        for conv, large in ((True, False), (True, True), (False, False), (False, True)):
            gate(kernel, launch, f"npu:{args.device}", conv=conv, large=large, emit=emit)
            torch.npu.empty_cache()


if __name__ == "__main__":
    main()
