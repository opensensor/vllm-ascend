# SPDX-License-Identifier: Apache-2.0
"""Private device-selected PLE round trip through a native host gather.

The producer graph is synchronized before native DMA. This measures that fence
explicitly; it is not an asynchronous serving integration or a graph host node.
One parent owns one immutable shared table across every worker and trial.
"""

import argparse
import ctypes
import fcntl
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from tools.qwen4exp.ple_request import PLEDeviceRequest, PLERequestGeometry, load
from tools.qwen4exp.ple_sources import IMMUTABLE_SEALS, FileSource, SharedRamSource
from tools.qwen4exp.ple_storage import register

PARAMETER = "/sys/module/drv_pcie_host/parameters/dma_query_spin_us"
SHAPES = (1, 3, 12)
ITERATIONS = 32


def distribution(values):
    values = sorted(values)
    return {
        "count": len(values),
        "p50_ns": statistics.median(values),
        "p95_ns": values[max(0, (95 * len(values) + 99) // 100 - 1)],
        "max_ns": max(values),
    }


def expected_rows(tokens, history, starts, geometry):
    result = []
    for request, (start, end) in enumerate(zip(starts, starts[1:])):
        for position in range(start, end):
            local = position - start
            current = tokens[position]
            recent = tokens[position - 1] if local else history[request][1]
            older = tokens[position - 2] if local >= 2 else history[request][local]
            if recent == geometry.eos:
                older = geometry.eos
            first = current * geometry.multipliers[0] ^ recent * geometry.multipliers[1]
            result.extend(
                (first ^ (older * geometry.multipliers[2] if head >= 8 else 0)) % size + offset
                for head, (size, offset) in enumerate(zip(geometry.sizes, geometry.offsets))
            )
    return result


def worker(args):
    # Framework and ACL are initialized only in each independent worker.
    import mmap
    import random

    import torch
    import torch_npu  # noqa: F401

    metadata = json.loads(args.metadata.read_text())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.npu.set_device(args.rank)
    torch.npu.set_compile_mode(jit_compile=False)
    kernel, launch = load(args.build)
    library = ctypes.CDLL(str(args.library.resolve()))
    library.ple_host_create.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.ple_host_create.restype = ctypes.c_int
    library.ple_host_fetch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
    library.ple_host_fetch.restype = ctypes.c_int
    library.ple_host_close.argtypes = [ctypes.c_void_p]
    library.ple_host_close.restype = ctypes.c_int
    if args.native_producer:
        library.ple_host_set_producer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
        ]
        library.ple_host_set_producer.restype = ctypes.c_int
        library.ple_host_produce_fetch.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        library.ple_host_produce_fetch.restype = ctypes.c_int
        library.ple_host_produce_fetch_poll.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint]
        library.ple_host_produce_fetch_poll.restype = ctypes.c_int
    identity = os.fstat(args.fd)
    rng = random.Random(15102026)
    checkpoint = (
        FileSource(register(Path(metadata["model"]), index_filename="model.safetensors.index.json"))
        if metadata.get("model")
        else None
    )
    with mmap.mmap(args.fd, identity.st_size, access=mmap.ACCESS_READ) as table, args.output.open("x") as log:
        for count in SHAPES:
            requests = 4 if count == 12 else 1
            geometry = PLERequestGeometry(tokens=count, requests=requests, **metadata["geometry"])
            owner = PLEDeviceRequest(kernel, launch, geometry, f"npu:{args.rank}")
            # Keep the existing kernel ABI, but place both outputs in one stable
            # allocation so the native consumer needs one request readback.
            packet = torch.empty(count * 160, dtype=torch.uint8, device=f"npu:{args.rank}")
            owner.rows = packet[: count * 128].view(torch.int64).view(count, 16)
            owner.receipt = packet[count * 128 :].view(torch.int32).view(count, 8)
            output = torch.empty((count, 16, 320), dtype=torch.uint8, device=packet.device)
            tokens = torch.empty(count, dtype=torch.int64, device=packet.device)
            history = torch.empty((requests, 2), dtype=torch.int64, device=packet.device)
            host_starts = [i * (count // requests) for i in range(requests + 1)]
            starts = torch.tensor(host_starts, dtype=torch.int32, device=packet.device)
            tokens.fill_(0)
            history.fill_(geometry.eos)
            stream = torch.npu.Stream(device=packet.device)
            stream.wait_stream(torch.npu.current_stream())
            graph = torch.npu.NPUGraph()
            with torch.npu.stream(stream):
                for _ in range(3):
                    owner(tokens, starts, history)
                torch.npu.synchronize()
                with torch.npu.graph(graph, stream=stream):
                    owner(tokens, starts, history)
                torch.npu.synchronize()
            torch.npu.current_stream().wait_stream(stream)
            torch.npu.synchronize()
            # Pin only the calling thread after runtime/compiler initialization.
            # Pinning the entire Python process starves its helper processes.
            os.sched_setaffinity(0, {args.cpu})
            handle = ctypes.c_void_p()
            status = library.ple_host_create(
                args.rank, args.fd, count, packet.data_ptr(), output.data_ptr(), ctypes.byref(handle)
            )
            if status:
                raise RuntimeError(f"create native transport: {status}")
            timing = (ctypes.c_uint64 * 8)()
            try:
                if args.native_producer:
                    addresses = (ctypes.c_void_p * 7)(
                        *(
                            tensor.data_ptr()
                            for tensor in (
                                tokens,
                                starts,
                                history,
                                owner.config,
                                owner.rows,
                                owner.offsets,
                                owner.receipt,
                            )
                        )
                    )
                    binary = json.loads((args.build / "provenance.json").read_text())["binary"]["path"]
                    status = library.ple_host_set_producer(handle, os.fsencode(binary), addresses, min(8, count))
                    if status:
                        raise RuntimeError(f"admit native producer: {status}")
                modes = (2, 3, 4) if args.native_only else (0, 1, 2, 3, 4) if args.native_producer else (0, 1)
                paired_cases = None
                for mode in modes:
                    samples = []
                    poll_completions = 0
                    poll_queries = 0
                    caller_cpu = []
                    # Equal inputs across modes, changing requests across calls.
                    mode_rng = random.Random(rng.randrange(1 << 30))
                    cases = []
                    for iteration in range(ITERATIONS + 4):
                        host_tokens = [mode_rng.randrange(geometry.vocabulary) for _ in range(count)]
                        host_history = [
                            [mode_rng.randrange(geometry.vocabulary) for _ in range(2)] for _ in range(requests)
                        ]
                        if iteration % 8 == 0:
                            host_history[0][1] = geometry.eos
                        cases.append((host_tokens, host_history))
                    if paired_cases is None:
                        paired_cases = cases
                    else:
                        cases = paired_cases
                    for iteration, (host_tokens, host_history) in enumerate(cases):
                        tokens.copy_(torch.tensor(host_tokens, dtype=torch.int64))
                        history.copy_(torch.tensor(host_history, dtype=torch.int64))
                        torch.npu.synchronize()
                        cpu_start = time.thread_time_ns()
                        start = time.perf_counter_ns()
                        if mode >= 2:
                            if mode >= 3:
                                status = library.ple_host_produce_fetch_poll(handle, timing, 20 if mode == 3 else 40)
                            else:
                                status = library.ple_host_produce_fetch(handle, timing)
                            producer_ns = timing[4]
                        else:
                            graph.replay()
                            torch.npu.synchronize()
                            producer_ns = time.perf_counter_ns() - start
                            status = library.ple_host_fetch(handle, mode, timing)
                        finished = time.perf_counter_ns()
                        cpu_elapsed = time.thread_time_ns() - cpu_start
                        if status:
                            raise RuntimeError(f"native fetch: {status}")
                        rows = expected_rows(host_tokens, host_history, host_starts, geometry)
                        assert owner.rows.cpu().reshape(-1).tolist() == rows
                        expected = b"".join(table[row * 320 : (row + 1) * 320] for row in rows)
                        if checkpoint is not None and iteration in (0, ITERATIONS + 3):
                            assert checkpoint.read_rows(rows) == expected
                        assert output.cpu().numpy().tobytes() == expected
                        if iteration >= 4:
                            samples.append([producer_ns, *timing[:4], finished - start])
                            caller_cpu.append(cpu_elapsed)
                            if mode >= 2:
                                poll_queries += timing[6]
                                poll_completions += timing[7]
                    names = (
                        "producer_and_fence",
                        "request_readback",
                        "validation_and_gather",
                        "response",
                        "native_transport",
                        "producer_fence_transport_python",
                    )
                    record = {
                        "rank": args.rank,
                        "caller_cpu_id": args.cpu,
                        "tokens": count,
                        "mode": (
                            "packed",
                            "direct_rows",
                            "native_packed",
                            "native_poll20_packed",
                            "native_poll40_packed",
                        )[mode],
                        "producer": "native_launch" if mode >= 2 else "torch_graph",
                        "producer_poll_us": (20 if mode == 3 else 40) if mode >= 3 else 0,
                        "producer_poll_completions": poll_completions,
                        "producer_poll_queries": poll_queries,
                        "caller_cpu": distribution(caller_cpu),
                        "shared_inode": identity.st_ino,
                        "table_bytes": identity.st_size,
                        "spin_us": int(Path(PARAMETER).read_text()),
                        "checked_requests": len(cases),
                        "metrics": {
                            name: distribution([sample[i] for sample in samples]) for i, name in enumerate(names)
                        },
                        "exact": True,
                    }
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(json.dumps(record), flush=True)
                # Failed device receipt must reject the entire request, preserve
                # prior output, and leave the transport reusable.
                previous = output.cpu().numpy().tobytes()
                tokens.fill_(-1)
                graph.replay()
                torch.npu.synchronize()
                assert library.ple_host_fetch(handle, 0, timing) == -22
                assert output.cpu().numpy().tobytes() == previous
                if args.native_producer:
                    assert library.ple_host_produce_fetch(handle, timing) == -22
                    assert output.cpu().numpy().tobytes() == previous
                tokens.fill_(0)
                graph.replay()
                torch.npu.synchronize()
                assert library.ple_host_fetch(handle, 0, timing) == 0
                previous = output.cpu().numpy().tobytes()
                owner.rows[-1, -1] = geometry.table_rows
                torch.npu.synchronize()
                assert library.ple_host_fetch(handle, 0, timing) == -34
                assert output.cpu().numpy().tobytes() == previous
                graph.replay()
                torch.npu.synchronize()
                assert library.ple_host_fetch(handle, 0, timing) == 0
                log.write(
                    json.dumps(
                        {
                            "rank": args.rank,
                            "tokens": count,
                            "invalid_receipt_rejected": True,
                            "invalid_row_rejected": True,
                            "output_preserved": True,
                            "recovered": True,
                        }
                    )
                    + "\n"
                )
            finally:
                torch.npu.synchronize()
                status = library.ple_host_close(handle)
                if status:
                    print(
                        f"close native transport: {status}; terminating with buffers retained",
                        file=sys.stderr,
                        flush=True,
                    )
                    os._exit(2)
                graph.reset()
    if checkpoint is not None:
        checkpoint.close()


def parent(args):
    args.output.mkdir(parents=True, exist_ok=False)
    source = None
    fd = None
    original_budget = Path(PARAMETER).read_text().strip()
    try:
        if args.synthetic:
            fd = os.memfd_create("ple-gather-synthetic", os.MFD_ALLOW_SEALING)
            os.write(fd, bytes((i * 17 + 13) % 251 for i in range(64 * 320)))
            fcntl.fcntl(fd, fcntl.F_ADD_SEALS, IMMUTABLE_SEALS)
            geometry = dict(
                eos=127,
                vocabulary=256,
                multipliers=[113, 167, 191],
                sizes=[4] * 16,
                offsets=list(range(0, 64, 4)),
                table_rows=64,
            )
        else:
            from safetensors import safe_open

            storage = register(args.model, index_filename="model.safetensors.index.json")
            weights = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
            cfg = json.loads((args.model / "config.json").read_text())["text_config"]
            geometry = dict(
                eos=cfg["eos_token_id"], vocabulary=cfg["vocab_size"], table_rows=storage.table_bytes // 320
            )
            for name, suffix in (
                ("multipliers", "layer_multipliers"),
                ("sizes", "ngram_heads_vocab_sizes"),
                ("offsets", "ngram_heads_offsets"),
            ):
                key = next(key for key in weights if key.endswith("ple_embedding." + suffix))
                with safe_open(args.model / weights[key], framework="numpy") as file:
                    geometry[name] = file.get_tensor(key).reshape(-1).tolist()
            print(json.dumps({"stage": "loading_shared_table", "bytes": storage.table_bytes}), flush=True)
            started = time.monotonic()
            source = SharedRamSource.load(storage, budget_bytes=storage.table_bytes, reserve_bytes=48 * 1024**3)
            fd = source.duplicate_fd()
            print(json.dumps({"stage": "shared_table_ready", "seconds": time.monotonic() - started}), flush=True)
        metadata = args.output / "metadata.json"
        metadata.write_text(
            json.dumps(
                {
                    "geometry": geometry,
                    "synthetic": args.synthetic,
                    "model": str(args.model) if args.model else None,
                    "shared_inode": os.fstat(fd).st_ino,
                    "table_bytes": os.fstat(fd).st_size,
                },
                indent=2,
            )
        )
        for budget in args.spin_budgets:
            phase = "baseline" if budget == 0 else "spin5"
            subprocess.run(
                ["sudo", "tee", PARAMETER], input=f"{budget}\n", text=True, stdout=subprocess.DEVNULL, check=True
            )
            for rank in args.devices:
                command = [
                    sys.executable,
                    "-m",
                    "tools.qwen4exp.benchmark_ple_host_gather_310",
                    "--worker",
                    "--fd",
                    str(fd),
                    "--rank",
                    str(rank),
                    "--cpu",
                    str(args.cpu),
                    "--metadata",
                    str(metadata),
                    "--build",
                    str(args.build),
                    "--library",
                    str(args.library),
                    "--output",
                    str(args.output / f"{phase}-rank{rank}.jsonl"),
                ]
                if args.native_producer:
                    command.append("--native-producer")
                if args.native_only:
                    command.append("--native-only")
                with (
                    (args.output / f"{phase}-rank{rank}.stdout").open("w") as stdout,
                    (args.output / f"{phase}-rank{rank}.stderr").open("w") as stderr,
                ):
                    subprocess.run(command, pass_fds=(fd,), stdout=stdout, stderr=stderr, check=True, timeout=120)
                print(json.dumps({"stage": "worker_passed", "phase": phase, "rank": rank}), flush=True)
    finally:
        subprocess.run(
            ["sudo", "tee", PARAMETER], input=original_budget + "\n", text=True, stdout=subprocess.DEVNULL, check=True
        )
        if fd is not None:
            os.close(fd)
        if source is not None:
            source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--native-producer", action="store_true")
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--spin-budgets", type=int, nargs="+", choices=(0, 5), default=[0, 5])
    parser.add_argument("--devices", type=int, nargs="+", choices=range(4), default=[0, 1, 2, 3])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--fd", type=int)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    if args.native_only:
        args.native_producer = True
    if args.worker:
        worker(args)
    else:
        if args.synthetic == (args.model is not None):
            parser.error("select exactly one of --synthetic or --model")
        parent(args)


if __name__ == "__main__":
    main()
