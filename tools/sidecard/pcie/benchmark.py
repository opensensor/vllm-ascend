# SPDX-License-Identifier: Apache-2.0
"""Raw isolated CUDA IPC -> Ascend -> CUDA round trips, without a model reload.

Two vendor-pinned buffers and event polling implement the bridge. The reference
uses AF_UNIX tensor payloads and separate driver-pinned buffers. Both tests copy
the same bytes on the same devices. They measure transport, not MoE execution.
"""

import argparse
import ctypes
import hashlib
import json
import os
import socket
import statistics
import subprocess
import time
from pathlib import Path

from .bus import Bus
from .runtime import Runtime

CAPACITY = 1048576
SLOTS = 2
SIZES = (8256, 16384, 264192, 524288, 1048576)


def send(connection, message):
    connection.sendall(json.dumps(message).encode() + b"\n")


def receive_line(connection):
    body = bytearray()
    while not body.endswith(b"\n"):
        part = connection.recv(1)
        if not part:
            raise EOFError("bridge control peer closed")
        body.extend(part)
        if len(body) > 65536:
            raise ValueError("oversized bridge control frame")
    return json.loads(body)


def receive_bytes(connection, pointer, size):
    target = memoryview((ctypes.c_ubyte * size).from_address(pointer)).cast("B")
    while target:
        count = connection.recv_into(target)
        if not count:
            raise EOFError("bridge payload peer closed")
        target = target[count:]


def wait_for(bus, transfers):
    deadline = time.monotonic() + 60
    while not all(transfer.done for transfer in transfers):
        bus.progress()
        if time.monotonic() >= deadline:
            raise TimeoutError("native bridge DMA did not complete")


def worker(arguments):
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3":
        raise ValueError("benchmark requires only assigned physical chips 2,3")
    cuda = Runtime("cuda", arguments.cuda_library)
    ascend = Runtime("ascend", arguments.ascend_library, arguments.rank)
    connection = socket.socket(fileno=arguments.fd)
    descriptor = receive_line(connection)
    sources = [cuda.import_memory(item) for item in descriptor["sources"]]
    destinations = [cuda.import_memory(item) for item in descriptor["destinations"]]
    ready = [cuda.import_event(item) for item in descriptor["ready"]]
    native = [ascend.allocate(CAPACITY) for _ in range(SLOTS)]
    bus = Bus(cuda, ascend, capacity=CAPACITY, slots=SLOTS)
    host, stream, event = ascend.allocate(CAPACITY, host=True), ascend.stream(), ascend.event()
    send(connection, {"ready": True, "physical_chip": arguments.rank + 2})
    expected = 0
    while True:
        message = receive_line(connection)
        if message["mode"] == "close":
            break
        mode, size, count = message["mode"], message["size"], message["count"]
        if message["sequence"] != expected or size not in SIZES or count not in (1, SLOTS):
            raise ValueError("invalid bridge invocation")
        if mode == "bridge":
            uploads = [
                bus.submit("cuda_to_ascend", sources[i], native[i], size, source_ready=ready[i]) for i in range(count)
            ]
            wait_for(bus, uploads)
            for ticket in uploads:
                bus.release(ticket)
            downloads = [bus.submit("ascend_to_cuda", native[i], destinations[i], size) for i in range(count)]
            wait_for(bus, downloads)
            for ticket in downloads:
                bus.release(ticket)
        elif mode == "socket":
            for i in range(count):
                receive_bytes(connection, host.pointer, size)
                ascend.copy(native[i].pointer, host.pointer, size, "h2d", stream)
                event.record(stream)
                event.wait()
                ascend.copy(host.pointer, native[i].pointer, size, "d2h", stream)
                event.record(stream)
                event.wait()
                connection.sendall(memoryview((ctypes.c_ubyte * size).from_address(host.pointer)).cast("B"))
        else:
            raise ValueError("invalid benchmark mode")
        send(connection, {"sequence": expected})
        expected += 1
    bus.close()
    for value in [*sources, *destinations, *ready, *native, event, stream, host]:
        value.close()
    send(connection, {"closed": True})
    connection.close()


def measure(arguments, rank):
    """Verify every timed round trip; preserve independent native lifetimes."""
    cuda = Runtime("cuda", arguments.cuda_library)
    sources = [cuda.allocate(CAPACITY) for _ in range(SLOTS)]
    destinations = [cuda.allocate(CAPACITY) for _ in range(SLOTS)]
    ready = [cuda.event(ipc=True) for _ in range(SLOTS)]
    host, stream, event = cuda.allocate(CAPACITY, host=True), cuda.stream(), cuda.event()
    for item in ready:
        item.record(stream)
    event.record(stream)
    event.wait()
    parent, child = socket.socketpair()
    parent.settimeout(120)
    environment = dict(os.environ)
    environment.update(json.loads(arguments.npu_environment.read_bytes()))
    environment["ASCEND_RT_VISIBLE_DEVICES"] = "2,3"
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[3]) + ":" + environment.get("PYTHONPATH", "")
    log = (arguments.output / f"worker-rank{rank}.log").open("x")
    command = [
        str(arguments.npu_python),
        "-m",
        "tools.sidecard.pcie.benchmark",
        "--worker",
        "--fd",
        str(child.fileno()),
        "--rank",
        str(rank),
        "--cuda-library",
        str(arguments.cuda_library),
        "--ascend-library",
        str(arguments.ascend_library),
    ]
    process = subprocess.Popen(
        command, env=environment, pass_fds=(child.fileno(),), stdout=log, stderr=subprocess.STDOUT
    )
    child.close()
    records, sequence = [], 0
    try:
        send(
            parent,
            dict(
                sources=[item.export() for item in sources],
                destinations=[item.export() for item in destinations],
                ready=[item.export() for item in ready],
            ),
        )
        if receive_line(parent) != {"ready": True, "physical_chip": rank + 2}:
            raise RuntimeError("bridge readiness ownership differs")
        for size in SIZES:
            for count in (1, SLOTS):
                for mode in ("socket", "bridge"):
                    samples = []
                    for iteration in range(arguments.iterations + arguments.warmup):
                        values = [(sequence + i * 37) % 251 + 1 for i in range(count)]
                        for i in range(count):
                            ctypes.memset(host.pointer, values[i], size)
                            cuda.copy(sources[i].pointer, host.pointer, size, "h2d", stream)
                            ready[i].record(stream)
                            ready[i].wait()  # Host source is reused; outside the timed transport interval.
                        begin = time.perf_counter_ns()
                        send(parent, dict(mode=mode, size=size, count=count, sequence=sequence))
                        if mode == "socket":
                            for i in range(count):
                                cuda.copy(host.pointer, sources[i].pointer, size, "d2h", stream)
                                event.record(stream)
                                event.wait()
                                parent.sendall(memoryview((ctypes.c_ubyte * size).from_address(host.pointer)).cast("B"))
                                receive_bytes(parent, host.pointer, size)
                                cuda.copy(destinations[i].pointer, host.pointer, size, "h2d", stream)
                                event.record(stream)
                                event.wait()
                        if receive_line(parent) != {"sequence": sequence}:
                            raise RuntimeError("bridge completion identity differs")
                        elapsed = (time.perf_counter_ns() - begin) / 1e6
                        for i in range(count):
                            cuda.copy(host.pointer, destinations[i].pointer, size, "d2h", stream)
                            event.record(stream)
                            event.wait()
                            if ctypes.string_at(host.pointer, size) != bytes([values[i]]) * size:
                                raise AssertionError("PCIe round trip corrupted bytes")
                        if iteration >= arguments.warmup:
                            samples.append(elapsed)
                        sequence += 1
                    records.append(
                        dict(
                            mode=mode,
                            size=size,
                            count=count,
                            samples_ms=samples,
                            mean_ms=statistics.mean(samples),
                            median_ms=statistics.median(samples),
                            min_ms=min(samples),
                            max_ms=max(samples),
                            exact=True,
                        )
                    )
        send(parent, {"mode": "close"})
        if receive_line(parent) != {"closed": True}:
            raise RuntimeError("bridge shutdown acknowledgement differs")
        process.wait(timeout=60)
        if process.returncode:
            raise RuntimeError("bridge worker failed after completion")
    finally:
        parent.close()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=60)
        log.close()
        for value in [*ready, *sources, *destinations, event, stream, host]:
            value.close()
    return dict(physical_chip=rank + 2, records=records, verified_invocations=sequence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-library", type=Path, default=Path("/usr/local/cuda/lib64/libcudart.so"))
    parser.add_argument(
        "--ascend-library", type=Path, default=Path("/usr/local/Ascend/cann-9.1.0/x86_64-linux/lib64/libascendcl.so")
    )
    parser.add_argument("--npu-environment", type=Path)
    parser.add_argument("--npu-python", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--fd", type=int)
    parser.add_argument("--rank", type=int, choices=(0, 1))
    arguments = parser.parse_args()
    if arguments.worker:
        worker(arguments)
        return
    if arguments.iterations <= 0 or arguments.warmup < 0:
        parser.error("iterations must be positive and warmup nonnegative")
    arguments.output.mkdir(exist_ok=False)
    results = [measure(arguments, rank) for rank in (0, 1)]
    source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    report = dict(
        backend="cuda-ipc-dual-pinned-host-bounce",
        cpu_payload_copies_per_direction=1,
        native_device_wide_synchronizations=0,
        direct_cross_vendor_peer_dma=False,
        iterations=arguments.iterations,
        warmup=arguments.warmup,
        source_sha256=source,
        results=results,
    )
    (arguments.output / "roundtrip.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"exact": True, "output": str(arguments.output)}), flush=True)


if __name__ == "__main__":
    main()
