# SPDX-License-Identifier: Apache-2.0
"""Verify direct DMA, varied payloads, poisoned lifetime guards, and red zones."""

import argparse
import ctypes as c
import errno
import fcntl
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .peer import ASCEND_TO_CUDA, CUDA_TO_ASCEND, EXECUTE, VERSION, Driver, KernelPeer, QuarantinedDMA, Request
from .runtime import Runtime

CAPACITY = 2 * 1024 * 1024
SIZES = (4096, 8256, 16384, 264192, 524288, 1048576)
NPU_RED_ZONE = 0xA5
GPU_RED_ZONE = 0x5A
GUARD_POISON = 0xD3


def pattern(size, sequence):
    period = bytes((index * 37 + sequence) % 251 for index in range(251))
    body = (period * ((size + len(period) - 1) // len(period)))[:size]
    return sequence.to_bytes(8, "little") + body[8:]


def summarize(samples):
    return dict(
        samples_ms=samples,
        median_ms=statistics.median(samples),
        mean_ms=statistics.mean(samples),
        min_ms=min(samples),
        max_ms=max(samples),
    )


def rejection_gates(peer, exported):
    checks = []
    cases = (
        ("wrong_version", dict(version=VERSION + 1), errno.EINVAL),
        ("reserved_card_0", dict(physical_chip=0), errno.EINVAL),
        ("reserved_card_1", dict(physical_chip=1), errno.EINVAL),
        ("zero_bytes", dict(length=0), errno.EINVAL),
        ("oversize", dict(length=8 * 1024 * 1024 + 1), errno.EINVAL),
        ("unknown_operation", dict(operation=99), errno.EINVAL),
        ("nonzero_reserved", dict(reserved=1), errno.EINVAL),
        ("unknown_mapping_mode", dict(gpu_mapping_mode=2), errno.EINVAL),
        ("spoofed_mapping_hit", dict(gpu_mapping_reused=1), errno.EINVAL),
    )
    for name, changes, expected in cases:
        fields = dict(version=VERSION, physical_chip=peer.physical_chip, dmabuf_fd=exported.fd, length=65536)
        fields.update(changes)
        body = bytearray(bytes(Request(**fields)))
        try:
            fcntl.ioctl(peer.fd, EXECUTE, body, True)
        except OSError as error:
            if error.errno != expected:
                raise
            checks.append(dict(case=name, errno=error.errno, rejected=True))
        else:
            raise AssertionError("kernel accepted an invalid direct request")
    body = bytearray(bytes(Request(version=VERSION, physical_chip=peer.physical_chip, dmabuf_fd=-1, length=65536)))
    fcntl.ioctl(peer.fd, EXECUTE, body, True)
    result = Request.from_buffer_copy(body)
    if result.map_status != -errno.EBADF:
        raise AssertionError("kernel accepted an invalid DMA-BUF descriptor")
    checks.append(dict(case="invalid_dma_buf_fd", errno=errno.EBADF, rejected=True))
    return checks


def mapping_gates(cuda, driver, physical_chip):
    """Prove cache identity uses DMA-BUF objects rather than reusable fd numbers."""
    allocations = [cuda.allocate(CAPACITY) for _ in range(3)]
    exports = [driver.export(allocation) for allocation in allocations]
    peer = KernelPeer(physical_chip, persistent_gpu_mappings=True)
    alias = None
    checks = []

    def mapped(fd):
        body = bytearray(
            bytes(Request(version=VERSION, physical_chip=physical_chip, dmabuf_fd=fd, length=65536, gpu_mapping_mode=1))
        )
        fcntl.ioctl(peer.fd, EXECUTE, body, True)
        return Request.from_buffer_copy(body)

    try:
        for exported in exports[:2]:
            result = mapped(exported.fd)
            if result.map_status or result.gpu_mapping_reused:
                raise AssertionError("fresh GPU mapping did not register independently")
            checks.append(dict(case="fresh_export", reused=False, passed=True))
        alias = os.dup(exports[0].fd)
        result = mapped(alias)
        if result.map_status or result.gpu_mapping_reused != 1:
            raise AssertionError("duplicate fd did not identify its original DMA-BUF")
        checks.append(dict(case="duplicate_fd_same_export", reused=True, passed=True))
        os.close(alias)
        os.dup2(exports[2].fd, alias)
        result = mapped(alias)
        if result.map_status != -errno.ENOSPC or result.gpu_mapping_reused:
            raise AssertionError("reused fd number bypassed GPU mapping capacity")
        checks.append(dict(case="reused_fd_different_export_capacity", reused=False, passed=True))
    finally:
        if alias is not None:
            os.close(alias)
        peer.close()
        for exported in exports:
            exported.close()
        for allocation in allocations:
            allocation.close()
    return checks


def worker(arguments):
    """All CPU initialization and full red-zone checks occur outside timing."""
    if os.geteuid() != 0 or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3":
        raise ValueError("direct diagnostic requires root and assigned physical chips 2,3 only")
    cuda = Runtime("cuda", arguments.cuda_library)
    ascend = Runtime("ascend", arguments.ascend_library, arguments.rank)
    driver = Driver(arguments.driver_library)
    gpu, npu = cuda.allocate(CAPACITY), ascend.allocate(CAPACITY)
    chost, nhost = cuda.allocate(CAPACITY, host=True), ascend.allocate(CAPACITY, host=True)
    guard = ascend.allocate(CAPACITY, host=True)
    cs, ns, ce, ne = cuda.stream(), ascend.stream(), cuda.event(), ascend.event()
    exported, peer = (
        driver.export(gpu),
        KernelPeer(arguments.rank + 2, persistent_gpu_mappings=arguments.persistent_gpu_mappings),
    )
    report = dict(
        pid=os.getpid(),
        physical_chip=arguments.rank + 2,
        capacity=CAPACITY,
        transport="ascend-physical-dma-nvidia-dma-buf",
        state="running",
        records=[],
        verified_copies=0,
        guard_descriptor_submitted=False,
        payload_host_staging=False,
        persistent_gpu_mappings=arguments.persistent_gpu_mappings,
    )
    output = arguments.output / f"rank{arguments.rank}.json"

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    c.memset(guard.pointer, GUARD_POISON, CAPACITY)
    sequence = 1
    save()
    try:
        for size in SIZES:
            samples = {CUDA_TO_ASCEND: [], ASCEND_TO_CUDA: []}
            completions = {}
            for iteration in range(arguments.iterations + arguments.warmup):
                source = pattern(size, sequence)
                sequence += 1
                c.memset(chost.pointer, GPU_RED_ZONE, CAPACITY)
                c.memmove(chost.pointer, source, size)
                cuda.copy(gpu.pointer, chost.pointer, CAPACITY, "h2d", cs)
                ce.record(cs)
                ce.wait()
                c.memset(nhost.pointer, NPU_RED_ZONE, CAPACITY)
                ascend.copy(npu.pointer, nhost.pointer, CAPACITY, "h2d", ns)
                ne.record(ns)
                ne.wait()
                begin = time.perf_counter_ns()
                completions[CUDA_TO_ASCEND] = peer.copy(CUDA_TO_ASCEND, exported, npu, guard, size, source_ready=ce)
                upload_ms = (time.perf_counter_ns() - begin) / 1e6
                if (
                    arguments.persistent_gpu_mappings
                    and iteration
                    and not completions[CUDA_TO_ASCEND]["gpu_mapping_reused"]
                ):
                    raise AssertionError("GPU upload mapping was not reused")
                ascend.copy(nhost.pointer, npu.pointer, CAPACITY, "d2h", ns)
                ne.record(ns)
                ne.wait()
                actual = c.string_at(nhost.pointer, CAPACITY)
                if actual[:size] != source or actual[size:] != bytes([NPU_RED_ZONE]) * (CAPACITY - size):
                    raise AssertionError("direct GPU->HBM payload or HBM red zone changed")
                report["verified_copies"] += 1
                source = pattern(size, sequence)
                sequence += 1
                c.memset(nhost.pointer, NPU_RED_ZONE, CAPACITY)
                c.memmove(nhost.pointer, source, size)
                ascend.copy(npu.pointer, nhost.pointer, CAPACITY, "h2d", ns)
                ne.record(ns)
                ne.wait()
                begin = time.perf_counter_ns()
                completions[ASCEND_TO_CUDA] = peer.copy(ASCEND_TO_CUDA, exported, npu, guard, size, source_ready=ne)
                download_ms = (time.perf_counter_ns() - begin) / 1e6
                if arguments.persistent_gpu_mappings and not completions[ASCEND_TO_CUDA]["gpu_mapping_reused"]:
                    raise AssertionError("GPU return mapping was not reused")
                cuda.copy(chost.pointer, gpu.pointer, CAPACITY, "d2h", cs)
                ce.record(cs)
                ce.wait()
                actual = c.string_at(chost.pointer, CAPACITY)
                if actual[:size] != source or actual[size:] != bytes([GPU_RED_ZONE]) * (CAPACITY - size):
                    raise AssertionError("direct HBM->GPU payload or GPU red zone changed")
                if c.string_at(guard.pointer, CAPACITY) != bytes([GUARD_POISON]) * CAPACITY:
                    raise AssertionError("unused lifetime guard was modified")
                report["verified_copies"] += 1
                if iteration >= arguments.warmup:
                    samples[CUDA_TO_ASCEND].append(upload_ms)
                    samples[ASCEND_TO_CUDA].append(download_ms)
            report["records"].append(
                dict(
                    size=size,
                    exact=True,
                    red_zones_exact=True,
                    guard_poison_exact=True,
                    cuda_to_ascend=summarize(samples[CUDA_TO_ASCEND]),
                    ascend_to_cuda=summarize(samples[ASCEND_TO_CUDA]),
                    last_completions=completions,
                )
            )
            save()
            print(
                json.dumps(
                    dict(
                        physical_chip=arguments.rank + 2,
                        size=size,
                        exact=True,
                        upload_ms=statistics.median(samples[CUDA_TO_ASCEND]),
                        download_ms=statistics.median(samples[ASCEND_TO_CUDA]),
                    )
                ),
                flush=True,
            )
        report["rejection_gates"] = rejection_gates(peer, exported)
        report["mapping_gates"] = mapping_gates(cuda, driver, arguments.rank + 2)
        report["state"] = "passed"
        save()
    except BaseException:
        if peer.failed:
            report["state"] = "quarantined"
            save()
            print(json.dumps(dict(quarantined=True, keep_alive_pid=os.getpid())), flush=True)
            # Deliberately retain contexts, allocations and device fd. Killing
            # this worker cannot establish that the hardware stopped DMA.
            while True:
                time.sleep(30)
        report["state"] = "failed"
        save()
        raise
    finally:
        peer.close()
        exported.close()
        for item in [ce, ne, cs, ns, guard, nhost, chost, npu, gpu]:
            item.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu-environment", type=Path, required=True)
    parser.add_argument("--npu-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--persistent-gpu-mappings", action="store_true")
    parser.add_argument("--rank", type=int, choices=(0, 1))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--inner", action="store_true")
    parser.add_argument("--cuda-library", type=Path, default=Path("/usr/local/cuda/lib64/libcudart.so"))
    parser.add_argument("--driver-library", default="libcuda.so.1")
    parser.add_argument(
        "--ascend-library", type=Path, default=Path("/usr/local/Ascend/cann-9.1.0/x86_64-linux/lib64/libascendcl.so")
    )
    arguments = parser.parse_args()
    if arguments.iterations <= 0 or arguments.warmup < 0:
        parser.error("iterations must be positive and warmup nonnegative")
    if arguments.worker:
        if not arguments.inner:
            environment = dict(os.environ)
            environment.update(json.loads(arguments.npu_environment.read_bytes()))
            environment["ASCEND_RT_VISIBLE_DEVICES"] = "2,3"
            source = str(Path(__file__).resolve().parents[3])
            environment["PYTHONPATH"] = source + ":" + environment.get("PYTHONPATH", "")
            os.execve(
                sys.executable,
                [sys.executable, "-m", "tools.sidecard.pcie.peer_benchmark", "--inner", *sys.argv[1:]],
                environment,
            )
        worker(arguments)
        return
    arguments.output.mkdir(exist_ok=False)
    for rank in (0, 1):
        command = [
            "sudo",
            "-n",
            str(arguments.npu_python),
            "-m",
            "tools.sidecard.pcie.peer_benchmark",
            *sys.argv[1:],
            "--worker",
            "--rank",
            str(rank),
        ]
        with (arguments.output / f"rank{rank}.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    results = [json.loads((arguments.output / f"rank{rank}.json").read_bytes()) for rank in (0, 1)]
    if not all(result["state"] == "passed" for result in results):
        raise QuarantinedDMA("direct diagnostic did not qualify both assigned chips")
    source = {
        str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.rglob("*"))
        if p.is_file() and (p.suffix in (".py", ".c", ".h") or p.name == "Makefile")
    }
    report = dict(
        results=results,
        source_sha256=source,
        iterations=arguments.iterations,
        warmup=arguments.warmup,
        direct_cross_vendor_dma=True,
        model_integrated=False,
        verified_copies=sum(result["verified_copies"] for result in results),
    )
    (arguments.output / "direct.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(direct_dma=True, verified_copies=report["verified_copies"])), flush=True)


if __name__ == "__main__":
    main()
