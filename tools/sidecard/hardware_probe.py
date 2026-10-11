# SPDX-License-Identifier: Apache-2.0
"""Bounded, synchronous CUDA -> host -> Ascend -> host -> CUDA copy probe.

This deliberately uses a separate byte protocol from the expert protocol:
the NPU casts and repeats synthetic activations, and never loads experts.
CUDA uses libcudart directly; torch/torch_npu are imported only in an NPU child.
"""

import argparse
import ctypes
import hashlib
import json
import math
import os
import socket
import statistics
import struct
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from .wire import unique_metadata

MAGIC = b"SCOPY001"
PREFIX = struct.Struct("!8sII")
MAX_METADATA_BYTES = 4096
HIDDEN_WIDTH = 4096
MAX_TOKENS = 640
MAX_FANOUT = 8
MAX_BODY_BYTES = MAX_TOKENS * MAX_FANOUT * HIDDEN_WIDTH * 4
HOST_TO_DEVICE = 1
DEVICE_TO_HOST = 2
DEFAULT_TIMEOUT = 60.0


def validate_shape(rows, fanout):
    if type(rows) is not int or not 1 <= rows <= MAX_TOKENS:
        raise ValueError("probe rows exceed the GLM batch limit")
    if type(fanout) is not int or not 1 <= fanout <= MAX_FANOUT:
        raise ValueError("probe fanout exceeds the GLM top-k limit")
    return rows * HIDDEN_WIDTH * 2, rows * fanout * HIDDEN_WIDTH * 4


def send_frame(connection, metadata, body, deadline):
    header = json.dumps(metadata, allow_nan=False, separators=(",", ":")).encode()
    if not 0 < len(header) <= MAX_METADATA_BYTES or len(body) > MAX_BODY_BYTES:
        raise ValueError("oversized probe frame")
    for data in (PREFIX.pack(MAGIC, len(header), len(body)), header, body):
        if not data:
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("probe send deadline")
        connection.settimeout(remaining)
        connection.sendall(data)


def receive_frame(connection, deadline):
    def read(count):
        chunks = bytearray()
        while len(chunks) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("probe receive deadline")
            connection.settimeout(remaining)
            chunk = connection.recv(min(count - len(chunks), 1024 * 1024))
            if not chunk:
                raise EOFError("probe peer closed")
            chunks.extend(chunk)
        return bytes(chunks)

    magic, header_bytes, body_bytes = PREFIX.unpack(read(PREFIX.size))
    if magic != MAGIC or not 0 < header_bytes <= MAX_METADATA_BYTES or body_bytes > MAX_BODY_BYTES:
        raise ValueError("invalid or oversized probe prefix")
    metadata = json.loads(read(header_bytes), object_pairs_hook=unique_metadata)
    if not isinstance(metadata, dict):
        raise ValueError("probe metadata must be an object")
    return metadata, read(body_bytes)


def fixture(rows, fanout):
    validate_shape(rows, fanout)
    # All values are exact in FP16, including negative zero. Vary rows as well
    # as columns to catch stale buffers and incorrect route-row repetition.
    values = (0.0, -0.0, 0.5, -0.5, 1.0, -2.0, 0.125, 32.0)
    hidden, expected = bytearray(), bytearray()
    for row in range(rows):
        rotated = values[row % len(values) :] + values[: row % len(values)]
        hidden.extend(struct.pack("<8e", *rotated) * (HIDDEN_WIDTH // len(values)))
        output_row = struct.pack("<8f", *rotated) * (HIDDEN_WIDTH // len(values))
        expected.extend(output_row * fanout)
    return bytes(hidden), bytes(expected)


class CudaCopies:
    """Own two small allocations and fence every blocking, pageable copy."""

    def __init__(self, library, max_input, max_output):
        self.library = ctypes.CDLL(str(library))
        signatures = {
            "cudaSetDevice": [ctypes.c_int],
            "cudaMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
            "cudaFree": [ctypes.c_void_p],
            "cudaMemcpy": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
            "cudaDeviceSynchronize": [],
            "cudaRuntimeGetVersion": [ctypes.POINTER(ctypes.c_int)],
            "cudaDriverGetVersion": [ctypes.POINTER(ctypes.c_int)],
        }
        for name, arguments in signatures.items():
            function = getattr(self.library, name)
            function.argtypes, function.restype = arguments, ctypes.c_int
        self.library.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.library.cudaGetErrorString.restype = ctypes.c_char_p
        self.input, self.output = ctypes.c_void_p(), ctypes.c_void_p()
        self.max_input, self.max_output = max_input, max_output
        self.check(self.library.cudaSetDevice(0))
        try:
            self.check(self.library.cudaMalloc(ctypes.byref(self.input), max_input))
            self.check(self.library.cudaMalloc(ctypes.byref(self.output), max_output))
        except BaseException:
            self.close()
            raise

    def check(self, status):
        if status:
            message = self.library.cudaGetErrorString(status).decode(errors="replace")
            raise RuntimeError(f"CUDA status {status}: {message}")

    def versions(self):
        result = {}
        for name in ("Runtime", "Driver"):
            value = ctypes.c_int()
            self.check(getattr(self.library, f"cuda{name}GetVersion")(ctypes.byref(value)))
            result[name.lower()] = value.value
        return result

    def stage(self, body, output=False):
        budget = self.max_output if output else self.max_input
        if not body or len(body) > budget:
            raise ValueError("CUDA copy exceeds its allocation")
        device = self.output if output else self.input
        source = ctypes.create_string_buffer(body, len(body))
        destination = ctypes.create_string_buffer(len(body))
        start = time.perf_counter()
        self.check(self.library.cudaMemcpy(device, source, len(body), HOST_TO_DEVICE))
        self.check(self.library.cudaDeviceSynchronize())
        uploaded = time.perf_counter()
        self.check(self.library.cudaMemcpy(destination, device, len(body), DEVICE_TO_HOST))
        self.check(self.library.cudaDeviceSynchronize())
        downloaded = time.perf_counter()
        return destination.raw, {
            "h2d_ms": (uploaded - start) * 1000,
            "d2h_ms": (downloaded - uploaded) * 1000,
        }

    def close(self):
        # Free only allocations owned by this probe. Never reset the device.
        for pointer in (self.output, self.input):
            if pointer.value:
                self.check(self.library.cudaFree(pointer))
                pointer.value = None


class NpuCopies:
    def __init__(self, device):
        # Lazy imports keep CUDA and Ascend runtime contexts in separate PIDs.
        import numpy as np
        import torch
        import torch_npu

        self.np, self.torch = np, torch
        torch.set_num_threads(1)
        torch.npu.set_device(device)
        self.device = f"npu:{device}"
        self.info = {"torch": torch.__version__, "torch_npu": torch_npu.__version__, "device": self.device}
        # Initialization and first small transfer must complete before ready.
        tensor = torch.zeros(32, dtype=torch.float16).to(self.device)
        torch.npu.synchronize()
        del tensor

    def transform(self, body, rows, fanout):
        host = self.torch.from_numpy(self.np.frombuffer(body, dtype="<f2").copy()).reshape(rows, HIDDEN_WIDTH)
        start = time.perf_counter()
        device = host.to(self.device)
        self.torch.npu.synchronize()
        uploaded = time.perf_counter()
        output = device.float().repeat_interleave(fanout, dim=0)
        self.torch.npu.synchronize()
        computed = time.perf_counter()
        result = output.cpu()
        self.torch.npu.synchronize()
        downloaded = time.perf_counter()
        body = result.numpy().astype("<f4", copy=False).tobytes()
        return body, {
            "h2d_ms": (uploaded - start) * 1000,
            "cast_repeat_ms": (computed - uploaded) * 1000,
            "d2h_ms": (downloaded - computed) * 1000,
        }


def serve_worker(connection, backend, timeout):
    send_frame(connection, {"status": "ready", "info": backend.info}, b"", time.monotonic() + timeout)
    sequence = 0
    while True:
        metadata, body = receive_frame(connection, time.monotonic() + timeout)
        if metadata == {"status": "stop"} and not body:
            send_frame(connection, {"status": "stopped"}, b"", time.monotonic() + timeout)
            return
        if set(metadata) != {"sequence", "rows", "fanout", "sha256"}:
            raise ValueError("invalid copy request fields")
        if type(metadata["sequence"]) is not int or metadata["sequence"] != sequence:
            raise ValueError("stale copy sequence")
        input_bytes, output_bytes = validate_shape(metadata["rows"], metadata["fanout"])
        if len(body) != input_bytes or hashlib.sha256(body).hexdigest() != metadata["sha256"]:
            raise ValueError("copy request shape or checksum mismatch")
        output, timings = backend.transform(body, metadata["rows"], metadata["fanout"])
        if len(output) != output_bytes:
            raise ValueError("copy backend returned the wrong byte count")
        reply = dict(metadata, sha256=hashlib.sha256(output).hexdigest(), timings=timings)
        send_frame(connection, reply, output, time.monotonic() + timeout)
        sequence += 1


def summarize(samples):
    return {
        key: {"median": statistics.median(values), "min": min(values), "max": max(values)}
        for key in samples[0]
        for values in ([sample[key] for sample in samples],)
    }


def run_probe(npu_python, cuda_library, devices, rows, fanouts, iterations, warmup, timeout, log_dir):
    if (
        not devices
        or len(set(devices)) != len(devices)
        or any(type(device) is not int or not 0 <= device < 4 for device in devices)
        or type(iterations) is not int
        or not 1 <= iterations <= 100
        or type(warmup) is not int
        or not 0 <= warmup <= 20
        or not math.isfinite(timeout)
        or timeout <= 0
        or not rows
        or not fanouts
    ):
        raise ValueError("invalid probe bounds")
    sizes = [validate_shape(row, fanout) for row in rows for fanout in fanouts]
    log_dir.mkdir(parents=True, exist_ok=False)
    cuda = CudaCopies(cuda_library, max(size[0] for size in sizes), max(size[1] for size in sizes))
    result = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "kind": "synthetic_host_staged_copy_probe",
        "cuda_versions": cuda.versions(),
        "cuda_pid": os.getpid(),
        "hidden_width": HIDDEN_WIDTH,
        "iterations": iterations,
        "warmup": warmup,
        "pinned_host_buffers": False,
        "synchronous_copies": True,
        "real_glm_inference_validated": False,
        "expert_math_validated": False,
        "capacity_admitted": False,
        "cases": [],
        "workers": [],
    }
    try:
        for device in devices:
            parent, child = socket.socketpair()
            process = None
            with (log_dir / f"npu-{device}.log").open("wb") as log:
                try:
                    command = [
                        str(npu_python),
                        "-m",
                        "tools.sidecard.hardware_probe",
                        "--worker-fd",
                        str(child.fileno()),
                        "--device",
                        str(device),
                        "--timeout",
                        str(timeout),
                    ]
                    process = subprocess.Popen(command, pass_fds=(child.fileno(),), stdout=log, stderr=log)
                    child.close()
                    ready, body = receive_frame(parent, time.monotonic() + timeout)
                    if set(ready) != {"status", "info"} or ready["status"] != "ready" or body:
                        raise ValueError("probe worker did not initialize")
                    result["workers"].append({"pid": process.pid, "device": device, "info": ready["info"]})
                    sequence = 0
                    for row in rows:
                        for fanout in fanouts:
                            hidden, expected = fixture(row, fanout)
                            samples = []
                            for iteration in range(warmup + iterations):
                                started = time.perf_counter()
                                staged, upload = cuda.stage(hidden)
                                transfer_started = time.perf_counter()
                                request = {
                                    "sequence": sequence,
                                    "rows": row,
                                    "fanout": fanout,
                                    "sha256": hashlib.sha256(staged).hexdigest(),
                                }
                                deadline = time.monotonic() + timeout
                                send_frame(parent, request, staged, deadline)
                                reply, output = receive_frame(parent, deadline)
                                transfer_finished = time.perf_counter()
                                if (
                                    set(reply) != {*request, "timings"}
                                    or any(reply[key] != request[key] for key in ("sequence", "rows", "fanout"))
                                    or reply["sha256"] != hashlib.sha256(output).hexdigest()
                                    or output != expected
                                    or staged != hidden
                                ):
                                    raise ValueError("CUDA/Ascend copy integrity check failed")
                                returned, download = cuda.stage(output, output=True)
                                finished = time.perf_counter()
                                if returned != expected:
                                    raise ValueError("CUDA returned copy integrity check failed")
                                timings = {
                                    "cuda_input_h2d_ms": upload["h2d_ms"],
                                    "cuda_input_d2h_ms": upload["d2h_ms"],
                                    "worker_exchange_ms": (transfer_finished - transfer_started) * 1000,
                                    "cuda_output_h2d_ms": download["h2d_ms"],
                                    "cuda_output_d2h_ms": download["d2h_ms"],
                                    "validated_round_trip_ms": (finished - started) * 1000,
                                }
                                timings.update({f"npu_{key}": value for key, value in reply["timings"].items()})
                                if iteration >= warmup:
                                    samples.append(timings)
                                sequence += 1
                            result["cases"].append(
                                {
                                    "device": device,
                                    "rows": row,
                                    "fanout": fanout,
                                    "input_bytes": len(hidden),
                                    "output_bytes": len(expected),
                                    "bitwise_equal": True,
                                    "timings_ms": summarize(samples),
                                }
                            )
                    deadline = time.monotonic() + timeout
                    send_frame(parent, {"status": "stop"}, b"", deadline)
                    stopped, body = receive_frame(parent, deadline)
                    if stopped != {"status": "stopped"} or body:
                        raise ValueError("worker did not acknowledge shutdown")
                finally:
                    parent.close()
                    child.close()
                    if process is not None:
                        # Closing the socket asks a failed worker to unwind. Do
                        # not hard-kill a process with an active device context.
                        try:
                            exit_code = process.wait(timeout=30)
                        except subprocess.TimeoutExpired as error:
                            raise RuntimeError(f"NPU worker {process.pid} did not exit; inspect its log") from error
                        if exit_code:
                            raise RuntimeError(f"NPU worker {process.pid} exited {exit_code}; inspect {log.name}")
    finally:
        cuda.close()
    result["transfer_hardware_validated"] = True
    result["workers_exited_cleanly"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu-python", type=Path)
    parser.add_argument("--cuda-library", type=Path, default=Path("/usr/local/cuda/lib64/libcudart.so"))
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, MAX_TOKENS])
    parser.add_argument("--fanouts", type=int, nargs="+", default=[1, MAX_FANOUT])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--worker-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--device", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_fd is not None:
        with socket.socket(fileno=args.worker_fd) as connection:
            serve_worker(connection, NpuCopies(args.device), args.timeout)
        return
    if args.npu_python is None or args.log_dir is None:
        parser.error("--npu-python and --log-dir are required")
    print(
        json.dumps(
            run_probe(
                args.npu_python,
                args.cuda_library,
                args.devices,
                args.rows,
                args.fanouts,
                args.iterations,
                args.warmup,
                args.timeout,
                args.log_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
