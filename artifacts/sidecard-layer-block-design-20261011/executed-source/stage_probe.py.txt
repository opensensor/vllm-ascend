# SPDX-License-Identifier: Apache-2.0
"""Integrated pinned CUDA -> Ascend -> CUDA transfer of complete deferred mHC.

No model math is executed. This qualifies the TWO architecture boundaries with
real device allocations and full byte integrity. It excludes TP broadcast,
attention/state correctness and stage compute; it does not qualify model speed.
"""

import argparse
import ctypes
import hashlib
import json
import os
import socket
import struct
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from tools.raw_math.cases import summary

from .stage_wire import MAX_BODY_BYTES, descriptors, metadata, receive_into, send

READY = b"READY\n"
TIMEOUT = 60


def fixture(rows, dtype, phase):
    # FP32 residual values exceed FP16's range to catch accidental downcasting.
    fields, _ = descriptors(rows, dtype)
    data = bytearray()
    for field in fields:
        if field["dtype"] == "int64":
            data.extend(b"".join(struct.pack("<q", index + phase) for index in range(rows)))
        else:
            code = "e" if field["dtype"] == "fp16" else "f"
            value = 70000.0 + phase if field["name"] == "residual" and dtype == "fp32" else 0.5 + phase
            pattern = struct.pack("<4" + code, value, -value, 0.0, -0.0)
            data.extend(pattern * (field["bytes"] // len(pattern)))
    return bytes(data)


class CudaBuffers:
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(library))
        ptr = ctypes.c_void_p
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p
        self.lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ptr), ctypes.c_size_t]
        self.lib.cudaHostAlloc.argtypes = [ctypes.POINTER(ptr), ctypes.c_size_t, ctypes.c_uint]
        self.lib.cudaMemcpy.argtypes = [ptr, ptr, ctypes.c_size_t, ctypes.c_int]
        self.lib.cudaFree.argtypes = [ptr]
        self.lib.cudaFreeHost.argtypes = [ptr]
        self.lib.cudaDeviceSynchronize.argtypes = []
        self.device = [ptr(), ptr()]
        self.host = [ptr(), ptr()]
        try:
            self.check(self.lib.cudaSetDevice(0))
            for target in self.device:
                self.check(self.lib.cudaMalloc(ctypes.byref(target), MAX_BODY_BYTES))
            for target in self.host:
                self.check(self.lib.cudaHostAlloc(ctypes.byref(target), MAX_BODY_BYTES, 0))
            self.views = [
                memoryview((ctypes.c_ubyte * MAX_BODY_BYTES).from_address(p.value)).cast("B") for p in self.host
            ]
        except BaseException:
            self.close()
            raise

    def check(self, code):
        if code:
            raise RuntimeError(self.lib.cudaGetErrorString(code).decode())

    def copy(self, target, source, size, direction):
        self.check(self.lib.cudaMemcpy(target, source, size, direction))
        self.check(self.lib.cudaDeviceSynchronize())

    def stage(self, body):
        ctypes.memmove(self.host[0], body, len(body))
        self.copy(self.device[0], self.host[0], len(body), 1)

    def readback(self, size):
        self.copy(self.host[1], self.device[1], size, 2)
        return ctypes.string_at(self.host[1], size)

    def close(self):
        for pointer in self.device:
            if pointer.value:
                self.check(self.lib.cudaFree(pointer))
                pointer.value = None
        for pointer in self.host:
            if pointer.value:
                self.check(self.lib.cudaFreeHost(pointer))
                pointer.value = None


def worker(fd, checkpoint):
    # Lazy import keeps CUDA and CANN contexts in separate processes.
    import torch
    import torch_npu

    torch.set_num_threads(1)
    torch_npu.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    host_in = torch.empty(MAX_BODY_BYTES, dtype=torch.uint8, pin_memory=True)
    host_out = torch.empty_like(host_in, pin_memory=True)
    device = torch.empty(MAX_BODY_BYTES, dtype=torch.uint8, device="npu:0")
    torch.npu.synchronize()
    source, target = memoryview(host_in.numpy()), memoryview(host_out.numpy())
    sequence = 0
    with socket.socket(fileno=fd) as connection:
        connection.settimeout(TIMEOUT)
        connection.sendall(READY)
        while True:
            try:
                value = receive_into(connection, source, checkpoint_sha256=checkpoint, sequence=sequence)
            except EOFError:
                break
            size = value["payload_bytes"]
            device[:size].copy_(host_in[:size], non_blocking=True)
            torch.npu.synchronize()
            host_out[:size].copy_(device[:size], non_blocking=True)
            torch.npu.synchronize()
            send(connection, value, target[:size])
            sequence += 1


def probe(args):
    cuda = CudaBuffers(args.cuda_runtime)
    parent, child = socket.socketpair()
    parent.settimeout(TIMEOUT)
    process = None
    report = dict(
        completed=False,
        started_utc=datetime.now(UTC).isoformat(),
        checkpoint_sha256=args.checkpoint_sha256,
        physical_npu=2,
        pinned_host_buffers=True,
        rows=args.rows,
        samples=args.samples,
        warmups=args.warmups,
        full_model_speed_validated=False,
        model_math_executed=False,
        tp_broadcast_included=False,
        description=__doc__,
        cases=[],
        source_sha256={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("stage_probe.py", "stage_wire.py", "layer_block_design.py")
        },
    )

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save()
    try:
        with args.output.with_suffix(".log").open("x") as log:
            command = [
                str(args.npu_python),
                "-m",
                "tools.sidecard.stage_probe",
                "--worker-fd",
                str(child.fileno()),
                "--checkpoint-sha256",
                args.checkpoint_sha256,
            ]
            process = subprocess.Popen(command, pass_fds=(child.fileno(),), stdout=log, stderr=log)
            report["worker_command"] = command
            child.close()
            ready = bytearray()
            while len(ready) < len(READY):
                piece = parent.recv(len(READY) - len(ready))
                if not piece:
                    raise EOFError("NPU worker closed during startup")
                ready.extend(piece)
            if ready != READY:
                raise ValueError("NPU worker readiness differs")
            sequence = 0
            for dtype in ("fp16", "fp32"):
                for rows in args.rows:
                    size = descriptors(rows, dtype)[1]
                    samples = []
                    validations = []
                    for phase in (0, 1):
                        body = fixture(rows, dtype, phase)
                        cuda.stage(body)
                        # Gate a changed fixture before timing; all timed calls
                        # also receive a post-timer GPU readback and byte check.
                        repeats = 1 if phase == 0 else args.warmups + args.samples
                        for iteration in range(repeats):
                            value = metadata(rows, sequence, args.checkpoint_sha256, residual_dtype=dtype)
                            started = time.perf_counter()
                            cuda.copy(cuda.host[0], cuda.device[0], size, 2)
                            send(parent, value, cuda.views[0][:size])
                            reply = receive_into(
                                parent, cuda.views[1], checkpoint_sha256=args.checkpoint_sha256, sequence=sequence
                            )
                            if reply != value:
                                raise ValueError("stage response differs")
                            cuda.copy(cuda.device[1], cuda.host[1], size, 1)
                            elapsed = (time.perf_counter() - started) * 1000
                            if cuda.readback(size) != body:
                                raise AssertionError("complete GPU/NPU state transfer differs")
                            if phase == 1 and iteration >= args.warmups:
                                samples.append(elapsed)
                            sequence += 1
                        validations.append(
                            dict(phase=phase, bitwise_equal=True, sha256=hashlib.sha256(body).hexdigest())
                        )
                    report["cases"].append(
                        dict(
                            rows=rows,
                            residual_dtype=dtype,
                            payload_bytes_per_boundary=size,
                            boundaries=2,
                            total_device_copy_bytes=4 * size,
                            wall=summary(samples),
                            validations=validations,
                        )
                    )
                    save()
                    print(json.dumps(report["cases"][-1]), flush=True)
            parent.shutdown(socket.SHUT_WR)
            process.wait(timeout=TIMEOUT)
            if process.returncode:
                raise RuntimeError("NPU stage probe worker failed")
        report.update(completed=True, ended_utc=datetime.now(UTC).isoformat())
        save()
    finally:
        parent.close()
        child.close()
        if process and process.poll() is None:
            process.terminate()
            process.wait(timeout=TIMEOUT)
        cuda.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-fd", type=int)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--npu-python", type=Path)
    parser.add_argument("--cuda-runtime", type=Path)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 128, 512])
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3":
        parser.error("this experiment owns only physical NPU 2,3")
    metadata(1, 0, args.checkpoint_sha256)
    if args.worker_fd is not None:
        worker(args.worker_fd, args.checkpoint_sha256)
        return
    if args.output is None or args.output.exists() or args.npu_python is None or args.cuda_runtime is None:
        parser.error("fresh output, NPU interpreter and CUDA runtime required")
    if args.samples < 3 or args.warmups < 1:
        parser.error("at least three samples and one warmup required")
    for rows in args.rows:
        descriptors(rows)
    probe(args)


if __name__ == "__main__":
    main()
