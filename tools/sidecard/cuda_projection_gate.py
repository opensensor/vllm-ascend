# SPDX-License-Identifier: Apache-2.0
"""Compare optimized CUDA projection with the qualified scalar library.

Uses authenticated real W4/W3/W2 banks and records device-event and full
Python/copy timings separately. Neither selected-expert throughput nor an
exact kernel comparison establishes end-to-end serving performance.
"""

import argparse
import ctypes
import json
import struct
import time
from pathlib import Path

from .device_backend import CudaOracle, prepare_cpu_banks
from .native_expert_probe import digest_file

LAYERS = (3, 8, 33)
ROUTE_COUNTS = (1, 8, 32, 128, 256)
ACTIVATION_BITS = (4, 8)
WIDTH = 4096


class DeviceTimer:
    def __init__(self, oracle):
        self.runtime = oracle.cuda.library
        signatures = {
            "cudaEventCreate": [ctypes.POINTER(ctypes.c_void_p)],
            "cudaEventRecord": [ctypes.c_void_p, ctypes.c_void_p],
            "cudaEventSynchronize": [ctypes.c_void_p],
            "cudaEventElapsedTime": [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p],
            "cudaEventDestroy": [ctypes.c_void_p],
        }
        for name, args in signatures.items():
            method = getattr(self.runtime, name)
            method.argtypes, method.restype = args, ctypes.c_int
        self.check = oracle.cuda.check
        self.begin, self.end = ctypes.c_void_p(), ctypes.c_void_p()
        self.check(self.runtime.cudaEventCreate(ctypes.byref(self.begin)))
        self.check(self.runtime.cudaEventCreate(ctypes.byref(self.end)))
        self.original = oracle.launch
        self.elapsed_ms = None
        oracle.launch = self.launch

    def launch(self, *args):
        self.check(self.runtime.cudaEventRecord(self.begin, None))
        status = self.original(*args)
        self.check(status)
        self.check(self.runtime.cudaEventRecord(self.end, None))
        self.check(self.runtime.cudaEventSynchronize(self.end))
        elapsed = ctypes.c_float()
        self.check(self.runtime.cudaEventElapsedTime(ctypes.byref(elapsed), self.begin, self.end))
        self.elapsed_ms = elapsed.value
        return status

    def close(self):
        for event in (self.end, self.begin):
            if event.value:
                self.check(self.runtime.cudaEventDestroy(event))
                event.value = None


def output_bytes(rows):
    return b"".join(struct.pack(f"<{len(row)}f", *row) for row in rows)


def run(checkpoint, checkpoint_sha, reference, candidate, output):
    if digest_file(checkpoint / "native-layout.json") != checkpoint_sha:
        raise ValueError("CUDA projection gate checkpoint differs from pinned manifest")
    # CPU Torch only in this isolated CUDA process.
    import torch

    torch.set_num_threads(8)
    banks = prepare_cpu_banks(checkpoint, tuple((layer, 0) for layer in LAYERS), checkpoint_sha)
    oracles, timers = {}, {}
    report = {"passed": False, "checkpoint_sha256": checkpoint_sha, "cases": [], "end_to_end_speedup_validated": False}
    try:
        for label, library in (("reference", reference), ("candidate", candidate)):
            specification = dict(
                library=str(library),
                library_sha256=digest_file(library),
                cuda_runtime="/usr/local/cuda/lib64/libcudart.so",
            )
            oracles[label] = CudaOracle(banks, specification, max(ROUTE_COUNTS))
            timers[label] = DeviceTimer(oracles[label])
            report[label + "_sha256"] = specification["library_sha256"]
        generator = torch.Generator().manual_seed(20261011)
        for layer in LAYERS:
            for rows in ROUTE_COUNTS:
                hidden = tuple(
                    tuple(row) for row in torch.randn(rows, WIDTH, generator=generator).half().float().tolist()
                )
                for bits in ACTIVATION_BITS:
                    outputs, measurements = {}, {}
                    for label in ("reference", "candidate"):
                        start = time.perf_counter()
                        outputs[label] = oracles[label].compute(layer, hidden, (0,) * rows, bits)
                        measurements[label] = {
                            "device_ms": timers[label].elapsed_ms,
                            "wall_ms": (time.perf_counter() - start) * 1000,
                        }
                    equal = output_bytes(outputs["reference"]) == output_bytes(outputs["candidate"])
                    input_bytes = b"".join(struct.pack("<4096e", *row) for row in hidden)
                    start = time.perf_counter()
                    raw_output = oracles["candidate"].compute_bytes(layer, input_bytes, (0,) * rows, bits)
                    raw_ms = (time.perf_counter() - start) * 1000
                    expected_bytes = b"".join(struct.pack("<4096e", *row) for row in outputs["reference"])
                    if raw_output != expected_bytes:
                        raise AssertionError("CUDA raw tensor path differs from scalar reference")
                    measurements["candidate_raw_tensor"] = {
                        "device_ms": timers["candidate"].elapsed_ms,
                        "wall_ms": raw_ms,
                    }
                    case = dict(layer=layer, routes=rows, activation_bits=bits, bitwise_equal=equal, **measurements)
                    report["cases"].append(case)
                    print(json.dumps(case), flush=True)
                    if not equal:
                        raise AssertionError("CUDA projection candidate differs from scalar reference")
        report["passed"] = True
    finally:
        for timer in timers.values():
            timer.close()
        for oracle in oracles.values():
            oracle.close()
        output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "reference", "candidate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.checkpoint_sha256, args.reference, args.candidate, args.output)


if __name__ == "__main__":
    main()
