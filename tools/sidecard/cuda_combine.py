# SPDX-License-Identifier: Apache-2.0
"""Synchronous CUDA final reduction with validated original-slot FP32 arithmetic."""

import ctypes
import math
import struct
from pathlib import Path

from .hardware_probe import DEVICE_TO_HOST, HOST_TO_DEVICE, CudaCopies
from .native_expert_probe import digest_file
from .reference import Route, validated_outputs

WIDTH = 4096
MAX_TOKENS = 32
MAX_TOP_K = 8


def pack_float_rows(rows):
    try:
        body = b"".join(struct.pack(f"<{len(row)}f", *row) for row in rows)
    except (struct.error, OverflowError) as error:
        raise ValueError("CUDA combine input exceeds FP32 range") from error
    if any(not math.isfinite(value[0]) for value in struct.iter_unpack("<f", body)):
        raise ValueError("CUDA combine input exceeds FP32 range")
    return body


class CudaCombiner:
    def __init__(self, library: Path, expected_sha, runtime: Path, max_tokens=MAX_TOKENS, max_top_k=MAX_TOP_K):
        if (
            type(max_tokens) is not int
            or not 1 <= max_tokens <= MAX_TOKENS
            or type(max_top_k) is not int
            or not 1 <= max_top_k <= MAX_TOP_K
        ):
            raise ValueError("invalid CUDA combiner bounds")
        if digest_file(library) != expected_sha:
            raise ValueError("CUDA combiner library differs from pinned identity")
        self.library = ctypes.CDLL(str(library))
        self.launch = self.library.sidecard_oracle_combine
        self.launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_float]
        self.launch.restype = ctypes.c_int
        self.cuda = CudaCopies(runtime, 1, 1)
        self.max_tokens, self.max_top_k = max_tokens, max_top_k
        self.buffers = {}
        try:
            sizes = {
                "routes": max_tokens * max_top_k * WIDTH * 4,
                "weights": max_tokens * max_top_k * 4,
                "shared": max_tokens * WIDTH * 4,
                "output": max_tokens * WIDTH * 4,
            }
            for name, size in sizes.items():
                pointer = ctypes.c_void_p()
                self.cuda.check(self.cuda.library.cudaMalloc(ctypes.byref(pointer), size))
                self.buffers[name] = pointer
        except BaseException:
            self.close()
            raise

    def __call__(self, placement, execution_id, layer_id, hidden, expert_ids, weights, replies, shared, routed_scale):
        outputs = validated_outputs(
            placement, execution_id, layer_id, hidden, expert_ids, weights, replies, shared, routed_scale
        )
        rows, top_k = len(hidden), len(expert_ids[0])
        if rows > self.max_tokens or top_k > self.max_top_k or len(hidden[0]) != WIDTH:
            raise ValueError("CUDA combine request exceeds its allocation")
        ordered = tuple(
            outputs[Route(token, slot, expert)]
            for token, ids in enumerate(expert_ids)
            for slot, expert in enumerate(ids)
        )
        data = {
            "routes": pack_float_rows(ordered),
            "weights": pack_float_rows(weights),
            "shared": pack_float_rows(shared),
        }
        scale = struct.unpack("<f", pack_float_rows(((routed_scale,),)))[0]
        for name, body in data.items():
            storage = ctypes.create_string_buffer(body, len(body))
            self.cuda.check(self.cuda.library.cudaMemcpy(self.buffers[name], storage, len(body), HOST_TO_DEVICE))
            self.cuda.check(self.cuda.library.cudaDeviceSynchronize())
        self.cuda.check(
            self.launch(
                self.buffers["routes"],
                self.buffers["weights"],
                self.buffers["shared"],
                self.buffers["output"],
                rows,
                top_k,
                WIDTH,
                scale,
            )
        )
        self.cuda.check(self.cuda.library.cudaDeviceSynchronize())
        storage = ctypes.create_string_buffer(rows * WIDTH * 4)
        self.cuda.check(self.cuda.library.cudaMemcpy(storage, self.buffers["output"], len(storage), DEVICE_TO_HOST))
        self.cuda.check(self.cuda.library.cudaDeviceSynchronize())
        result = tuple(struct.iter_unpack(f"<{WIDTH}f", storage.raw))
        if any(not math.isfinite(value) for row in result for value in row):
            raise ValueError("CUDA combine produced nonfinite output")
        return result

    def close(self):
        self.cuda.check(self.cuda.library.cudaDeviceSynchronize())
        for pointer in self.buffers.values():
            if pointer.value:
                self.cuda.check(self.cuda.library.cudaFree(pointer))
                pointer.value = None
        self.cuda.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
