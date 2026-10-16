# SPDX-License-Identifier: Apache-2.0
"""Bounded private control messages for worker-owned direct DMA buffers."""

import json
import os
import struct

from .cold_expert_probe import receive
from .layer_weights import MAX_CHUNK, NPU_LAYERS, TOP_K, WIDTH, expert_batch_size

PROTOCOL = "glm-direct-dma-v1"
CAPACITY = 2 * 1024 * 1024
MAX_CONTROL_BYTES = 16384
LENGTH = struct.Struct("!I")
COMMAND = struct.Struct("!III")
RELEASE = b"RELEASE\n"
CLOSE_TAG = COMMAND.pack(2**32 - 1, 0, 0)
DIAGNOSTIC_TAG = COMMAND.pack(2**32 - 2, 0, 0)
INPUT_ROW_BYTES = WIDTH * 2 + TOP_K * 8
OUTPUT_ROW_BYTES = WIDTH * 4


def dma_capacity(spec):
    """Derive dedicated slot bounds from the authenticated cold batch geometry."""
    return max(CAPACITY, expert_batch_size(spec) * OUTPUT_ROW_BYTES)


def send_control(connection, value):
    body = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    if not 0 < len(body) <= MAX_CONTROL_BYTES:
        raise ValueError("direct control frame exceeds its budget")
    connection.sendall(LENGTH.pack(len(body)) + body)


def receive_control(connection):
    prefix = bytearray(LENGTH.size)
    receive(connection, memoryview(prefix))
    length = LENGTH.unpack(prefix)[0]
    if not 0 < length <= MAX_CONTROL_BYTES:
        raise ValueError("invalid direct control frame size")
    body = bytearray(length)
    receive(connection, memoryview(body))
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError("direct control frame must be an object")
    return value


def validate_identity(value, spec, nonce, rank):
    expected = {
        "protocol": PROTOCOL,
        "nonce": nonce,
        "rank": rank,
        "physical_chip": rank + 2,
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "placement_sha256": spec["expert_placement_sha256"],
        "capacity": dma_capacity(spec),
    }
    if any(value.get(key) != field for key, field in expected.items()):
        raise ValueError("direct DMA worker identity differs")


def validate_descriptor(value, *, capacity=CAPACITY):
    if (
        not isinstance(value, dict)
        or value.get("backend") != "cuda"
        or type(value.get("size")) is not int
        or value["size"] != capacity
        or type(value.get("offset")) is not int
        or value["offset"] != 0
        or not isinstance(value.get("handle"), str)
    ):
        raise ValueError("invalid dedicated CUDA IPC descriptor")
    try:
        raw = bytes.fromhex(value["handle"])
    except ValueError as error:
        raise ValueError("invalid CUDA IPC handle") from error
    if len(raw) != 64:
        raise ValueError("invalid CUDA IPC handle size")


def validate_command(tag, expected, layers=NPU_LAYERS, *, max_chunk=MAX_CHUNK):
    sequence, layer, rows = COMMAND.unpack(tag)
    if sequence != expected or layer not in layers or not 1 <= rows <= max_chunk:
        raise ValueError("invalid direct expert request identity")
    return sequence, layer, rows


class CudaByteView:
    """Borrow an IPC mapping without placing it in PyTorch's allocator."""

    def __init__(self, allocation):
        self.allocation = allocation
        self.__cuda_array_interface__ = {
            "shape": (allocation.size,),
            "strides": None,
            "typestr": "|u1",
            "data": (allocation.pointer, False),
            "version": 3,
        }


class CudaSourceEvent:
    """Process-owned event; serialized requests may use different ASGI threads.

    Record on the caller's explicit Torch stream and finish before socket
    admission. No CUDA IPC event crosses the privileged worker boundary.
    """

    def __init__(self, runtime, driver, stream):
        from .pcie.runtime import Stream

        self.pid, self.driver = os.getpid(), driver
        self.event = runtime.event()
        self.event.record(Stream(runtime, stream.cuda_stream))
        self.event.wait()

    def record(self, stream):
        if os.getpid() != self.pid or self.event.pointer is None:
            raise RuntimeError("direct source event is closed or belongs to another process")
        self.driver.call("cuEventRecord", self.event.pointer, stream.cuda_stream)

    def wait(self):
        if os.getpid() != self.pid or self.event.pointer is None:
            raise RuntimeError("direct source event is closed or belongs to another process")
        self.driver.call("cuEventSynchronize", self.event.pointer)

    def close(self):
        self.event.close()


class AdmittedCudaSource:
    """An authenticated command follows completed CUDA source work on the sender."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.admitted = False
        self.leases = 0

    def require_recorded(self):
        self.runtime.check_owner()
        if not self.admitted:
            raise RuntimeError("CUDA source has not been admitted after sender completion")

    def wait(self):
        self.require_recorded()
