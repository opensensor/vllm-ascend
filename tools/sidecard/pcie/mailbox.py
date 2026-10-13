# SPDX-License-Identifier: Apache-2.0
"""Bounded DMA invocation records and CUDA stream completion waits.

This does not submit DMA or launch Ascend work. The current 310P rejects
aclrtValueWait/Write; a separate Ascend dispatcher remains necessary.
"""

import ctypes
import struct
from dataclasses import dataclass

from ..dma_protocol import INPUT_ROW_BYTES, OUTPUT_ROW_BYTES
from ..layer_weights import MAX_COLD_PREFILL_WINDOW, NPU_LAYERS

MAGIC = b"SCMAIL01"
RECORD = struct.Struct("<8sQQIIIIQQQ")
MAX_SEQUENCE = 2**64 - 1
MAX_SLOTS = 8
CUDA_WAIT_EQUAL = 1
CUDA_ORDERING_OWNER = 100
CUDA_STREAM_MEM_OPS_64 = 122
CUDA_RDMA_WRITE_ORDERING = 118


@dataclass(frozen=True)
class Invocation:
    epoch: int
    sequence: int
    layer: int
    rows: int
    slot: int
    rank: int

    def encode(self, *, slots):
        if (
            any(type(value) is not int for value in vars(self).values())
            or not 1 <= self.epoch <= MAX_SEQUENCE
            or not 1 <= self.sequence <= MAX_SEQUENCE
            or self.layer not in NPU_LAYERS
            or not 1 <= self.rows <= MAX_COLD_PREFILL_WINDOW
            or type(slots) is not int
            or not 1 <= slots <= MAX_SLOTS
            or not 0 <= self.slot < slots
            or self.rank not in (0, 1)
        ):
            raise ValueError("DMA mailbox invocation exceeds its registered contract")
        return RECORD.pack(
            MAGIC,
            self.epoch,
            self.sequence,
            self.layer,
            self.rows,
            self.slot,
            self.rank,
            self.rows * INPUT_ROW_BYTES,
            self.rows * OUTPUT_ROW_BYTES,
            0,
        )

    @classmethod
    def decode(cls, body, *, epoch, sequence, slots, rank):
        magic, generation, number, layer, rows, slot, owner, input_bytes, output_bytes, flags = RECORD.unpack(body)
        value = cls(generation, number, layer, rows, slot, owner)
        if (
            (magic, generation, number, owner) != (MAGIC, epoch, sequence, rank)
            or flags
            or value.encode(slots=slots) != body
        ):
            raise ValueError("stale, malformed, or foreign DMA mailbox invocation")
        return value


class CudaCompletionWait:
    """Queue a wait only where the owner context can consume remote writes.

    A final completion-flag DMA must follow completed payload DMA. This class
    does not make a running polling kernel coherent or order separate DMA jobs.
    """

    def __init__(self, driver, runtime):
        if runtime.kind != "cuda":
            raise ValueError("completion wait requires a CUDA owner runtime")
        runtime.check_owner()
        self.driver, self.runtime = driver, runtime
        self.sequences, self.active, self.events = {}, {}, {}
        self.closed = self.failed = False
        attribute = driver.library.cuDeviceGetAttribute
        attribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        attribute.restype = ctypes.c_int
        self.capabilities = {}
        for name, index in (
            ("stream_mem_ops_64", CUDA_STREAM_MEM_OPS_64),
            ("rdma_write_ordering", CUDA_RDMA_WRITE_ORDERING),
        ):
            value = ctypes.c_int()
            driver.call("cuDeviceGetAttribute", ctypes.byref(value), index, runtime.device)
            self.capabilities[name] = value.value
        if not self.capabilities["stream_mem_ops_64"] or self.capabilities["rdma_write_ordering"] < CUDA_ORDERING_OWNER:
            raise RuntimeError("CUDA completion waits require 64-bit memory ops and native owner ordering")
        function = driver.library.cuStreamWaitValue64_v2
        function.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint]
        function.restype = ctypes.c_int

    def register(self, flag):
        self.runtime.check_owner()
        if self.active or self.closed or self.failed:
            raise RuntimeError("register completion resources before queuing waits")
        if flag in self.events or len(self.events) == MAX_SLOTS:
            raise ValueError("completion flag is already registered or the slot budget is full")
        if (
            flag.runtime is not self.runtime
            or flag.host
            or flag.imported
            or flag.size < ctypes.sizeof(ctypes.c_uint64)
            or flag.pointer % ctypes.sizeof(ctypes.c_uint64)
        ):
            raise ValueError("completion flag must be an aligned CUDA owner allocation")
        self.events[flag] = self.runtime.event()
        flag.leases += 1

    def enqueue(self, flag, sequence, stream):
        self.runtime.check_owner()
        if self.closed or self.failed:
            raise RuntimeError("completion queue is closed or failed")
        stream.require(self.runtime)
        if (
            flag not in self.events
            or flag.runtime is not self.runtime
            or flag.host
            or flag.imported
            or flag.size < ctypes.sizeof(ctypes.c_uint64)
            or flag.pointer % ctypes.sizeof(ctypes.c_uint64)
            or type(sequence) is not int
            or not 1 <= sequence <= MAX_SEQUENCE
        ):
            raise ValueError("completion flag must be an aligned owner allocation with a nonzero sequence")
        if flag in self.active or sequence <= self.sequences.get(flag, 0):
            raise ValueError("completion flag is leased or its sequence is stale")
        if flag not in self.sequences and len(self.sequences) == MAX_SLOTS:
            raise BufferError("completion queue exceeds its registered slot budget")
        ticket = CompletionTicket(self, flag, stream)
        self.active[flag], self.sequences[flag] = ticket, sequence
        flag.leases += 1
        try:
            self.driver.call("cuStreamWaitValue64_v2", stream.pointer, flag.pointer, sequence, CUDA_WAIT_EQUAL)
        except BaseException:
            self.failed = True
            # A submission error does not establish device quiescence.
            # Preserve the flag and event until explicit external recovery.
            raise
        return ticket

    def close(self):
        self.runtime.check_owner()
        if self.active or self.failed:
            raise RuntimeError("pending or uncertain completion waits retain their owners")
        self.sequences.clear()
        for flag, event in self.events.items():
            event.close()
            flag.leases -= 1
        self.events.clear()
        self.closed = True


class CompletionTicket:
    """Retain the flag until consumers queued on the waiting stream complete."""

    def __init__(self, queue, flag, stream):
        self.queue, self.flag, self.stream = queue, flag, stream
        self.complete = queue.events[flag]
        self.sealed = self.released = False

    def seal(self):
        self.queue.runtime.check_owner()
        if self.sealed or self.released or self.queue.failed:
            raise RuntimeError("completion consumers are sealed or uncertain")
        self.complete.record(self.stream)
        self.sealed = True

    def release(self):
        self.queue.runtime.check_owner()
        if self.released or not self.sealed or self.queue.failed or not self.complete.ready():
            raise RuntimeError("completion consumers have not established quiescence")
        self.flag.leases -= 1
        del self.queue.active[self.flag]
        self.released = True
