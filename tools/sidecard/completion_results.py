# SPDX-License-Identifier: Apache-2.0
"""Preallocated coordinator-owned CUDA results and ordered DMA consumers."""

import ctypes
import hashlib
from pathlib import Path

from .completion_protocol import CONTROL_CAPACITY, epoch, input_record
from .dma_protocol import COMMAND, CudaByteView, dma_capacity
from .layer_weights import COLD_PREFILL_WINDOW_ROWS, MAX_COLD_PREFILL_WINDOW, WIDTH
from .pcie.cuda_context import CudaContextRuntime
from .pcie.mailbox import CudaCompletionWait
from .pcie.runtime import Stream


def validate_capacity(native, rows, separate):
    """Reject an older consumer before allocating owners or submitting waits."""
    query = getattr(native, "sidecard_glm_completion_max_rows", None)
    capacity = COLD_PREFILL_WINDOW_ROWS
    if query is not None:
        query.argtypes, query.restype = [ctypes.c_uint32], ctypes.c_uint32
        capacity = query(int(separate))
    if type(rows) is not int or not 1 <= rows <= capacity <= MAX_COLD_PREFILL_WINDOW:
        raise ValueError("completion binary does not support the admitted cold row capacity")
    return capacity


class CompletionResults:
    def __init__(self, spec, nonce, driver, torch, max_chunk):
        self.torch, self.epoch = torch, epoch(nonce)
        self.input_record = input_record(spec)
        self.capacity = dma_capacity(spec)
        library = Path(spec["direct_dma"]["completion_library"])
        if hashlib.sha256(library.read_bytes()).hexdigest() != spec["direct_dma"]["completion_library_sha256"]:
            raise ValueError("GLM completion merge binary failed authentication")
        self.native = ctypes.CDLL(str(library))
        validate_capacity(self.native, max_chunk, self.input_record)
        preload = self.native.sidecard_glm_completion_preload
        preload.argtypes, preload.restype = [], ctypes.c_int
        if preload():
            raise RuntimeError("GLM completion merge preload failed before any future waits")
        self.runtime = CudaContextRuntime(spec["cuda_backend"]["cuda_runtime"], driver)
        self.queue = CudaCompletionWait(driver, self.runtime)
        self.allocations, self.exports, self.views = [], [], []
        self.controls, self.control_exports, self.control_views = [], [], []
        for _ in range(2):
            allocation = self.runtime.allocate(self.capacity)
            self.allocations.append(allocation)
            self.exports.append(driver.export(allocation))
            view = torch.as_tensor(CudaByteView(allocation), device="cuda:0")
            if view.data_ptr() != allocation.pointer:
                raise ValueError("completion result does not share its owned CUDA allocation")
            view.zero_()
            self.views.append(view)
            if self.input_record:
                control = self.runtime.allocate(CONTROL_CAPACITY)
                self.controls.append(control)
                self.control_exports.append(driver.export(control))
                control_view = torch.as_tensor(CudaByteView(control), device="cuda:0")
                control_view.zero_()
                self.control_views.append(control_view)
                self.queue.register(control)
            else:
                self.queue.register(allocation)
        self.scratch = torch.empty((max_chunk, WIDTH), dtype=torch.float32, device="cuda")
        self.merge_rank = self.native.sidecard_glm_merge_rank
        self.merge_rank.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        self.merge_rank.restype = ctypes.c_int
        if self.input_record:
            self.merge_rank = self.native.sidecard_glm_merge_separate
            self.merge_rank.argtypes = (
                [ctypes.c_void_p] * 4 + [ctypes.c_uint64] * 2 + [ctypes.c_uint32] * 3 + [ctypes.c_void_p]
            )
            self.merge_rank.restype = ctypes.c_int
            self.prepare_control = self.native.sidecard_glm_prepare_control
            self.prepare_control.argtypes = (
                [ctypes.c_void_p] * 2 + [ctypes.c_uint64] * 2 + [ctypes.c_uint32] * 3 + [ctypes.c_void_p]
            )
            self.prepare_control.restype = ctypes.c_int
        torch.cuda.synchronize()
        self.calls = self.waits = 0

    def retire(self):
        # The admission-flags CPU read has already completed preceding stream
        # consumers. Do not add a new synchronization to the model hot path.
        for ticket in list(self.queue.active.values()):
            ticket.release()

    def merge(self, hot, pending, layer, rows):
        if not pending:
            return hot
        stream = Stream(self.runtime, self.torch.cuda.current_stream().cuda_stream)
        output = self.scratch[:rows]
        current = hot
        for rank, tag in pending:
            sequence, _, _ = COMMAND.unpack(tag)
            use_input = getattr(self, "input_record", False)
            control = self.controls[rank] if use_input else self.allocations[rank]
            ticket = self.queue.enqueue(control, sequence + 1, stream)
            status = self.merge_rank(
                current.data_ptr(),
                output.data_ptr(),
                control.pointer,
                *([self.allocations[rank].pointer] if use_input else []),
                self.epoch,
                sequence + 1,
                layer,
                rows,
                rank,
                stream.pointer,
            )
            if status:
                raise RuntimeError(f"GLM completion merge launch failed: {status}")
            ticket.seal()
            self.waits += 1
            current = output
        self.calls += 1
        return output

    def prepare(self, rank, sequence, layer, rows, packed):
        status = self.prepare_control(
            self.controls[rank].pointer,
            packed.data_ptr(),
            self.epoch,
            sequence + 1,
            layer,
            rows,
            rank,
            self.torch.cuda.current_stream().cuda_stream,
        )
        if status:
            raise RuntimeError(f"CUDA input-record publication failed: {status}")

    def close(self):
        self.retire()
        self.queue.close()
        self.views.clear()
        self.control_views.clear()
        for exported in self.exports + self.control_exports:
            exported.close()
        for allocation in self.allocations + self.controls:
            allocation.close()
