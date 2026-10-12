# SPDX-License-Identifier: Apache-2.0
"""Experimental direct GPU/HBM DMA through the restricted kernel bridge.

The Ascend DMA engine consumes an NVIDIA-exported DMA-BUF scatterlist and
local HBM physical addresses. The lifetime descriptor for the unrelated ACL
host guard is never submitted. No tensor payload traverses host RAM here.
This initial API waits for a source event and performs a blocking ioctl.
"""

import ctypes as c
import fcntl
import os
import threading

VERSION = 5
MAX_GPU_MAPPINGS = 2
MAP_ONLY = 0
BAR_PIN_ONLY = 1
LOCAL_PIN_ONLY = 2
CUDA_TO_ASCEND = 3
ASCEND_TO_CUDA = 4
MAX_BYTES = 8 * 1024 * 1024
CUDA_DMA_BUF_FD = 1
CUDA_MEMORY_TYPE_DEVICE = 2
CUDA_POINTER_MEMORY_TYPE = 2
CUDA_POINTER_SYNC_MEMOPS = 6
CUDA_POINTER_IS_MANAGED = 8
CUDA_POINTER_RDMA_CAPABLE = 15
CUDA_FLUSH_CURRENT_CONTEXT = 0
CUDA_FLUSH_OWNER = 100
CUDA_FLUSH_ALL_DEVICES = 200


class Request(c.Structure):
    _fields_ = [
        ("version", c.c_uint32),
        ("physical_chip", c.c_uint32),
        ("dmabuf_fd", c.c_int32),
        ("operation", c.c_uint32),
        ("npu_va", c.c_uint64),
        ("length", c.c_uint64),
        ("pin_length", c.c_uint64),
        ("guard_va", c.c_uint64),
        ("map_status", c.c_int32),
        ("pin_status", c.c_int32),
        ("dma_status", c.c_int32),
        ("gpu_segments", c.c_uint32),
        ("npu_page_size", c.c_uint64),
        ("gpu_mapped_bytes", c.c_uint64),
        ("revoked", c.c_uint32),
        ("cache_id", c.c_uint32),
        ("retained", c.c_uint32),
        ("reserved", c.c_uint32),
        ("gpu_mapping_mode", c.c_uint32),
        ("gpu_mapping_reused", c.c_uint32),
    ]


EXECUTE = (3 << 30) | (c.sizeof(Request) << 16) | (ord("S") << 8) | 0x40


class Driver:
    def __init__(self, library="libcuda.so.1"):
        self.library = c.CDLL(str(library))
        signatures = {
            "cuPointerGetAttribute": [c.c_void_p, c.c_int, c.c_uint64],
            "cuPointerSetAttribute": [c.c_void_p, c.c_int, c.c_uint64],
            "cuMemGetHandleForAddressRange": [c.c_void_p, c.c_uint64, c.c_size_t, c.c_int, c.c_uint64],
            "cuFlushGPUDirectRDMAWrites": [c.c_int, c.c_int],
            "cuEventRecord": [c.c_void_p, c.c_void_p],
            "cuEventSynchronize": [c.c_void_p],
        }
        for name, arguments in signatures.items():
            function = getattr(self.library, name)
            function.argtypes, function.restype = arguments, c.c_int

    def call(self, name, *arguments):
        result = getattr(self.library, name)(*arguments)
        if result:
            raise RuntimeError(f"{name} failed with CUDA driver status {result}")

    def export(self, allocation):
        if allocation.runtime.kind != "cuda" or allocation.host or allocation.imported:
            raise ValueError("direct export requires an owned dedicated CUDA allocation")
        if allocation.leases:
            raise ValueError("CUDA allocation is already leased or exported")
        for attribute, expected in (
            (CUDA_POINTER_MEMORY_TYPE, CUDA_MEMORY_TYPE_DEVICE),
            (CUDA_POINTER_IS_MANAGED, 0),
            (CUDA_POINTER_RDMA_CAPABLE, 1),
        ):
            value = c.c_uint()
            self.call("cuPointerGetAttribute", c.byref(value), attribute, allocation.pointer)
            if value.value != expected:
                raise ValueError("CUDA allocation is not dedicated RDMA-capable device memory")
        flag = c.c_uint(1)
        self.call("cuPointerSetAttribute", c.byref(flag), CUDA_POINTER_SYNC_MEMOPS, allocation.pointer)
        fd = c.c_int(-1)
        self.call("cuMemGetHandleForAddressRange", c.byref(fd), allocation.pointer, allocation.size, CUDA_DMA_BUF_FD, 0)
        return DmaBuffer(self, allocation, fd.value)

    def flush(self, *, all_devices=False):
        scope = CUDA_FLUSH_ALL_DEVICES if all_devices else CUDA_FLUSH_OWNER
        self.call("cuFlushGPUDirectRDMAWrites", CUDA_FLUSH_CURRENT_CONTEXT, scope)

    def ensure_sync_memops(self, allocation):
        """Check RDMA/CUDA consistency on the coordinator's imported alias."""
        if allocation.runtime.kind != "cuda" or allocation.host:
            raise ValueError("RDMA synchronization requires CUDA device memory")
        value = c.c_uint()
        self.call("cuPointerGetAttribute", c.byref(value), CUDA_POINTER_SYNC_MEMOPS, allocation.pointer)
        before = value.value
        if before not in (0, 1):
            raise ValueError("invalid CUDA SYNC_MEMOPS attribute")
        if not before:
            value.value = 1
            self.call("cuPointerSetAttribute", c.byref(value), CUDA_POINTER_SYNC_MEMOPS, allocation.pointer)
            value.value = 0
            self.call("cuPointerGetAttribute", c.byref(value), CUDA_POINTER_SYNC_MEMOPS, allocation.pointer)
            if value.value != 1:
                raise ValueError("CUDA alias did not enable SYNC_MEMOPS")
        return bool(before)


class DmaBuffer:
    def __init__(self, driver, allocation, fd):
        self.driver, self.allocation, self.fd = driver, allocation, fd
        allocation.leases += 1

    def close(self):
        self.allocation.runtime.check_owner()
        if self.fd is not None:
            if self.allocation.leases != 1:
                raise RuntimeError("DMA-BUF allocation is held by an active or quarantined transfer")
            os.close(self.fd)
            self.fd = None
            self.allocation.leases -= 1


class QuarantinedDMA(RuntimeError):
    """The driver may still access retained allocations; keep the owner alive."""


class KernelPeer:
    def __init__(self, physical_chip, path="/dev/sidecard_peer", *, ioctl=fcntl.ioctl, persistent_gpu_mappings=False):
        if isinstance(physical_chip, bool) or physical_chip not in (2, 3):
            raise ValueError("direct qualification is restricted to physical chips 2,3")
        if type(persistent_gpu_mappings) is not bool:
            raise ValueError("persistent GPU mapping mode must be boolean")
        self.physical_chip, self.pid, self.thread = physical_chip, os.getpid(), threading.get_ident()
        self.fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
        self.ioctl, self.failed, self.retained = ioctl, False, None
        self.persistent_gpu_mappings = persistent_gpu_mappings
        self.exports = []

    def check_owner(self):
        if os.getpid() != self.pid or threading.get_ident() != self.thread:
            raise RuntimeError("direct peer must stay in its creating process and thread")
        if self.fd is None or self.failed:
            raise RuntimeError("direct peer is closed or quarantined")

    def copy(self, operation, gpu, npu, guard, size, *, source_ready):
        self.check_owner()
        if operation not in (CUDA_TO_ASCEND, ASCEND_TO_CUDA):
            raise ValueError("invalid direct transfer direction")
        if gpu.fd is None or npu.runtime.kind != "ascend" or npu.host:
            raise ValueError("invalid GPU/HBM transfer endpoints")
        if guard.runtime is not npu.runtime or not guard.host:
            raise ValueError("HBM lifetime guard must be ACL host memory in the same runtime")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 < size <= min(gpu.allocation.size, npu.size, MAX_BYTES)
        ):
            raise ValueError("direct copy exceeds its allocation capacity")
        if npu.size > MAX_BYTES or guard.size < npu.size:
            raise ValueError("invalid HBM pin extent or guard capacity")
        new_export = self.persistent_gpu_mappings and not any(exported is gpu for exported in self.exports)
        if new_export and len(self.exports) >= MAX_GPU_MAPPINGS:
            raise ValueError("persistent peer exceeds its GPU mapping capacity")
        ready_runtime = gpu.allocation.runtime if operation == CUDA_TO_ASCEND else npu.runtime
        if source_ready.runtime is not ready_runtime:
            raise ValueError("source readiness event belongs to another runtime")
        source_ready.require_recorded()
        source_ready.wait()
        request = Request(
            version=VERSION,
            physical_chip=self.physical_chip,
            dmabuf_fd=gpu.fd,
            operation=operation,
            npu_va=npu.pointer,
            length=size,
            pin_length=npu.size,
            guard_va=guard.pointer,
            gpu_mapping_mode=int(self.persistent_gpu_mappings),
        )
        if new_export:
            # Native static mappings outlive individual copies. Hold the owner
            # even if setup fails, until a clean device-fd close releases them.
            gpu.allocation.leases += 1
            self.exports.append(gpu)
        resources = (gpu.allocation, npu, guard, source_ready)
        for resource in resources:
            resource.leases += 1
        completed = False
        try:
            body = bytearray(bytes(request))
            self.ioctl(self.fd, EXECUTE, body, True)
            result = Request.from_buffer_copy(body)
            identity = (
                "version",
                "physical_chip",
                "dmabuf_fd",
                "operation",
                "npu_va",
                "length",
                "pin_length",
                "guard_va",
                "gpu_mapping_mode",
            )
            if any(getattr(result, key) != getattr(request, key) for key in identity):
                raise QuarantinedDMA("kernel completion identity differs from the submitted transfer")
            if result.retained:
                raise QuarantinedDMA(f"native DMA status {result.dma_status}; kernel retains mappings")
            if not (result.map_status or result.pin_status or result.dma_status) and (
                result.cache_id != self.physical_chip
                or not result.gpu_segments
                or result.gpu_mapped_bytes < size
                or result.revoked
                or result.reserved
                or result.gpu_mapping_reused not in (0, 1)
                or (result.gpu_mapping_reused and not self.persistent_gpu_mappings)
            ):
                raise QuarantinedDMA("invalid successful kernel completion geometry")
            completed = True  # Kernel has either rejected setup or completed DMA.
            if result.map_status or result.pin_status or result.dma_status:
                raise RuntimeError(
                    f"direct DMA rejected: map={result.map_status} pin={result.pin_status} dma={result.dma_status}"
                )
            if operation == ASCEND_TO_CUDA:
                gpu.driver.flush()
            return {
                key: getattr(result, key)
                for key, _ in Request._fields_
                if key not in ("npu_va", "dmabuf_fd", "guard_va")
            }
        finally:
            if completed:
                for resource in resources:
                    resource.leases -= 1
            else:
                # Includes an ioctl error: a response-copy failure could happen
                # after native submission. Preserve allocation lifetimes.
                self.failed, self.retained = True, resources

    def close(self):
        if self.failed:
            raise QuarantinedDMA("keep the worker and its native allocations alive; DMA quiescence is unproved")
        if self.fd is not None:
            self.check_owner()
            os.close(self.fd)
            self.fd = None
            for exported in self.exports:
                exported.allocation.leases -= 1
            self.exports.clear()
