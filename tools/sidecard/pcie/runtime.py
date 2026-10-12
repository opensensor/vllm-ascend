# SPDX-License-Identifier: Apache-2.0
"""Small native SDK adapter; no PyTorch imports or cross-vendor peer claims."""

import ctypes as c
import os
import threading

IPC_HANDLE_BYTES = 64
CUDA_EVENT_DISABLE_TIMING = 2
CUDA_EVENT_INTERPROCESS = 4
CUDA_NOT_READY = 600
CUDA_STREAM_NON_BLOCKING = 1


class IpcHandle(c.Structure):
    _fields_ = [("reserved", c.c_char * IPC_HANDLE_BYTES)]


class Runtime:
    def __init__(self, kind, library, device=0, *, initialize=True):
        if kind not in ("cuda", "ascend"):
            raise ValueError("backend must be cuda or ascend")
        self.kind, self.pid, self.thread = kind, os.getpid(), threading.get_ident()
        self.device = device
        self.library = c.CDLL(str(library))
        pointer = c.c_void_p
        output = c.POINTER(pointer)
        size = c.c_size_t
        signatures = (
            {
                "cudaSetDevice": [c.c_int],
                "cudaMalloc": [output, size],
                "cudaFree": [pointer],
                "cudaMallocHost": [output, size],
                "cudaFreeHost": [pointer],
                "cudaStreamCreateWithFlags": [output, c.c_uint],
                "cudaStreamDestroy": [pointer],
                "cudaEventCreateWithFlags": [output, c.c_uint],
                "cudaEventDestroy": [pointer],
                "cudaEventRecord": [pointer, pointer],
                "cudaEventQuery": [pointer],
                "cudaEventSynchronize": [pointer],
                "cudaStreamWaitEvent": [pointer, pointer, c.c_uint],
                "cudaMemcpyAsync": [pointer, pointer, size, c.c_int, pointer],
                "cudaIpcGetMemHandle": [c.POINTER(IpcHandle), pointer],
                "cudaIpcOpenMemHandle": [output, IpcHandle, c.c_uint],
                "cudaIpcCloseMemHandle": [pointer],
                "cudaIpcGetEventHandle": [c.POINTER(IpcHandle), pointer],
                "cudaIpcOpenEventHandle": [output, IpcHandle],
            }
            if kind == "cuda"
            else {
                "aclInit": [c.c_char_p],
                "aclrtSetDevice": [c.c_int],
                "aclrtMalloc": [output, size, c.c_int],
                "aclrtFree": [pointer],
                "aclrtMallocHost": [output, size],
                "aclrtFreeHost": [pointer],
                "aclrtCreateStream": [output],
                "aclrtDestroyStream": [pointer],
                "aclrtCreateEvent": [output],
                "aclrtDestroyEvent": [pointer],
                "aclrtRecordEvent": [pointer, pointer],
                "aclrtQueryEventStatus": [pointer, c.POINTER(c.c_int)],
                "aclrtSynchronizeEvent": [pointer],
                "aclrtStreamWaitEvent": [pointer, pointer],
                "aclrtMemcpyAsync": [pointer, size, pointer, size, c.c_int, pointer],
            }
        )
        for name, arguments in signatures.items():
            function = getattr(self.library, name)
            function.argtypes, function.restype = arguments, c.c_int
        if kind == "ascend":
            if initialize:
                self.call("aclInit", None)
            self.call("aclrtSetDevice", device)
        else:
            self.call("cudaSetDevice", device)

    def check_owner(self):
        if self.pid != os.getpid() or self.thread != threading.get_ident():
            raise RuntimeError("SDK objects must stay in their creating process and thread")

    def call(self, name, *arguments):
        self.check_owner()
        result = getattr(self.library, name)(*arguments)
        if result:
            raise RuntimeError(f"{name} failed with native status {result}")

    def allocate(self, size, *, host=False):
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("allocation size must be a positive integer")
        pointer = c.c_void_p()
        if self.kind == "cuda":
            self.call("cudaMallocHost" if host else "cudaMalloc", c.byref(pointer), size)
        elif host:
            self.call("aclrtMallocHost", c.byref(pointer), size)
        else:
            self.call("aclrtMalloc", c.byref(pointer), size, 0)
        return Allocation(self, pointer.value, size, host=host)

    def stream(self):
        pointer = c.c_void_p()
        if self.kind == "cuda":
            self.call("cudaStreamCreateWithFlags", c.byref(pointer), CUDA_STREAM_NON_BLOCKING)
        else:
            self.call("aclrtCreateStream", c.byref(pointer))
        return Stream(self, pointer.value)

    def event(self, *, ipc=False):
        if ipc and self.kind != "cuda":
            raise ValueError("this backend only exports CUDA IPC events")
        pointer = c.c_void_p()
        if self.kind == "cuda":
            flags = CUDA_EVENT_DISABLE_TIMING | (CUDA_EVENT_INTERPROCESS if ipc else 0)
            self.call("cudaEventCreateWithFlags", c.byref(pointer), flags)
        else:
            self.call("aclrtCreateEvent", c.byref(pointer))
        return Event(self, pointer.value, ipc=ipc)

    def copy(self, destination, source, size, direction, stream):
        if direction not in ("h2d", "d2h", "d2d") or size <= 0 or not destination or not source:
            raise ValueError("invalid native copy")
        stream.require(self)
        kind = {"h2d": 1, "d2h": 2, "d2d": 3}[direction]
        if self.kind == "cuda":
            self.call("cudaMemcpyAsync", destination, source, size, kind, stream.pointer)
        else:
            self.call("aclrtMemcpyAsync", destination, size, source, size, kind, stream.pointer)

    def import_memory(self, descriptor):
        if self.kind != "cuda" or descriptor.get("backend") != "cuda" or descriptor.get("offset") != 0:
            raise ValueError("only dedicated CUDA allocations can be imported")
        size = descriptor.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("invalid IPC allocation capacity")
        handle = decode_handle(descriptor["handle"])
        pointer = c.c_void_p()
        self.call("cudaIpcOpenMemHandle", c.byref(pointer), handle, 1)
        return Allocation(self, pointer.value, size, imported=True)

    def import_event(self, handle):
        if self.kind != "cuda":
            raise ValueError("only CUDA IPC events can be imported")
        pointer = c.c_void_p()
        self.call("cudaIpcOpenEventHandle", c.byref(pointer), decode_handle(handle))
        return Event(self, pointer.value, imported=True)


def decode_handle(encoded):
    raw = bytes.fromhex(encoded)
    if len(raw) != IPC_HANDLE_BYTES:
        raise ValueError("invalid CUDA IPC handle size")
    return IpcHandle.from_buffer_copy(raw)


class Allocation:
    def __init__(self, runtime, pointer, size, *, host=False, imported=False):
        self.runtime, self._pointer, self.size = runtime, pointer, size
        self.host, self.imported = host, imported
        self.leases = 0

    @property
    def pointer(self):
        self.runtime.check_owner()
        if self._pointer is None:
            raise RuntimeError("allocation is closed")
        return self._pointer

    def export(self):
        if self.host or self.imported or self.runtime.kind != "cuda":
            raise ValueError("only owned dedicated CUDA device allocations can be exported")
        handle = IpcHandle()
        self.runtime.call("cudaIpcGetMemHandle", c.byref(handle), self.pointer)
        return dict(backend="cuda", handle=bytes(handle).hex(), size=self.size, offset=0)

    def close(self):
        if self.leases:
            raise RuntimeError("allocation is owned by pending bridge tickets")
        if self._pointer is not None:
            prefix = "cuda" if self.runtime.kind == "cuda" else "aclrt"
            name = "cudaIpcCloseMemHandle" if self.imported else prefix + ("FreeHost" if self.host else "Free")
            self.runtime.call(name, self.pointer)
            self._pointer = None


class Stream:
    def __init__(self, runtime, pointer):
        self.runtime, self.pointer = runtime, pointer

    def require(self, runtime):
        runtime.check_owner()
        if self.runtime is not runtime or self.pointer is None:
            raise ValueError("stream is closed or belongs to another runtime")

    def wait(self, event):
        self.require(event.runtime)
        event.require_recorded()
        if self.runtime.kind == "cuda":
            self.runtime.call("cudaStreamWaitEvent", self.pointer, event.pointer, 0)
        else:
            self.runtime.call("aclrtStreamWaitEvent", self.pointer, event.pointer)

    def close(self):
        if self.pointer is not None:
            self.runtime.call(
                "cudaStreamDestroy" if self.runtime.kind == "cuda" else "aclrtDestroyStream", self.pointer
            )
            self.pointer = None


class Event:
    def __init__(self, runtime, pointer, *, ipc=False, imported=False):
        self.runtime, self.pointer, self.ipc, self.imported = runtime, pointer, ipc, imported
        self.recorded = imported
        self.leases = 0

    def require_recorded(self):
        self.runtime.check_owner()
        if self.pointer is None or not self.recorded:
            raise RuntimeError("event is closed or has not been recorded")

    def record(self, stream):
        stream.require(self.runtime)
        if self.pointer is None or self.imported or self.leases:
            raise RuntimeError("cannot record a closed, imported, or leased event")
        self.runtime.call(
            "cudaEventRecord" if self.runtime.kind == "cuda" else "aclrtRecordEvent", self.pointer, stream.pointer
        )
        self.recorded = True

    def ready(self):
        self.require_recorded()
        if self.runtime.kind == "cuda":
            result = self.runtime.library.cudaEventQuery(self.pointer)
            if result == CUDA_NOT_READY:
                return False
            if result:
                raise RuntimeError(f"cudaEventQuery failed with native status {result}")
            return True
        status = c.c_int()
        self.runtime.call("aclrtQueryEventStatus", self.pointer, c.byref(status))
        if status.value not in (0, 1):
            raise RuntimeError(f"invalid ACL event state {status.value}")
        return status.value == 1

    def wait(self):
        self.require_recorded()
        self.runtime.call(
            "cudaEventSynchronize" if self.runtime.kind == "cuda" else "aclrtSynchronizeEvent", self.pointer
        )

    def export(self):
        if not self.ipc or self.imported:
            raise ValueError("event was not created for IPC export")
        self.require_recorded()
        handle = IpcHandle()
        self.runtime.call("cudaIpcGetEventHandle", c.byref(handle), self.pointer)
        return bytes(handle).hex()

    def close(self):
        if self.leases:
            raise RuntimeError("event is owned by pending bridge tickets")
        if self.pointer is not None:
            self.runtime.call("cudaEventDestroy" if self.runtime.kind == "cuda" else "aclrtDestroyEvent", self.pointer)
            self.pointer = None
