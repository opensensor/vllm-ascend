# SPDX-License-Identifier: Apache-2.0
"""Direct CANN profiling lifecycle, without torch profiler allocator hooks."""

import ctypes
from pathlib import Path


class AclProfile:
    # Values from the deployed CANN 9.1.0 include/acl/acl_prof.h.
    ACL_API = 0x0001
    TASK_TIME = 0x0002
    AICORE_METRICS = 0x0004
    HCCL_TRACE = 0x0020
    TRAINING_TRACE = 0x0040
    RUNTIME_API = 0x0100
    OP_ATTR = 0x4000
    PIPE_UTILIZATION = 1

    def __init__(self):
        self.library = ctypes.CDLL("/usr/local/Ascend/cann-9.1.0/lib64/libascendcl.so")
        signatures = {
            "aclprofInit": ([ctypes.c_char_p, ctypes.c_size_t], ctypes.c_int),
            "aclprofCreateConfig": (
                [ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64],
                ctypes.c_void_p,
            ),
            "aclprofStart": ([ctypes.c_void_p], ctypes.c_int),
            "aclprofStop": ([ctypes.c_void_p], ctypes.c_int),
            "aclprofDestroyConfig": ([ctypes.c_void_p], ctypes.c_int),
            "aclprofFinalize": ([], ctypes.c_int),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.library, name)
            function.argtypes = arguments
            function.restype = result
        self.config = None
        self.initialized = False
        self.active = False

    @staticmethod
    def check(result, operation):
        if result != 0:
            raise RuntimeError(f"{operation} returned CANN error {result}")

    def start(self, output, device):
        if self.initialized:
            raise RuntimeError("profiling session already initialized")
        Path(output).mkdir(parents=True, exist_ok=True)
        encoded = str(output).encode()
        self.check(self.library.aclprofInit(encoded, len(encoded)), "aclprofInit")
        self.initialized = True
        devices = (ctypes.c_uint32 * 1)(device)
        flags = (
            self.ACL_API
            | self.TASK_TIME
            | self.AICORE_METRICS
            | self.HCCL_TRACE
            | self.TRAINING_TRACE
            | self.RUNTIME_API
            | self.OP_ATTR
        )
        try:
            self.config = self.library.aclprofCreateConfig(devices, 1, self.PIPE_UTILIZATION, None, flags)
            if not self.config:
                raise RuntimeError("aclprofCreateConfig returned null")
            self.check(self.library.aclprofStart(self.config), "aclprofStart")
            self.active = True
        except Exception:
            self.stop()
            raise

    def stop(self):
        # Stop collection before releasing the configuration or recapturing graphs.
        if self.active:
            self.check(self.library.aclprofStop(self.config), "aclprofStop")
            self.active = False
        if self.config:
            self.check(self.library.aclprofDestroyConfig(self.config), "aclprofDestroyConfig")
            self.config = None
        if self.initialized:
            self.check(self.library.aclprofFinalize(), "aclprofFinalize")
            self.initialized = False
