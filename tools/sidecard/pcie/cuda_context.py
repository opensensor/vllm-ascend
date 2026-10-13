# SPDX-License-Identifier: Apache-2.0
"""Serialize native CUDA ownership across GLM's ASGI generation threads."""

import ctypes
import os

from .runtime import Runtime


class CudaContextRuntime(Runtime):
    """Require the same process and CUDA context, rather than a fixed thread.

    The caller must hold the existing GLM dispatch lock. General runtime
    objects and independent worker runtimes retain their stricter thread rule.
    """

    def __init__(self, library, driver):
        self.owner_context = None
        self.current_context = driver.library.cuCtxGetCurrent
        self.current_context.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.current_context.restype = ctypes.c_int
        super().__init__("cuda", library)
        self.owner_context = self.context()
        if not self.owner_context:
            raise RuntimeError("CUDA result ownership requires an initialized context")

    def context(self):
        value = ctypes.c_void_p()
        if self.current_context(ctypes.byref(value)):
            raise RuntimeError("cannot authenticate the current CUDA context")
        return value.value

    def check_owner(self):
        if self.owner_context is None:
            return super().check_owner()
        if self.pid != os.getpid() or self.context() != self.owner_context:
            raise RuntimeError("completion results belong to another process or CUDA context")
