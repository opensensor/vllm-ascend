# SPDX-License-Identifier: Apache-2.0
"""Private CANN IPC/Notify backend for pair_collective; device-unqualified.

Construct only inside the assigned, initialized SDK owner. This module never
initializes/resets a device, modifies its environment, or imports Torch. Native
enqueue owns a dedicated stream: direct calls must not bypass a Torch task queue.
"""

import ctypes as c
import hashlib
import os
import threading
from pathlib import Path

from .pair_add import LocalAdd, LocalAddSpec
from .pair_collective import PairConfig

IPC_KEY_BYTES = 65
P2P_ALLOCATION_POLICY = 3
MEMORY_IMPORT_PEER_ACCESS = 1
NOTIFY_IMPORT_PEER_ACCESS = 2
WORKSPACE_BYTES = 1024 * 1024
STREAM_DRAIN_TIMEOUT_MS = 30000
PAIR_ABI = 2


class Bindings(c.Structure):
    _fields_ = [
        ("abi", c.c_uint64),
        ("elements", c.c_uint64),
        *(
            (name, c.c_void_p)
            for name in (
                "input",
                "inbox",
                "output",
                "peer_data",
                "ready",
                "consumed",
                "peer_ready",
                "peer_consumed",
                "workspace",
            )
        ),
        ("workspace_bytes", c.c_uint64),
        ("exchange_mode", c.c_uint64),
    ]


def _key(value):
    if type(value) is not str or not 0 < len(value) < IPC_KEY_BYTES or "\0" in value:
        raise ValueError("invalid private IPC key")
    return value.encode("ascii")


def _bind(library, name, arguments):
    function = getattr(library, name)
    function.argtypes, function.restype = arguments, c.c_int


class AclPairBackend:
    """Own one persistent inbox, source/output buffers, and two notifications.

    Paths/SHA256 authenticate loaded binaries. Caller establishes and checks the
    assigned context before construction; resource creation is explicit prepare.
    Partially created resources remain in this object on failure. No __del__ or
    automatic cleanup can invalidate mappings whose peer ownership is uncertain.
    """

    def __init__(self, acl_library, native_library, *, acl_sha256, native_sha256, local_add=None):
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.allocations, self.notifications, self.imports = [], {}, {}
        self.export_key, self.stream, self.handle = None, c.c_void_p(), c.c_void_p()
        self.extra_export_key = None
        self.config = None
        if local_add is not None and not isinstance(local_add, LocalAddSpec):
            raise ValueError("local_add requires an authenticated LocalAddSpec")
        self.local_add_spec, self.local_add = local_add, None
        for path, digest in ((acl_library, acl_sha256), (native_library, native_sha256)):
            path = Path(path)
            if not path.is_absolute() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("private collective library path or SHA256 differs")
        self.acl = c.CDLL(str(acl_library), mode=c.RTLD_GLOBAL)
        self.native = c.CDLL(str(native_library))
        for name, expected in (("sidecard_pair_abi", PAIR_ABI), ("sidecard_pair_bindings_bytes", c.sizeof(Bindings))):
            function = getattr(self.native, name)
            function.argtypes, function.restype = [], c.c_uint64
            if function() != expected:
                raise ValueError("private collective ABI or bindings size differs")
        pointer, output, size = c.c_void_p, c.POINTER(c.c_void_p), c.c_size_t
        signatures = (
            ("aclrtDeviceGetBareTgid", [c.POINTER(c.c_int32)]),
            ("aclrtGetCurrentContext", [output]),
            ("aclrtMalloc", [output, size, c.c_int]),
            ("aclrtFree", [pointer]),
            ("aclrtCreateStream", [output]),
            ("aclrtDestroyStream", [pointer]),
            ("aclrtSynchronizeStreamWithTimeout", [pointer, c.c_int32]),
            ("aclrtIpcMemGetExportKey", [pointer, size, pointer, size, c.c_uint64]),
            ("aclrtIpcMemSetImportPid", [c.c_char_p, c.POINTER(c.c_int32), size]),
            ("aclrtIpcMemImportByKey", [output, c.c_char_p, c.c_uint64]),
            ("aclrtIpcMemClose", [c.c_char_p]),
            ("aclrtCreateNotify", [output, c.c_uint64]),
            ("aclrtDestroyNotify", [pointer]),
            ("aclrtNotifyGetExportKey", [pointer, pointer, size, c.c_uint64]),
            ("aclrtNotifySetImportPid", [pointer, c.POINTER(c.c_int32), size]),
            ("aclrtNotifyImportByKey", [output, c.c_char_p, c.c_uint64]),
        )
        for name, arguments in signatures:
            _bind(self.acl, name, arguments)
        _bind(self.native, "sidecard_pair_create", [c.POINTER(Bindings), output])
        _bind(self.native, "sidecard_pair_enqueue", [pointer, pointer, c.c_uint32])
        _bind(self.native, "sidecard_pair_destroy", [pointer])
        self.context = c.c_void_p()
        status = self.acl.aclrtGetCurrentContext(c.byref(self.context))
        if status or not self.context.value:
            raise RuntimeError("an initialized SDK owner context is required")

    def _call(self, name, *arguments):
        if self.pid != os.getpid() or self.thread != threading.get_ident():
            raise RuntimeError("SDK backend belongs to another process or thread")
        context = c.c_void_p()
        if self.acl.aclrtGetCurrentContext(c.byref(context)) or context.value != self.context.value:
            raise RuntimeError("SDK owner context changed")
        library = self.native if name.startswith("sidecard_") else self.acl
        status = getattr(library, name)(*arguments)
        if status:
            raise RuntimeError(f"{name} failed: native status {status}")

    def bare_pid(self):
        pid = c.c_int32()
        self._call("aclrtDeviceGetBareTgid", c.byref(pid))
        if pid.value <= 0:
            raise RuntimeError("SDK returned an invalid bare-device PID")
        return pid.value

    def _allocate(self, size):
        pointer = c.c_void_p()
        self._call("aclrtMalloc", c.byref(pointer), size, P2P_ALLOCATION_POLICY)
        if pointer.value is None:
            raise RuntimeError("SDK returned a null allocation")
        self.allocations.append(pointer)
        return pointer

    def prepare(self, config, peer_pid):
        if self.config is not None or not isinstance(config, PairConfig):
            raise ValueError("backend preparation is single-use and requires PairConfig")
        if config.exchange_mode != "push":
            raise ValueError("direct remote vector reads are disabled after the 310P device timeout")
        expected_reducer = self.local_add_spec.identity if self.local_add_spec else "aclnn"
        if config.reducer_identity != expected_reducer:
            raise ValueError("local reducer differs from the peer protocol configuration")
        if self.local_add_spec is not None:
            self.local_add_spec.authenticate()
            if config.rows not in (1, 4, 32):
                raise ValueError("local add requires fixed rows 1, 4 or 32")
        if config.handshake == "ping-pong":
            if self.local_add_spec is not None:
                raise ValueError("ping-pong currently requires the ACLNN local reducer")
            # Verify optional binary capability before allocating resources.
            _bind(self.native, "sidecard_pair_enable_ping_pong", [c.c_void_p, c.c_void_p, c.c_void_p])
        if type(peer_pid) is not int or not 0 < peer_pid < 2**31:
            raise ValueError("invalid bare-device peer PID")
        self.config = config
        self._call("aclrtCreateStream", c.byref(self.stream))
        self.input, self.inbox, self.output = (self._allocate(config.nbytes) for _ in range(3))
        self.workspace = self._allocate(WORKSPACE_BYTES)
        pid = c.c_int32(peer_pid)
        key = c.create_string_buffer(IPC_KEY_BYTES)
        publication = self.input if config.exchange_mode == "remote-read" else self.inbox
        self._call("aclrtIpcMemGetExportKey", publication, config.nbytes, key, IPC_KEY_BYTES, 0)
        self.export_key = key.value.decode("ascii")
        self._call("aclrtIpcMemSetImportPid", _key(self.export_key), c.byref(pid), 1)
        keys = {"data": self.export_key}
        if config.handshake == "ping-pong":
            self.inbox_next = self._allocate(config.nbytes)
            self._call("aclrtIpcMemGetExportKey", self.inbox_next, config.nbytes, key, IPC_KEY_BYTES, 0)
            self.extra_export_key = key.value.decode("ascii")
            self._call("aclrtIpcMemSetImportPid", _key(self.extra_export_key), c.byref(pid), 1)
            keys["data_next"] = self.extra_export_key
        for name in ("ready", "consumed"):
            notify = c.c_void_p()
            self._call("aclrtCreateNotify", c.byref(notify), 0)
            self.notifications[name] = notify
            self._call("aclrtNotifyGetExportKey", notify, key, IPC_KEY_BYTES, 0)
            keys[name] = key.value.decode("ascii")
            self._call("aclrtNotifySetImportPid", notify, c.byref(pid), 1)
        return keys

    def connect(self, keys):
        expected = {"data", "ready", "consumed"}
        if self.config is not None and self.config.handshake == "ping-pong":
            expected.add("data_next")
        if self.config is None or self.imports or type(keys) is not dict or set(keys) != expected:
            raise ValueError("invalid or duplicate peer mapping")
        for key in keys.values():
            _key(key)
        # Memory and Notify keys live in separate SDK namespaces, even when
        # their exported strings coincide in a fresh device context.
        if keys["ready"] == keys["consumed"] or ("data_next" in keys and keys["data"] == keys["data_next"]):
            raise ValueError("keys must be distinct within each resource namespace")
        for name in ("data", "ready", "consumed", *(("data_next",) if "data_next" in expected else ())):
            pointer = c.c_void_p()
            memory = name in ("data", "data_next")
            function = "aclrtIpcMemImportByKey" if memory else "aclrtNotifyImportByKey"
            # Sibling 310P Duo chips already communicate over HCCS. Asking the
            # SDK to enable PCIe P2P on that shared PCIe device fails in the
            # driver before import (507899 / halDeviceEnableP2P drv status17).
            flag = 0
            if self.config.link_mode == "pcie":
                flag = MEMORY_IMPORT_PEER_ACCESS if memory else NOTIFY_IMPORT_PEER_ACCESS
            self._call(function, c.byref(pointer), _key(keys[name]), flag)
            self.imports[name] = (pointer, keys[name])
        b = Bindings(
            PAIR_ABI,
            self.config.elements,
            self.input,
            self.inbox,
            self.output,
            self.imports["data"][0],
            self.notifications["ready"],
            self.notifications["consumed"],
            self.imports["ready"][0],
            self.imports["consumed"][0],
            self.workspace,
            WORKSPACE_BYTES,
            int(self.config.exchange_mode == "remote-read"),
        )
        self._call("sidecard_pair_create", c.byref(b), c.byref(self.handle))
        if self.config.handshake == "ping-pong":
            self._call("sidecard_pair_enable_ping_pong", self.handle, self.inbox_next, self.imports["data_next"][0])
        if self.local_add_spec is not None:
            self.local_add = LocalAdd(self.local_add_spec, self)
            self.local_add.attach()

    def enqueue(self, timeout_raw):
        self._call("sidecard_pair_enqueue", self.handle, self.stream, timeout_raw)

    def synchronize(self):
        self._call("aclrtSynchronizeStreamWithTimeout", self.stream, STREAM_DRAIN_TIMEOUT_MS)

    def release_imports(self):
        for name in ("ready", "consumed", "data", *(("data_next",) if "data_next" in self.imports else ())):
            pointer, key = self.imports[name]
            if name in ("data", "data_next"):
                self._call("aclrtIpcMemClose", _key(key))
            else:
                self._call("aclrtDestroyNotify", pointer)
            del self.imports[name]

    def close(self):
        if self.imports:
            raise RuntimeError("imported mappings remain live")
        self._call("sidecard_pair_destroy", self.handle)
        if self.local_add is not None:
            self.local_add.close()
        for notify in self.notifications.values():
            self._call("aclrtDestroyNotify", notify)
        self._call("aclrtIpcMemClose", _key(self.export_key))
        if self.extra_export_key is not None:
            self._call("aclrtIpcMemClose", _key(self.extra_export_key))
        for allocation in reversed(self.allocations):
            self._call("aclrtFree", allocation)
        self._call("aclrtDestroyStream", self.stream)
