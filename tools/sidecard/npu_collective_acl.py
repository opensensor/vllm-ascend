# SPDX-License-Identifier: Apache-2.0
"""Owner-local SDK resources for an arbitrary-rank binomial SUM.

No device initialization, Torch import, remote vector read, global setting or
automatic cleanup occurs here. A failed operation retains all attached owners.
The native aggregate plan submits every tree step on one caller-owned stream.
"""

import ctypes as c
import hashlib
import os
import threading
from pathlib import Path

from .npu_collective_schedule import BinomialSchedule
from .pair_collective_acl import IPC_KEY_BYTES, STREAM_DRAIN_TIMEOUT_MS, WORKSPACE_BYTES, _bind, _key


class ExchangeBindings(c.Structure):
    _fields_ = (
        [(n, c.c_uint64) for n in ("abi", "elements", "dtype")]
        + [
            (n, c.c_void_p)
            for n in (
                "input",
                "output",
                "inbox0",
                "inbox1",
                "peer_inbox0",
                "peer_inbox1",
                "ready0",
                "ready1",
                "peer_ready0",
                "peer_ready1",
                "workspace",
            )
        ]
        + [("workspace_bytes", c.c_uint64)]
    )


class GroupStep(c.Structure):
    _fields_ = [("exchange", c.c_void_p), ("operation", c.c_uint32), ("reserved", c.c_uint32)]


def authenticated_library(path, checksum, *, global_symbols=False):
    path = Path(path)
    if not path.is_absolute() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
        raise ValueError("collective library path or SHA256 differs")
    return c.CDLL(str(path), mode=c.RTLD_GLOBAL if global_symbols else c.RTLD_LOCAL)


class NpuCollectiveOwner:
    """Retain all edges of one fixed geometry until every rank has drained.

    An SDK error quarantines the entire group; another collective cannot be
    selected after any part of this group's work was submitted. Input producers
    and output consumers must use the same stream as enqueue. Torch integration
    must call the aggregate native plan THROUGH the NPU task queue.
    """

    def __init__(
        self,
        schedule,
        rank,
        session,
        links,
        acl_library,
        native_library,
        *,
        acl_sha256,
        native_sha256,
        device_index,
        stream=None,
    ):
        if not isinstance(schedule, BinomialSchedule) or type(rank) is not int or rank not in schedule.members:
            raise ValueError("an immutable schedule and exact member rank are required")
        if type(session) is not str or len(session) != 64 or any(v not in "0123456789abcdef" for v in session):
            raise ValueError("a fresh shared 256-bit session nonce is required")
        if type(device_index) is not int or device_index < 0 or schedule.elements > 16 * 1024 * 1024:
            raise ValueError("invalid device or native tensor geometry")
        if stream is not None and (type(stream) is not int or stream <= 0):
            raise ValueError("borrowed stream must be a nonzero native address")
        self.steps = schedule.steps_for_rank(rank)
        self.peers = tuple(sorted({s.peer for s in self.steps}))
        if (
            type(links) is not dict
            or any(type(peer) is not int for peer in links)
            or set(links) != set(self.peers)
            or any(v not in ("hccs", "pcie") for v in links.values())
        ):
            raise ValueError("declare a supported link mode for every local tree peer")
        self.schedule, self.rank, self.session, self.links = schedule, rank, session, dict(links)
        self.device_index = device_index
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.state, self.calls = "new", 0
        self.allocations, self.edges = [], {}
        self.plan, self.stream = c.c_void_p(), c.c_void_p(stream)
        self.graph_leases, self._graph_leases_bound = [], False
        self.borrowed_stream = stream is not None
        self.library_inputs = ((str(acl_library), acl_sha256), (str(native_library), native_sha256))
        self.acl = authenticated_library(acl_library, acl_sha256, global_symbols=True)
        self.native = authenticated_library(native_library, native_sha256)
        for name, expected in (
            ("sidecard_exchange_abi", 1),
            ("sidecard_exchange_bindings_bytes", c.sizeof(ExchangeBindings)),
            ("sidecard_group_abi", 1),
        ):
            function = getattr(self.native, name)
            function.argtypes, function.restype = [], c.c_uint64
            if function() != expected:
                raise ValueError("collective native ABI differs")
        pointer, output, size = c.c_void_p, c.POINTER(c.c_void_p), c.c_size_t
        for name, args in (
            ("aclrtGetCurrentContext", [output]),
            ("aclrtDeviceGetBareTgid", [c.POINTER(c.c_int32)]),
            ("aclrtCreateStream", [output]),
            ("aclrtDestroyStream", [pointer]),
            ("aclrtMalloc", [output, size, c.c_int]),
            ("aclrtFree", [pointer]),
            ("aclrtSynchronizeStreamWithTimeout", [pointer, c.c_int32]),
            ("aclrtCreateNotify", [output, c.c_uint64]),
            ("aclrtDestroyNotify", [pointer]),
            ("aclrtIpcMemGetExportKey", [pointer, size, pointer, size, c.c_uint64]),
            ("aclrtIpcMemSetImportPid", [c.c_char_p, c.POINTER(c.c_int32), size]),
            ("aclrtIpcMemImportByKey", [output, c.c_char_p, c.c_uint64]),
            ("aclrtIpcMemClose", [c.c_char_p]),
            ("aclrtNotifyGetExportKey", [pointer, pointer, size, c.c_uint64]),
            ("aclrtNotifySetImportPid", [pointer, c.POINTER(c.c_int32), size]),
            ("aclrtNotifyImportByKey", [output, c.c_char_p, c.c_uint64]),
        ):
            _bind(self.acl, name, args)
        for name, args in (
            ("sidecard_exchange_create", [c.POINTER(ExchangeBindings), output]),
            ("sidecard_exchange_destroy", [output, c.c_uint32]),
            (
                "sidecard_group_create",
                [pointer, pointer, c.c_uint64, c.c_uint32, c.c_int32, c.POINTER(GroupStep), c.c_uint64, output],
            ),
            ("sidecard_group_enqueue", [pointer, pointer, pointer, pointer, c.c_uint32]),
            ("sidecard_group_destroy", [output, c.c_uint32]),
        ):
            _bind(self.native, name, args)
        # Older eager DSOs need not export this opt-in handshake ABI. Fixed
        # graph layouts require it before any exchange or group submission.
        if schedule.algorithm == "recursive-doubling-fixed":
            _bind(self.native, "sidecard_exchange_set_fixed_handshake", [pointer])
        self.context = c.c_void_p()
        if self.acl.aclrtGetCurrentContext(c.byref(self.context)) or not self.context.value:
            raise RuntimeError("collective owner requires an initialized SDK context")

    @property
    def nbytes(self):
        return self.schedule.elements * (2 if self.schedule.dtype == "float16" else 4)

    def _require(self, *states):
        if self.pid != os.getpid() or self.thread != threading.get_ident():
            raise RuntimeError("collective belongs to another process or thread")
        if self.state not in states:
            raise RuntimeError("invalid or quarantined collective state: " + self.state)

    def _call(self, name, *args):
        self._require("new", "preparing", "prepared", "connecting", "active", "drained", "imports-closed", "closing")
        context = c.c_void_p()
        if self.acl.aclrtGetCurrentContext(c.byref(context)) or context.value != self.context.value:
            self.state = "quarantined"
            raise RuntimeError("collective SDK context changed; retain owners")
        library = self.native if name.startswith("sidecard_") else self.acl
        status = getattr(library, name)(*args)
        if status:
            self.state = "quarantined"
            raise RuntimeError(f"{name} failed: {status}; retain all collective owners")

    def bare_pid(self):
        pid = c.c_int32()
        self._call("aclrtDeviceGetBareTgid", c.byref(pid))
        if pid.value <= 0:
            raise RuntimeError("invalid SDK bare-device PID")
        return pid.value

    def _allocate(self, size):
        pointer = c.c_void_p()
        self.allocations.append(pointer)
        self._call("aclrtMalloc", c.byref(pointer), size, 3)
        if not pointer.value:
            self.state = "quarantined"
            raise RuntimeError("SDK returned a null allocation; retain owner")
        return pointer

    def prepare(self, peer_pids):
        self._require("new")
        if (
            type(peer_pids) is not dict
            or any(type(peer) is not int for peer in peer_pids)
            or set(peer_pids) != set(self.peers)
            or any(type(pid) is not int or not 0 < pid < 2**31 for pid in peer_pids.values())
        ):
            raise ValueError("declare every peer bare-device PID before allocation")
        for path, checksum in self.library_inputs:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != checksum:
                raise ValueError("collective library SHA256 changed before preparation")
        self.state = "preparing"
        if not self.borrowed_stream:
            self._call("aclrtCreateStream", c.byref(self.stream))
        if not self.stream.value:
            self.state = "quarantined"
            raise RuntimeError("null owner stream")
        self.work_input, self.work_output = (self._allocate(self.nbytes) for _ in range(2))
        self.workspace = self._allocate(WORKSPACE_BYTES)
        exports = {}
        for peer in self.peers:
            edge = dict(local={}, exported={}, imported={}, key_buffers={}, handle=c.c_void_p())
            self.edges[peer] = edge
            pid = c.c_int32(peer_pids[peer])
            for name in ("inbox0", "inbox1", "ready0", "ready1"):
                key = c.create_string_buffer(IPC_KEY_BYTES)
                edge["key_buffers"][name] = key
                if name.startswith("inbox"):
                    pointer = self._allocate(self.nbytes)
                    edge["local"][name] = pointer
                    self._call("aclrtIpcMemGetExportKey", pointer, self.nbytes, key, IPC_KEY_BYTES, 0)
                    edge["exported"][name] = key.value.decode("ascii")
                    self._call("aclrtIpcMemSetImportPid", _key(edge["exported"][name]), c.byref(pid), 1)
                else:
                    pointer = c.c_void_p()
                    edge["local"][name] = pointer
                    self._call("aclrtCreateNotify", c.byref(pointer), 0)
                    self._call("aclrtNotifyGetExportKey", pointer, key, IPC_KEY_BYTES, 0)
                    edge["exported"][name] = key.value.decode("ascii")
                    self._call("aclrtNotifySetImportPid", pointer, c.byref(pid), 1)
            exports[peer] = dict(
                session=self.session,
                schedule_sha256=self.schedule.identity,
                rank=self.rank,
                peer=peer,
                keys=dict(edge["exported"]),
            )
        self.state = "prepared"
        return exports

    def connect(self, peer_exports):
        self._require("prepared")
        if (
            type(peer_exports) is not dict
            or any(type(peer) is not int for peer in peer_exports)
            or set(peer_exports) != set(self.peers)
        ):
            raise ValueError("every tree peer must publish exactly one export")
        for peer, packet in peer_exports.items():
            expected = dict(session=self.session, schedule_sha256=self.schedule.identity, rank=peer, peer=self.rank)
            if (
                type(packet) is not dict
                or set(packet) != set(expected) | {"keys"}
                or any(type(packet[k]) is not type(v) or packet[k] != v for k, v in expected.items())
            ):
                raise ValueError("peer session, group identity or rank differs")
            keys = packet["keys"]
            if type(keys) is not dict or set(keys) != {"inbox0", "inbox1", "ready0", "ready1"}:
                raise ValueError("invalid tree peer keys")
            for value in keys.values():
                _key(value)
            # Memory and Notify exports have separate SDK namespaces. Fresh
            # contexts can return the same key text for one of each; imports
            # select the namespace by API. Only aliasing within a namespace
            # would collapse the two ping-pong slots.
            for first, second in (("inbox0", "inbox1"), ("ready0", "ready1")):
                if keys[first] == keys[second]:
                    raise ValueError("tree peer keys must be distinct within each resource namespace")
        self.state = "connecting"
        for peer in self.peers:
            edge = self.edges[peer]
            for name, key in peer_exports[peer]["keys"].items():
                pointer = c.c_void_p()
                memory = name.startswith("inbox")
                flag = (1 if memory else 2) if self.links[peer] == "pcie" else 0
                edge["imported"][name] = (pointer, key)
                self._call(
                    "aclrtIpcMemImportByKey" if memory else "aclrtNotifyImportByKey", c.byref(pointer), _key(key), flag
                )
            local, imported = edge["local"], edge["imported"]
            bindings = ExchangeBindings(
                1,
                self.schedule.elements,
                int(self.schedule.dtype == "float16"),
                self.work_input,
                self.work_output,
                local["inbox0"],
                local["inbox1"],
                imported["inbox0"][0],
                imported["inbox1"][0],
                local["ready0"],
                local["ready1"],
                imported["ready0"][0],
                imported["ready1"][0],
                self.workspace,
                WORKSPACE_BYTES,
            )
            self._call("sidecard_exchange_create", c.byref(bindings), c.byref(edge["handle"]))
            if self.schedule.algorithm == "recursive-doubling-fixed":
                self._call("sidecard_exchange_set_fixed_handshake", edge["handle"])
        steps = (GroupStep * len(self.steps))(
            *(GroupStep(self.edges[s.peer]["handle"], s.opcode, 0) for s in self.steps)
        )
        self._call(
            "sidecard_group_create",
            self.work_input,
            self.work_output,
            self.nbytes,
            int(self.schedule.dtype == "float16"),
            self.device_index,
            steps,
            len(steps),
            c.byref(self.plan),
        )
        self.state = "active"

    def enqueue(self, input_pointer, output_pointer):
        self._require("active")
        self._call("sidecard_group_enqueue", self.plan, input_pointer, output_pointer, self.stream, 30000)
        self.calls += 1

    def synchronize(self):
        self._require("active")
        self._call("aclrtSynchronizeStreamWithTimeout", self.stream, STREAM_DRAIN_TIMEOUT_MS)

    def _bind_graph_leases(self):
        if self._graph_leases_bound:
            return
        try:
            for name, arguments in (
                ("sidecard_group_graph_lease_acquire", [c.c_void_p, c.POINTER(c.c_void_p)]),
                ("sidecard_group_graph_lease_release", [c.POINTER(c.c_void_p), c.c_uint32]),
                ("sidecard_group_graph_lease_count", [c.c_void_p, c.POINTER(c.c_uint64)]),
            ):
                _bind(self.native, name, arguments)
        except BaseException:
            self.state = "quarantined"
            raise
        self._graph_leases_bound = True

    def _graph_lease_call(self, name, *arguments):
        # These are cold CPU-only lifetime operations. In particular, count
        # must run BEFORE any SDK/context/stream operation during owner drain.
        self._require("active")
        try:
            status = getattr(self.native, name)(*arguments)
            if status:
                raise RuntimeError(f"{name} failed: {status}; retain native graph leases and owner")
        except BaseException:
            self.state = "quarantined"
            raise

    def acquire_graph_lease(self):
        """Retain one native graph ticket before capture; no per-replay work."""
        self._require("active")
        self._bind_graph_leases()
        ticket = c.c_void_p()
        self.graph_leases.append(ticket)
        self._graph_lease_call("sidecard_group_graph_lease_acquire", self.plan, c.byref(ticket))
        if not ticket.value:
            self.state = "quarantined"
            raise RuntimeError("native graph lease returned a null ticket; retain owner")
        return ticket

    def release_graph_lease(self, ticket):
        """Release only an owned ticket after ALL graphs reset and queue drain."""
        self._require("active")
        if (
            type(ticket) is not c.c_void_p
            or not any(owned is ticket for owned in self.graph_leases)
            or not ticket.value
        ):
            raise ValueError("release requires this owner's live native graph lease ticket")
        self._graph_lease_call("sidecard_group_graph_lease_release", c.byref(ticket), 1)
        if ticket.value:
            self.state = "quarantined"
            raise RuntimeError("native graph lease release did not clear its ticket; retain owner")
        self.graph_leases = [owned for owned in self.graph_leases if owned is not ticket]

    def drain(self, *, graphs_retired):
        if graphs_retired is not True:
            raise ValueError("retire graph and queued-handler owners before draining for shutdown")
        self._require("active")
        if self._graph_leases_bound:
            count = c.c_uint64()
            self._graph_lease_call("sidecard_group_graph_lease_count", self.plan, c.byref(count))
            if count.value or self.graph_leases:
                raise ValueError("retire and release every native graph lease before owner drain")
        self.synchronize()
        self.state = "drained"
        return dict(
            rank=self.rank,
            session=self.session,
            schedule_sha256=self.schedule.identity,
            calls=self.calls,
            graphs_retired=True,
        )

    def _receipts(self, receipts, *, imports_closed):
        if type(receipts) is not list or len(receipts) != len(self.schedule.members):
            raise ValueError("every rank must prove the same shutdown phase")
        seen = set()
        for packet in receipts:
            if type(packet) is not dict or type(packet.get("rank")) is not int:
                raise ValueError("invalid rank receipt")
            rank = packet["rank"]
            expected = dict(
                rank=rank,
                session=self.session,
                schedule_sha256=self.schedule.identity,
                calls=self.calls,
                graphs_retired=True,
            )
            if imports_closed:
                expected["imports_closed"] = True
            if (
                rank not in self.schedule.members
                or rank in seen
                or set(packet) != set(expected)
                or any(type(packet[k]) is not type(v) or packet[k] != v for k, v in expected.items())
            ):
                raise ValueError("ranks did not prove the same group/call sequence/shutdown phase")
            seen.add(rank)

    def release_imports(self, receipts):
        self._require("drained")
        self._receipts(receipts, imports_closed=False)
        for edge in self.edges.values():
            for name in tuple(edge["imported"]):
                pointer, key = edge["imported"][name]
                self._call("aclrtIpcMemClose", _key(key)) if name.startswith("inbox") else self._call(
                    "aclrtDestroyNotify", pointer
                )
                del edge["imported"][name]
        self.state = "imports-closed"
        return dict(
            rank=self.rank,
            session=self.session,
            schedule_sha256=self.schedule.identity,
            calls=self.calls,
            graphs_retired=True,
            imports_closed=True,
        )

    def close(self, receipts, *, graphs_retired):
        self._require("imports-closed")
        if graphs_retired is not True:
            raise ValueError("both-peer import closure and graph retirement are required")
        self._receipts(receipts, imports_closed=True)
        self.state = "closing"
        self._call("sidecard_group_destroy", c.byref(self.plan), 1)
        for edge in self.edges.values():
            self._call("sidecard_exchange_destroy", c.byref(edge["handle"]), 1)
            for name in ("ready0", "ready1"):
                self._call("aclrtDestroyNotify", edge["local"][name])
            for name in ("inbox0", "inbox1"):
                self._call("aclrtIpcMemClose", _key(edge["exported"][name]))
        for allocation in reversed(self.allocations):
            self._call("aclrtFree", allocation)
        if not self.borrowed_stream:
            self._call("aclrtDestroyStream", self.stream)
        self.state = "closed"
