# SPDX-License-Identifier: Apache-2.0
"""Optional serialized, owner-retaining Torch facade for local NPU SUM.

The caller must serialize input producers, this op,
and output consumers on one Torch stream. Construction borrows a cold stream
without queuing SDK work. The first eligible call claims the actual stream;
later calls compare its stream_id and never read the flushing npu_stream
property. No CPU distributed operations occur in all_reduce.

close(graphs_retired=True) acknowledges retirement of graph AND queued-handler
owners. It drains Torch before SDK owners, proves every rank's drain and import
closure, and only then releases exported resources. There is deliberately no
destructor or atexit cleanup. A failed create raises CollectiveSetupError whose
collective_owner attribute retains partially constructed owners for quarantine.
Public receipts contain counters and identities, never IPC keys or pointers.
"""

import contextlib
import hashlib
import json
import os
import secrets
import threading
from pathlib import Path

from .npu_collective_acl import NpuCollectiveOwner
from .npu_collective_schedule import BinomialSchedule

TIMEOUT_MS = 30000
MAX_ELEMENTS = 16 * 1024 * 1024
NPU_DEVICE_TYPES = frozenset({"npu", "privateuseone"})


class CollectiveSetupError(RuntimeError):
    """Retain the facade and every attached owner after partial setup failure."""

    def __init__(self, collective):
        super().__init__("NPU collective setup failed; retain collective owners and inspect the chained error")
        self.collective = collective
        self.collective_owner = collective


def validate_config(config, members, global_rank, device_index):
    """Validate explicit single-host geometry, topology and pinned libraries."""
    fields = {
        "schema_version",
        "acl_library",
        "acl_sha256",
        "native_library",
        "native_sha256",
        "torch_library",
        "torch_sha256",
        "algorithm",
        "geometries",
        "physical_devices",
        "link_groups",
        "output_root",
    }
    if type(config) is not dict or set(config) - {"execution_mode", "graph_replay_tracking"} != fields:
        raise ValueError("exact Torch collective configuration fields are required")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Torch collective schema_version must be integer one")
    # The immutable schedule validates ordered membership and algorithm without
    # importing Torch, initializing SDK contexts or allocating device memory.
    schedule = BinomialSchedule(members, "float16", 1, algorithm=config["algorithm"])
    mode = config.get("execution_mode", "eager")
    if mode not in ("eager", "graph"):
        raise ValueError("execution_mode must be eager or graph")
    tracking = config.get("graph_replay_tracking", False)
    if type(tracking) is not bool or tracking and mode != "graph":
        raise ValueError("graph_replay_tracking requires a boolean and graph execution mode")
    if mode == "graph" and not schedule.graph_safe_per_call:
        raise ValueError("graph mode requires a fixed-address handshake")
    if type(global_rank) is not int or global_rank not in members:
        raise ValueError("global rank must be an exact ordered group member")
    devices = config["physical_devices"]
    if (
        type(devices) is not list
        or len(devices) != len(members)
        or any(type(device) is not int or device < 0 for device in devices)
        or len(set(devices)) != len(devices)
        or type(device_index) is not int
        or devices[members.index(global_rank)] != device_index
    ):
        raise ValueError("physical_devices must map each distinct member to its actual current device")
    groups = config["link_groups"]
    if (
        type(groups) is not list
        or not groups
        or any(type(group) is not list or not group for group in groups)
        or any(type(device) is not int or device < 0 for group in groups for device in group)
    ):
        raise ValueError("link_groups must partition physical device IDs")
    flattened = [device for group in groups for device in group]
    if len(set(flattened)) != len(flattened) or set(flattened) != set(devices):
        raise ValueError("link_groups must partition physical device IDs exactly once")
    geometries = config["geometries"]
    if type(geometries) is not list or not geometries:
        raise ValueError("declare a nonempty fixed geometry list")
    seen = set()
    for geometry in geometries:
        if type(geometry) is not dict or set(geometry) != {"dtype", "elements"}:
            raise ValueError("fixed geometry requires exactly dtype and elements")
        schedule = BinomialSchedule(members, **geometry, algorithm=config["algorithm"])
        key = (schedule.dtype, schedule.elements)
        if key in seen or schedule.elements > MAX_ELEMENTS:
            raise ValueError("duplicate geometry or native element limit exceeded")
        seen.add(key)
    normalized = json.loads(json.dumps(config, allow_nan=False))
    for name in ("acl", "native", "torch"):
        path, checksum = config[name + "_library"], config[name + "_sha256"]
        if type(path) is not str or type(checksum) is not str:
            raise ValueError("explicit library path and SHA256 strings are required")
        path = Path(path)
        if not path.is_absolute() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(name + " library absolute path or SHA256 differs")
        normalized[name + "_library"] = str(path.resolve())
    if Path(normalized["torch_library"]).parent != Path(normalized["native_library"]).parent:
        raise ValueError("Torch addon and its pinned native dependency must share the private RPATH directory")
    root = config["output_root"]
    if type(root) is not str or not Path(root).is_absolute():
        raise ValueError("output_root must be an explicit absolute directory")
    root = Path(root).resolve()
    if root.exists() and not root.is_dir():
        raise ValueError("output_root is not a directory")
    normalized["output_root"] = str(root)
    return normalized


class TorchCollective:
    """One process/thread, one claimed Torch stream, one owner per geometry."""

    def __init__(self, config, members, global_rank, cpu_group, device_index, *, torch, dist, stream_getter):
        self.config = validate_config(config, members, global_rank, device_index)
        self.members, self.rank, self.device_index = members, global_rank, device_index
        self.torch, self.dist, self.cpu_group, self.stream_getter = torch, dist, cpu_group, stream_getter
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.identity = hashlib.sha256(
            json.dumps(dict(config=self.config, members=members), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.owners, self.fallbacks = {}, {}
        self.state, self.claimed_stream = "new", None
        self.initial_stream, self.claimed_stream_id = None, None
        self.session = None
        self.capture_streams = {}
        self.graph_observer = None
        if self.config.get("execution_mode", "eager") == "graph":
            from .npu_collective_graph import CollectiveGraphObserver

            self.graph_observer = CollectiveGraphObserver(
                self, track_replays=self.config.get("graph_replay_tracking", False)
            )

    def _gather(self, phase, payload):
        local = dict(phase=phase, rank=self.rank, config_sha256=self.identity, payload=payload)
        packets = [None] * len(self.members)
        self.dist.all_gather_object(packets, local, group=self.cpu_group)
        for rank, packet in zip(self.members, packets, strict=True):
            expected = dict(phase=phase, rank=rank, config_sha256=self.identity)
            if (
                type(packet) is not dict
                or set(packet) != set(expected) | {"payload"}
                or any(type(packet[k]) is not type(value) or packet[k] != value for k, value in expected.items())
            ):
                raise RuntimeError("all-rank collective phase, membership or configuration identity differs")
        return [packet["payload"] for packet in packets]

    def _context(self, *, allow_capture=False):
        if self.pid != os.getpid() or self.thread != threading.get_ident():
            raise RuntimeError("Torch collective belongs to another process or thread")
        if self.torch.npu.current_device() != self.device_index:
            raise RuntimeError("current NPU device differs from the explicit physical rank mapping")
        capturing = self.torch.npu.is_current_stream_capturing()
        if capturing:
            if not allow_capture or self.graph_observer is None:
                raise RuntimeError("Torch collective capture is forbidden without graph admission")
            self.graph_observer.require_capture()
            if self.claimed_stream_id is None:
                raise RuntimeError("graph capture requires an eligible eager warmup first")
        if self.graph_observer is not None and self.graph_observer.retired and allow_capture:
            raise RuntimeError("collective graphs have been retired")
        return capturing

    def _stream(self, *, capturing=False):
        stream = self.stream_getter()
        if (
            type(stream.stream_id) is not int
            or getattr(stream.device, "type", None) not in NPU_DEVICE_TYPES
            or stream.device.index != self.device_index
        ):
            raise RuntimeError("current Torch stream is not on the assigned physical NPU")
        if capturing:
            # Keep the capture stream alive without reading npu_stream: that
            # property flushes the Torch queue. The native op obtains the actual
            # current stream using stream(false) inside the capture context.
            self.capture_streams[stream.stream_id] = stream
            return stream
        if self.claimed_stream_id is not None and stream.stream_id != self.claimed_stream_id:
            raise RuntimeError("serialized Torch collective stream changed after its first eligible call")
        return stream

    def _receipt(self, phase, **extra):
        root = Path(self.config["output_root"])
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        packet = dict(
            schema_version=1,
            phase=phase,
            rank=self.rank,
            physical_device=self.device_index,
            members=self.members,
            config_sha256=self.identity,
            algorithm=self.config["algorithm"],
            state=self.state,
            geometries=[
                dict(dtype=key[0], elements=key[1], schedule_sha256=owner.schedule.identity, calls=owner.calls)
                for key, owner in self.owners.items()
            ],
            fallback_schemas=[
                dict(dtype=key[0], elements=key[1], contiguous=key[2], calls=count)
                for key, count in self.fallbacks.items()
            ],
            **extra,
        )
        (root / f"{phase}-{self.identity}-rank{self.rank}.json").write_text(json.dumps(packet, indent=2) + "\n")

    def _quarantine(self, error):
        self.state = "quarantined"
        for owner in self.owners.values():
            if owner.state != "closed":
                owner.state = "quarantined"
        # A receipt write failure must not replace the original failure or drop
        # the strong owners needed to retain uncertain native allocations.
        with contextlib.suppress(OSError):
            self._receipt("quarantine", error_type=type(error).__name__, automatic_cleanup=False)

    def _initialize(self, owner_factory):
        self._context()
        if (
            self.dist.get_world_size(group=self.cpu_group) != len(self.members)
            or self.dist.get_rank(group=self.cpu_group) != self.members.index(self.rank)
            or self.dist.get_rank() != self.rank
            or any(device >= self.torch.npu.device_count() for device in self.config["physical_devices"])
        ):
            raise ValueError("CPU group order or visible physical NPU mapping differs")
        nonce = secrets.token_hex(32) if self.rank == self.members[0] else None
        hello = self._gather("bootstrap", dict(members=list(self.members), device=self.device_index, nonce=nonce))
        for position, payload in enumerate(hello):
            if (
                type(payload) is not dict
                or set(payload) != {"members", "device", "nonce"}
                or type(payload["members"]) is not list
                or any(type(rank) is not int for rank in payload["members"])
                or payload["members"] != list(self.members)
                or type(payload["device"]) is not int
                or payload["device"] != self.config["physical_devices"][position]
                or position != 0
                and payload["nonce"] is not None
            ):
                raise ValueError("bootstrap rank membership, physical device or nonce differs")
        self.session = hello[0]["nonce"]
        if (
            type(self.session) is not str
            or len(self.session) != 64
            or any(char not in "0123456789abcdef" for char in self.session)
        ):
            raise ValueError("bootstrap requires a fresh root 256-bit nonce")
        # Recheck pins after CPU agreement, immediately before loading code.
        validate_config(self.config, self.members, self.rank, self.device_index)
        self.torch.ops.load_library(self.config["torch_library"])
        self.initial_stream = self._stream()
        native_stream = self.initial_stream.npu_stream
        if type(native_stream) is not int or native_stream <= 0:
            raise ValueError("cold borrowed Torch stream requires a live native handle")
        groups = {device: index for index, group in enumerate(self.config["link_groups"]) for device in group}
        for geometry in self.config["geometries"]:
            schedule = BinomialSchedule(self.members, **geometry, algorithm=self.config["algorithm"])
            links = {
                peer: "hccs"
                if groups[self.device_index] == groups[self.config["physical_devices"][self.members.index(peer)]]
                else "pcie"
                for peer in schedule.peers(self.rank)
            }
            owner = owner_factory(
                schedule,
                self.rank,
                self.session,
                links,
                self.config["acl_library"],
                self.config["native_library"],
                acl_sha256=self.config["acl_sha256"],
                native_sha256=self.config["native_sha256"],
                device_index=self.device_index,
                stream=native_stream,
            )
            self.owners[(schedule.dtype, schedule.elements)] = owner
            pids = self._gather("pids:" + schedule.identity, owner.bare_pid())
            if any(type(pid) is not int or not 0 < pid < 2**31 for pid in pids):
                raise ValueError("every rank must supply an exact bare-device PID")
            exports = owner.prepare({peer: pids[self.members.index(peer)] for peer in owner.peers})
            published = self._gather("exports:" + schedule.identity, exports)
            owner.connect({peer: published[self.members.index(peer)][self.rank] for peer in owner.peers})
            active = self._gather("active:" + schedule.identity, owner.state)
            if any(state != "active" for state in active):
                raise RuntimeError("all ranks must activate each geometry before admission")
        self.state = "active"
        self._receipt("setup")

    def all_reduce(self, input, fallback):
        """Return queued native SUM or call fallback(input) before admission."""
        try:
            if self.state != "active":
                raise RuntimeError("Torch collective is not active: " + self.state)
            capturing = self._context(allow_capture=True)
            stream = self._stream(capturing=capturing)
            if not isinstance(input, self.torch.Tensor):
                key, owner = (type(input).__name__, None, None), None
            else:
                if input.device.type not in NPU_DEVICE_TYPES or input.device.index != self.device_index:
                    raise RuntimeError("collective input is on a different device")
                dtype = (
                    "float16"
                    if input.dtype == self.torch.float16
                    else ("float32" if input.dtype == self.torch.float32 else str(input.dtype))
                )
                key = (dtype, input.numel(), input.is_contiguous())
                owner = self.owners.get(key[:2]) if key[2] else None
            if owner is None:
                self.fallbacks[key] = self.fallbacks.get(key, 0) + 1
                return fallback(input)
            if owner.state != "active":
                raise RuntimeError("fixed geometry owner is not active")
            if self.claimed_stream_id is None:
                native_stream = stream.npu_stream
                if type(native_stream) is not int or native_stream <= 0:
                    raise RuntimeError("claimed Torch stream requires a live native handle")
                self.claimed_stream, self.claimed_stream_id = stream, stream.stream_id
                for fixed_owner in self.owners.values():
                    fixed_owner.stream.value = native_stream
            result = self.torch.ops.sidecard_comm.all_reduce(input, int(owner.plan.value), TIMEOUT_MS)
            owner.calls += 1
            return result
        except BaseException as error:
            self._quarantine(error)
            raise

    def validate_graph_replay(self):
        """Validate the serialized owner stream before native graph submission."""
        self._context()
        self._stream()

    def close(self, *, graphs_retired):
        """Explicit all-rank shutdown after graph and queued-handler retirement."""
        if graphs_retired is not True:
            raise ValueError("retire graphs and queued Torch handlers before explicit close")
        try:
            if self.state != "active":
                raise RuntimeError("cannot close inactive or quarantined Torch collective")
            self._context()
            self._stream()
            if self.graph_observer is not None and not self.graph_observer.retired:
                raise RuntimeError("explicit graph observer retirement is required before native owner release")
            self.torch.npu.synchronize(self.device_index)
            for owner in self.owners.values():
                phase = owner.schedule.identity
                drains = self._gather("drain:" + phase, owner.drain(graphs_retired=True))
                imports = owner.release_imports(drains)
                closures = self._gather("imports-closed:" + phase, imports)
                owner.close(closures, graphs_retired=True)
            self.state = "closed"
            self._receipt("shutdown", graphs_retired=True, ownership_released=True)
        except BaseException as error:
            self._quarantine(error)
            raise


def create(
    config,
    members,
    global_rank,
    cpu_group,
    device_index,
    *,
    torch,
    dist,
    stream_getter,
    owner_factory=NpuCollectiveOwner,
):
    """Construct a facade; setup exceptions retain it as error.collective."""
    collective = TorchCollective(
        config,
        members,
        global_rank,
        cpu_group,
        device_index,
        torch=torch,
        dist=dist,
        stream_getter=stream_getter,
    )
    try:
        collective._initialize(owner_factory)
    except BaseException as error:
        collective._quarantine(error)
        raise CollectiveSetupError(collective) from error
    return collective
