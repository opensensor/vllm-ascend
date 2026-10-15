# SPDX-License-Identifier: Apache-2.0
"""Cold, phase-agreed lifecycle for a private resident collective experiment.

Only local SDK operations are wrapped. The qualified tensor submission path is
inherited unchanged, and the serving facade is never patched. ``checked`` must
agree every rank's local result/error before returning or raising. An SDK call
that hangs, or failure of the CPU group itself, cannot be recovered by this
protocol; callers must keep inference paused until coordinated recovery.
"""

import secrets

from .npu_collective_acl import NpuCollectiveOwner
from .npu_collective_schedule import BinomialSchedule
from .npu_collective_torch import CollectiveSetupError, TorchCollective, validate_config


class PrivateTorchCollective(TorchCollective):
    """Reuse native owners with collective error agreement around cold stages."""

    def __init__(self, *args, checked, **kwargs):
        super().__init__(*args, **kwargs)
        self.checked = checked

    def _initialize_private(self, owner_factory):
        # No device ownership is acquired until every rank has validated the
        # borrowed context, group mapping, pins and stream. Each owner is attached
        # before prepare can create any allocation or exported/imported handle.
        def context():
            self._context()
            if (
                self.config.get("execution_mode", "eager") != "eager"
                or self.graph_observer is not None
                or self.dist.get_world_size(group=self.cpu_group) != len(self.members)
                or self.dist.get_rank(group=self.cpu_group) != self.members.index(self.rank)
                or self.dist.get_rank() != self.rank
                or any(device >= self.torch.npu.device_count() for device in self.config["physical_devices"])
            ):
                raise ValueError("private eager group or physical NPU mapping differs")
            return secrets.token_hex(32) if self.rank == self.members[0] else None

        nonce = self.checked("private-context", context)
        hello = self._gather("bootstrap", dict(members=list(self.members), device=self.device_index, nonce=nonce))

        def validate_hello():
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
                    raise ValueError("private bootstrap membership, physical device or nonce differs")
            self.session = hello[0]["nonce"]
            if (
                type(self.session) is not str
                or len(self.session) != 64
                or any(value not in "0123456789abcdef" for value in self.session)
            ):
                raise ValueError("private bootstrap requires a fresh root 256-bit nonce")

        self.checked("private-bootstrap-validated", validate_hello)

        def load_stream():
            validate_config(self.config, self.members, self.rank, self.device_index)
            self.torch.ops.load_library(self.config["torch_library"])
            self.initial_stream = self._stream()
            native_stream = self.initial_stream.npu_stream
            if type(native_stream) is not int or native_stream <= 0:
                raise ValueError("private borrowed stream requires a live native handle")
            return native_stream

        native_stream = self.checked("private-load-stream", load_stream)
        groups = {device: index for index, group in enumerate(self.config["link_groups"]) for device in group}
        for index, geometry in enumerate(self.config["geometries"]):
            prefix = f"private-owner-{index}"

            def construct(geometry=geometry):
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
                return owner

            owner = self.checked(prefix + "-construct", construct)
            phase = owner.schedule.identity
            pid = self.checked(prefix + "-pid", owner.bare_pid)
            pids = self._gather("pids:" + phase, pid)

            def prepare(owner=owner, pids=pids):
                if any(type(value) is not int or not 0 < value < 2**31 for value in pids):
                    raise ValueError("every private rank must supply an exact bare-device PID")
                return owner.prepare({peer: pids[self.members.index(peer)] for peer in owner.peers})

            exports = self.checked(prefix + "-prepare", prepare)
            published = self._gather("exports:" + phase, exports)
            self.checked(
                prefix + "-connect",
                lambda owner=owner, published=published: owner.connect(
                    {peer: published[self.members.index(peer)][self.rank] for peer in owner.peers}
                ),
            )
            states = self._gather("active:" + phase, owner.state)

            def validate_active(states=states):
                if any(state != "active" for state in states):
                    raise RuntimeError("all private ranks must activate before admission")

            self.checked(prefix + "-active", validate_active)
        self.state = "active"
        self.checked("private-setup-receipt", lambda: self._receipt("setup"))

    def close(self, *, graphs_retired):
        """Agree drain, import closure, and release before advancing ranks."""
        try:

            def preflight():
                if graphs_retired is not True or self.graph_observer is not None or self.state != "active":
                    raise ValueError("private close requires active eager owners and retired queued handlers")
                self._context()
                self._stream()
                self.torch.npu.synchronize(self.device_index)

            self.checked("private-close-preflight", preflight)
            for index, owner in enumerate(self.owners.values()):
                prefix = f"private-close-{index}"
                phase = owner.schedule.identity
                local_drain = self.checked(prefix + "-drain", lambda owner=owner: owner.drain(graphs_retired=True))
                drains = self._gather("drain:" + phase, local_drain)
                imports = self.checked(
                    prefix + "-imports", lambda owner=owner, drains=drains: owner.release_imports(drains)
                )
                closures = self._gather("imports-closed:" + phase, imports)
                self.checked(
                    prefix + "-release",
                    lambda owner=owner, closures=closures: owner.close(closures, graphs_retired=True),
                )
            self.state = "closed"
            self.checked(
                "private-shutdown-receipt",
                lambda: self._receipt("shutdown", graphs_retired=True, ownership_released=True),
            )
        except BaseException as error:
            self._quarantine(error)
            raise


def create_private_collective(
    config,
    members,
    global_rank,
    cpu_group,
    device_index,
    *,
    torch,
    dist,
    stream_getter,
    checked,
    retain,
    owner_factory=NpuCollectiveOwner,
):
    """Retain a separate facade before any native owner acquires resources."""
    collective = checked(
        "private-construct",
        lambda: PrivateTorchCollective(
            config,
            members,
            global_rank,
            cpu_group,
            device_index,
            torch=torch,
            dist=dist,
            stream_getter=stream_getter,
            checked=checked,
        ),
    )
    try:
        checked("private-retain", lambda: retain(collective))
        collective._initialize_private(owner_factory)
    except BaseException as error:
        collective._quarantine(error)
        raise CollectiveSetupError(collective) from error
    return collective
