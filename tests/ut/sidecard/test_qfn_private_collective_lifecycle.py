# SPDX-License-Identifier: Apache-2.0
"""Cold private lifecycle failure gates; no device or process groups created."""

from types import SimpleNamespace

import pytest

from tools.sidecard import npu_collective_torch as base
from tools.sidecard import qfn_private_collective_lifecycle as lifecycle


def runtime(monkeypatch, *, peer_failure=None, owner_failure=None):
    events, retained, owners = [], [], []
    config = dict(
        algorithm="recursive-doubling-fixed",
        execution_mode="eager",
        physical_devices=[0, 1, 2, 3],
        link_groups=[[0, 1], [2, 3]],
        geometries=[dict(dtype="float16", elements=2560)],
        acl_library="/private/acl.so",
        acl_sha256="acl",
        native_library="/private/native.so",
        native_sha256="native",
        torch_library="/private/torch.so",
        output_root="/private",
    )
    monkeypatch.setattr(base, "validate_config", lambda value, *args: dict(value))
    monkeypatch.setattr(lifecycle, "validate_config", lambda value, *args: dict(value))

    def receipt(self, phase, **kwargs):
        events.append(("receipt", phase))

    monkeypatch.setattr(lifecycle.PrivateTorchCollective, "_receipt", receipt)
    stream = SimpleNamespace(stream_id=7, device=SimpleNamespace(type="npu", index=0), npu_stream=123)
    torch = SimpleNamespace(
        npu=SimpleNamespace(
            current_device=lambda: 0,
            is_current_stream_capturing=lambda: False,
            device_count=lambda: 4,
            synchronize=lambda device: events.append(("torch", "sync")),
        ),
        ops=SimpleNamespace(load_library=lambda value: events.append(("load", value))),
    )

    def gather(output, packet, *, group):
        events.append(("gather", packet["phase"]))
        for rank in range(4):
            payload = packet["payload"]
            if packet["phase"] == "bootstrap":
                payload = dict(members=[0, 1, 2, 3], device=rank, nonce=payload["nonce"] if rank == 0 else None)
            elif packet["phase"].startswith("exports:"):
                payload = {peer: {} for peer in range(4)}
            output[rank] = dict(packet, rank=rank, payload=payload)

    dist = SimpleNamespace(get_world_size=lambda **kwargs: 4, get_rank=lambda **kwargs: 0, all_gather_object=gather)

    def checked(phase, action):
        events.append(("checked", phase))
        value = action()
        if phase == peer_failure:
            raise RuntimeError("agreed peer failure at " + phase)
        return value

    class Owner:
        def __init__(self, schedule, rank, *args, **kwargs):
            self.schedule, self.rank = schedule, rank
            self.peers = schedule.peers(rank)
            self.state, self.calls, self.allocations = "new", 0, []
            owners.append(self)
            self.action("construct")

        def action(self, name):
            events.append(("owner", name))
            if name == owner_failure:
                raise RuntimeError("local SDK failure at " + name)

        def bare_pid(self):
            self.action("pid")
            return 11

        def prepare(self, peers):
            assert retained and self in retained[0].owners.values()
            self.allocations.append(object())
            self.action("prepare")
            self.state = "prepared"
            return {peer: {} for peer in peers}

        def connect(self, peers):
            self.action("connect")
            self.state = "active"

        def drain(self, *, graphs_retired):
            assert graphs_retired is True
            self.action("drain")
            self.state = "drained"
            return {}

        def release_imports(self, receipts):
            self.action("imports")
            self.state = "imports-closed"
            return {}

        def close(self, receipts, *, graphs_retired):
            assert graphs_retired is True
            self.action("release")
            self.state = "closed"

    def create():
        return lifecycle.create_private_collective(
            config,
            (0, 1, 2, 3),
            0,
            object(),
            0,
            torch=torch,
            dist=dist,
            stream_getter=lambda: stream,
            checked=checked,
            retain=retained.append,
            owner_factory=Owner,
        )

    return SimpleNamespace(create=create, events=events, retained=retained, owners=owners)


def test_private_success_preserves_qualified_hot_path_and_agrees_every_release(monkeypatch):
    run = runtime(monkeypatch)
    collective = run.create()
    assert collective is run.retained[0]
    assert collective.graph_observer is None
    assert lifecycle.PrivateTorchCollective.all_reduce is base.TorchCollective.all_reduce
    collective.close(graphs_retired=True)
    assert collective.state == "closed" and run.owners[0].state == "closed"
    stages = [name for kind, name in run.events if kind == "checked"]
    assert stages == [
        "private-construct",
        "private-retain",
        "private-context",
        "private-bootstrap-validated",
        "private-load-stream",
        "private-owner-0-construct",
        "private-owner-0-pid",
        "private-owner-0-prepare",
        "private-owner-0-connect",
        "private-owner-0-active",
        "private-setup-receipt",
        "private-close-preflight",
        "private-close-0-drain",
        "private-close-0-imports",
        "private-close-0-release",
        "private-shutdown-receipt",
    ]


@pytest.mark.parametrize(
    "stage",
    [
        "private-context",
        "private-bootstrap-validated",
        "private-load-stream",
        "private-owner-0-construct",
        "private-owner-0-pid",
        "private-owner-0-prepare",
        "private-owner-0-connect",
        "private-owner-0-active",
        "private-setup-receipt",
    ],
)
def test_peer_setup_error_stops_before_next_metadata_collective_and_retains(monkeypatch, stage):
    run = runtime(monkeypatch, peer_failure=stage)
    with pytest.raises(base.CollectiveSetupError) as failure:
        run.create()
    facade = run.retained[0]
    assert failure.value.collective_owner is facade
    assert facade.state == "quarantined"
    failure_index = run.events.index(("checked", stage))
    assert not any(kind == "gather" for kind, name in run.events[failure_index + 1 :])
    assert ("owner", "release") not in run.events
    assert all(owner.state == "quarantined" for owner in facade.owners.values())


@pytest.mark.parametrize("stage", ["construct", "pid", "prepare", "connect"])
def test_local_setup_error_preserves_partial_owner_without_automatic_cleanup(monkeypatch, stage):
    run = runtime(monkeypatch, owner_failure=stage)
    with pytest.raises(base.CollectiveSetupError):
        run.create()
    facade = run.retained[0]
    assert facade.state == "quarantined"
    if stage in ("prepare", "connect"):
        assert run.owners[0].allocations and run.owners[0] in facade.owners.values()
    assert ("owner", "release") not in run.events


@pytest.mark.parametrize(
    "stage", ["private-close-preflight", "private-close-0-drain", "private-close-0-imports", "private-close-0-release"]
)
def test_peer_teardown_error_never_advances_to_next_release_or_metadata_collective(monkeypatch, stage):
    run = runtime(monkeypatch, peer_failure=stage)
    facade = run.create()
    with pytest.raises(RuntimeError, match="agreed peer failure"):
        facade.close(graphs_retired=True)
    assert facade.state == "quarantined" and run.retained[0] is facade
    failure_index = run.events.index(("checked", stage))
    assert not any(kind == "gather" for kind, name in run.events[failure_index + 1 :])
    if stage != "private-close-0-release":
        assert ("owner", "release") not in run.events


@pytest.mark.parametrize("stage", ["drain", "imports", "release"])
def test_local_teardown_error_retains_uncertain_native_owner(monkeypatch, stage):
    run = runtime(monkeypatch, owner_failure=stage)
    facade = run.create()
    with pytest.raises(RuntimeError, match="local SDK failure"):
        facade.close(graphs_retired=True)
    assert facade.state == "quarantined"
    assert run.owners[0] in facade.owners.values()
    assert run.owners[0].allocations


def test_nonretired_close_is_collectively_rejected_before_device_drain(monkeypatch):
    run = runtime(monkeypatch)
    facade = run.create()
    with pytest.raises(ValueError, match="retired queued handlers"):
        facade.close(graphs_retired=False)
    assert ("torch", "sync") not in run.events
    assert ("owner", "drain") not in run.events
