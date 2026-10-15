# SPDX-License-Identifier: Apache-2.0
"""Owner admission/lifetime tests with an SDK mock; never touch an NPU."""

import copy
import ctypes as c
import hashlib

import pytest

from tools.sidecard import npu_collective_acl as module
from tools.sidecard.npu_collective_schedule import BinomialSchedule

RESOURCE_CREATORS = frozenset(
    {
        "aclrtCreateStream",
        "aclrtMalloc",
        "aclrtCreateNotify",
        "aclrtIpcMemImportByKey",
        "aclrtNotifyImportByKey",
        "sidecard_exchange_create",
        "sidecard_group_create",
    }
)
RELEASES = frozenset(
    {
        "aclrtFree",
        "aclrtDestroyNotify",
        "aclrtDestroyStream",
        "aclrtIpcMemClose",
        "sidecard_exchange_destroy",
        "sidecard_group_destroy",
    }
)


class Function:
    def __init__(self, name, sdk):
        self.name, self.sdk = name, sdk

    def __call__(self, *args):
        name, sdk = self.name, self.sdk
        sdk.calls.append((name, args))
        sdk.counts[name] = sdk.counts.get(name, 0) + 1
        if name in sdk.abis:
            return sdk.abis[name]
        failing = sdk.fail == name and sdk.counts[name] == sdk.fail_occurrence
        if failing and not sdk.partial:
            return 77
        if name == "aclrtGetCurrentContext":
            args[0]._obj.value = sdk.context
        elif name == "aclrtDeviceGetBareTgid":
            args[0]._obj.value = sdk.bare_pid
        elif name == "sidecard_group_graph_lease_acquire":
            sdk.next_pointer += 1024
            args[1]._obj.value = None if sdk.null_lease else sdk.next_pointer
            if args[1]._obj.value:
                sdk.graph_leases[args[1]._obj.value] = args[0].value
        elif name == "sidecard_group_graph_lease_count":
            args[1]._obj.value = sum(group == args[0].value for group in sdk.graph_leases.values())
        elif name == "sidecard_group_graph_lease_release" and not failing:
            assert args[1] == 1
            if not sdk.uncleared_lease:
                del sdk.graph_leases[args[0]._obj.value]
                args[0]._obj.value = None
        elif name in RESOURCE_CREATORS:
            sdk.next_pointer += 1024 * 1024
            output = -1 if name == "sidecard_group_create" else (1 if name == "sidecard_exchange_create" else 0)
            args[output]._obj.value = 0 if sdk.null_resource == name else sdk.next_pointer
            if name == "sidecard_exchange_create":
                sdk.bindings.append(module.ExchangeBindings.from_buffer_copy(args[0]._obj))
            elif name == "sidecard_group_create":
                sdk.plans.append(
                    dict(
                        input=args[0].value,
                        output=args[1].value,
                        nbytes=args[2],
                        dtype=args[3],
                        device=args[4],
                        steps=[(s.exchange, s.operation, s.reserved) for s in args[5]],
                        count=args[6],
                    )
                )
        elif name in ("aclrtIpcMemGetExportKey", "aclrtNotifyGetExportKey"):
            args[2 if name == "aclrtIpcMemGetExportKey" else 1].value = f"export-{args[0].value}".encode()
        elif name in ("sidecard_group_destroy", "sidecard_exchange_destroy") and not failing:
            args[0]._obj.value = None
        return 77 if failing else 0


class Sdk:
    def __init__(self):
        self.functions, self.calls, self.counts, self.bindings, self.plans = {}, [], {}, [], []
        self.context, self.next_pointer, self.bare_pid = 123, 1024 * 1024, 5678
        self.fail, self.fail_occurrence, self.partial, self.null_resource = None, 1, False, None
        self.missing = set()
        self.graph_leases, self.null_lease, self.uncleared_lease = {}, False, False
        self.abis = dict(sidecard_exchange_abi=1, sidecard_exchange_bindings_bytes=120, sidecard_group_abi=1)

    def __getattr__(self, name):
        if name in self.missing:
            raise AttributeError(name)
        if name not in self.functions:
            self.functions[name] = Function(name, self)
        return self.functions[name]


@pytest.fixture
def factory(tmp_path, monkeypatch):
    sdk, paths, loads = Sdk(), [], []
    for name in ("acl.so", "native.so"):
        path = tmp_path / name
        path.write_bytes(("authenticated " + name).encode())
        paths.append(path)

    def load(path, **kwargs):
        loads.append((path, kwargs))
        return sdk

    monkeypatch.setattr(module.c, "CDLL", load)

    def owner(
        members=(0, 1), rank=None, dtype="float32", elements=4096, link="hccs", algorithm="binomial", **overrides
    ):
        schedule = BinomialSchedule(members, dtype, elements, algorithm=algorithm)
        rank = members[0] if rank is None else rank
        values = dict(
            schedule=schedule,
            rank=rank,
            session="c" * 64,
            links={peer: link for peer in schedule.peers(rank)},
            acl_library=paths[0],
            native_library=paths[1],
            acl_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
            native_sha256=hashlib.sha256(paths[1].read_bytes()).hexdigest(),
            device_index=0,
        )
        values.update(overrides)
        return module.NpuCollectiveOwner(**values)

    return sdk, owner, paths, loads


def packets(owner):
    return {
        peer: dict(
            session=owner.session,
            schedule_sha256=owner.schedule.identity,
            rank=peer,
            peer=owner.rank,
            keys={name: f"peer-{peer}-{name}" for name in ("inbox0", "inbox1", "ready0", "ready1")},
        )
        for peer in owner.peers
    }


def prepared(owner):
    return owner.prepare({peer: 1000 + peer for peer in owner.peers})


def active(owner):
    prepared(owner)
    owner.connect(packets(owner))
    return owner


def receipts(owner, imports_closed=False):
    return [
        dict(
            rank=rank,
            session=owner.session,
            schedule_sha256=owner.schedule.identity,
            calls=owner.calls,
            graphs_retired=True,
            **({"imports_closed": True} if imports_closed else {}),
        )
        for rank in owner.schedule.members
    ]


def names(sdk):
    return [name for name, _ in sdk.calls]


@pytest.mark.parametrize("algorithm", ("binomial", "recursive-doubling"))
def test_existing_eager_algorithms_do_not_require_or_bind_fixed_handshake_symbol(factory, algorithm):
    sdk, make, _, _ = factory
    sdk.missing.add("sidecard_exchange_set_fixed_handshake")
    owner = active(make(algorithm=algorithm))
    owner.enqueue(0x100000, 0x200000)
    assert owner.state == "active" and owner.calls == 1
    assert "sidecard_exchange_set_fixed_handshake" not in sdk.functions


def test_fixed_mode_requires_its_native_setter_before_any_sdk_resource(factory):
    sdk, make, _, _ = factory
    sdk.missing.add("sidecard_exchange_set_fixed_handshake")
    with pytest.raises(AttributeError, match="fixed_handshake"):
        make(algorithm="recursive-doubling-fixed")
    assert not any(name.startswith("acl") for name in names(sdk))
    assert not RESOURCE_CREATORS.intersection(names(sdk))


@pytest.mark.parametrize("members", ((0,), (7, 3), (7, 3, 9, 2)))
@pytest.mark.parametrize("dtype", ("float16", "float32"))
def test_fixed_mode_sets_each_exchange_immediately_before_group_admission(factory, members, dtype):
    sdk, make, _, _ = factory
    owner = active(make(members, dtype=dtype, algorithm="recursive-doubling-fixed"))
    function = sdk.functions["sidecard_exchange_set_fixed_handshake"]
    assert function.argtypes == [c.c_void_p] and function.restype is c.c_int
    calls = names(sdk)
    assert calls.count("sidecard_exchange_set_fixed_handshake") == len(owner.peers)
    for index, (name, args) in enumerate(sdk.calls):
        if name == "sidecard_exchange_set_fixed_handshake":
            assert calls[index - 1] == "aclrtGetCurrentContext"
            assert calls[index - 2] == "sidecard_exchange_create"
            assert args[0].value in {edge["handle"].value for edge in owner.edges.values()}
            assert index < calls.index("sidecard_group_create")
    assert all(step.sequence == 0 and step.slot == 0 for step in owner.steps)
    owner.enqueue(0x100000, 0x200000)
    owner.enqueue(0x100000, 0x200000)
    assert owner.calls == 2
    owner.drain(graphs_retired=True)
    owner.release_imports(receipts(owner))
    owner.close(receipts(owner, imports_closed=True), graphs_retired=True)
    assert owner.state == "closed"


@pytest.mark.parametrize("failure_occurrence", (1, 2))
def test_fixed_setter_failure_retains_imported_mappings_and_created_exchange_handles(factory, failure_occurrence):
    sdk, make, _, _ = factory
    owner = make((0, 1, 2, 3), algorithm="recursive-doubling-fixed")
    prepared(owner)
    sdk.fail, sdk.fail_occurrence = "sidecard_exchange_set_fixed_handshake", failure_occurrence
    with pytest.raises(RuntimeError, match="77"):
        owner.connect(packets(owner))
    assert owner.state == "quarantined" and owner.calls == 0
    assert sum(bool(edge["handle"].value) for edge in owner.edges.values()) == failure_occurrence
    assert sum(len(edge["imported"]) for edge in owner.edges.values()) == 4 * failure_occurrence
    assert owner.allocations and owner.acl is sdk and owner.native is sdk
    assert not owner.plan.value and "sidecard_group_create" not in names(sdk)
    assert not RELEASES.intersection(names(sdk))
    with pytest.raises(RuntimeError, match="quarantined"):
        owner.enqueue(0x100000, 0x200000)


@pytest.mark.parametrize("other_mode", ("binomial", "recursive-doubling"))
def test_fixed_mode_rejects_peer_using_another_handshake_identity_before_imports(factory, other_mode):
    sdk, make, _, _ = factory
    owner = make(algorithm="recursive-doubling-fixed")
    prepared(owner)
    peer_packets = packets(owner)
    peer_packets[1]["schedule_sha256"] = BinomialSchedule((0, 1), "float32", 4096, algorithm=other_mode).identity
    before = list(sdk.calls)
    with pytest.raises(ValueError, match="identity"):
        owner.connect(peer_packets)
    assert owner.state == "prepared" and sdk.calls == before


def test_eager_owner_never_requires_graph_lease_symbols_if_not_requested(factory):
    sdk, make, _, _ = factory
    sdk.missing.update({"sidecard_group_graph_lease_" + name for name in ("acquire", "release", "count")})
    owner = active(make())
    owner.drain(graphs_retired=True)
    owner.release_imports(receipts(owner))
    owner.close(receipts(owner, imports_closed=True), graphs_retired=True)
    assert owner.state == "closed"
    assert not any("graph_lease" in name for name in sdk.functions)


def test_cold_graph_leases_bind_exact_abi_and_block_drain_before_any_sdk_operation(factory):
    sdk, make, _, _ = factory
    owner = active(make())
    assert not any("graph_lease" in name for name in sdk.functions)
    first, second = owner.acquire_graph_lease(), owner.acquire_graph_lease()
    assert first.value and second.value and first is not second
    assert sdk.functions["sidecard_group_graph_lease_acquire"].argtypes == [c.c_void_p, c.POINTER(c.c_void_p)]
    assert sdk.functions["sidecard_group_graph_lease_release"].argtypes == [c.POINTER(c.c_void_p), c.c_uint32]
    assert sdk.functions["sidecard_group_graph_lease_count"].argtypes == [c.c_void_p, c.POINTER(c.c_uint64)]
    before = len(sdk.calls)
    with pytest.raises(ValueError, match="native graph lease"):
        owner.drain(graphs_retired=True)
    assert names(sdk)[before:] == ["sidecard_group_graph_lease_count"]
    assert owner.state == "active" and len(owner.graph_leases) == 2
    owner.release_graph_lease(second)
    assert second.value is None and owner.graph_leases == [first]
    owner.release_graph_lease(first)
    assert first.value is None and not owner.graph_leases
    before = len(sdk.calls)
    owner.drain(graphs_retired=True)
    assert names(sdk)[before:] == [
        "sidecard_group_graph_lease_count",
        "aclrtGetCurrentContext",
        "aclrtSynchronizeStreamWithTimeout",
    ]
    assert owner.state == "drained" and not RELEASES.intersection(names(sdk))


@pytest.mark.parametrize("symbol", ("acquire", "release", "count"))
def test_missing_cold_graph_lease_symbol_rejects_capture_without_automatic_release(factory, symbol):
    sdk, make, _, _ = factory
    owner = active(make())
    sdk.missing.add("sidecard_group_graph_lease_" + symbol)
    before = list(sdk.calls)
    with pytest.raises(AttributeError):
        owner.acquire_graph_lease()
    assert owner.state == "quarantined" and owner.plan.value and owner.allocations
    assert sdk.calls == before and not RELEASES.intersection(names(sdk))


@pytest.mark.parametrize("partial", (False, True))
def test_acquire_error_retains_partial_ticket_and_native_owner(factory, partial):
    sdk, make, _, _ = factory
    owner = active(make())
    sdk.fail, sdk.partial = "sidecard_group_graph_lease_acquire", partial
    with pytest.raises(RuntimeError, match="77"):
        owner.acquire_graph_lease()
    assert owner.state == "quarantined" and owner.plan.value
    assert len(owner.graph_leases) == 1 and bool(owner.graph_leases[0].value) is partial
    assert bool(sdk.graph_leases) is partial
    assert not RELEASES.intersection(names(sdk))


def test_successful_null_ticket_is_quarantined_and_retained(factory):
    sdk, make, _, _ = factory
    owner = active(make())
    sdk.null_lease = True
    with pytest.raises(RuntimeError, match="null ticket"):
        owner.acquire_graph_lease()
    assert owner.state == "quarantined" and len(owner.graph_leases) == 1
    assert owner.plan.value and not RELEASES.intersection(names(sdk))


@pytest.mark.parametrize("failure", ("native", "uncleared"))
def test_release_failure_retains_exact_ticket_without_teardown_or_retry(factory, failure):
    sdk, make, _, _ = factory
    owner = active(make())
    ticket = owner.acquire_graph_lease()
    if failure == "native":
        sdk.fail = "sidecard_group_graph_lease_release"
    else:
        sdk.uncleared_lease = True
    with pytest.raises(RuntimeError):
        owner.release_graph_lease(ticket)
    assert owner.state == "quarantined" and owner.graph_leases == [ticket] and ticket.value
    assert ticket.value in sdk.graph_leases
    assert not RELEASES.intersection(names(sdk))


def test_native_lease_count_is_authoritative_over_empty_python_registry(factory):
    sdk, make, _, _ = factory
    owner = active(make())
    ticket = owner.acquire_graph_lease()
    owner.graph_leases.clear()
    before = len(sdk.calls)
    with pytest.raises(ValueError, match="native graph lease"):
        owner.drain(graphs_retired=True)
    assert names(sdk)[before:] == ["sidecard_group_graph_lease_count"]
    assert ticket.value in sdk.graph_leases and owner.state == "active"


def test_native_lease_count_failure_quarantines_before_sdk_stream_sync(factory):
    sdk, make, _, _ = factory
    owner = active(make())
    ticket = owner.acquire_graph_lease()
    owner.release_graph_lease(ticket)
    sdk.fail = "sidecard_group_graph_lease_count"
    before = len(sdk.calls)
    with pytest.raises(RuntimeError, match="77"):
        owner.drain(graphs_retired=True)
    assert names(sdk)[before:] == ["sidecard_group_graph_lease_count"]
    assert owner.state == "quarantined" and owner.plan.value


@pytest.mark.parametrize("fault", ("alias", "foreign", "cleared", "integer"))
def test_release_requires_exact_live_owned_ticket_before_any_native_operation(factory, fault):
    sdk, make, _, _ = factory
    owner = active(make())
    ticket = owner.acquire_graph_lease()
    wrong = {
        "alias": c.c_void_p(ticket.value),
        "foreign": c.c_void_p(99),
        "cleared": c.c_void_p(),
        "integer": ticket.value,
    }[fault]
    before = list(sdk.calls)
    with pytest.raises(ValueError):
        owner.release_graph_lease(wrong)
    assert sdk.calls == before and owner.graph_leases == [ticket] and owner.state == "active"


@pytest.mark.parametrize("members", ((0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (7, 2, 9, 4)))
@pytest.mark.parametrize("dtype", ("float16", "float32"))
@pytest.mark.parametrize("link", ("hccs", "pcie"))
def test_all_rank_setup_native_steps_and_proven_shutdown(factory, members, dtype, link):
    sdk, make, _, loads = factory
    owners = [make(members, rank, dtype=dtype, link=link) for rank in members]
    exports = {owner.rank: prepared(owner) for owner in owners}
    for owner in owners:
        assert owner.bare_pid() == 5678
        bindings_start = len(sdk.bindings)
        owner.connect({peer: exports[peer][owner.rank] for peer in owner.peers})
        assert len(owner.allocations) == 3 + 2 * len(owner.peers)
        plan = sdk.plans[-1]
        assert plan == dict(
            input=owner.work_input.value,
            output=owner.work_output.value,
            nbytes=owner.nbytes,
            dtype=int(dtype == "float16"),
            device=0,
            steps=[(owner.edges[s.peer]["handle"].value, s.opcode, 0) for s in owner.steps],
            count=len(owner.steps),
        )
        for edge in owner.edges.values():
            assert (
                set(edge["local"])
                == set(edge["exported"])
                == set(edge["key_buffers"])
                == {"inbox0", "inbox1", "ready0", "ready1"}
            )
        for peer, binding in zip(owner.peers, sdk.bindings[bindings_start:]):
            edge = owner.edges[peer]
            assert (binding.input, binding.output, binding.workspace) == (
                owner.work_input.value,
                owner.work_output.value,
                owner.workspace.value,
            )
            for field in ("inbox0", "inbox1", "ready0", "ready1"):
                assert getattr(binding, field) == edge["local"][field].value
                imported_field = "peer_" + field
                assert getattr(binding, imported_field) == edge["imported"][field][0].value
            for field in ("inbox0", "inbox1"):
                assert any(
                    name == "aclrtIpcMemSetImportPid"
                    and args[0] == edge["exported"][field].encode()
                    and args[1]._obj.value == 1000 + peer
                    and args[2] == 1
                    for name, args in sdk.calls
                )
            for field in ("ready0", "ready1"):
                assert any(
                    name == "aclrtNotifySetImportPid"
                    and args[0].value == edge["local"][field].value
                    and args[1]._obj.value == 1000 + peer
                    and args[2] == 1
                    for name, args in sdk.calls
                )
        owner.enqueue(0xAB0000 + owner.rank * 4096, 0xCD0000 + owner.rank * 4096)
        assert owner.calls == 1
    for bindings in sdk.bindings:
        assert bindings.abi == 1 and bindings.dtype == int(dtype == "float16")
        assert bindings.workspace_bytes == 1024 * 1024
        assert bindings.elements == 4096
        assert all(getattr(bindings, field) for field, _ in module.ExchangeBindings._fields_[3:-1])
    for name, args in sdk.calls:
        if name == "aclrtMalloc":
            assert args[-1] == 3
        elif name.endswith("GetExportKey"):
            assert args[-1] == 0
        elif name in ("aclrtIpcMemSetImportPid", "aclrtNotifySetImportPid"):
            assert args[1]._obj.value in {1000 + rank for rank in members} and args[2] == 1
        elif name == "aclrtIpcMemImportByKey":
            assert args[-1] == (1 if link == "pcie" else 0)
        elif name == "aclrtNotifyImportByKey":
            assert args[-1] == (2 if link == "pcie" else 0)
        elif name == "sidecard_group_enqueue":
            assert args[-1] == 30000
    assert len(loads) == 2 * len(members)
    assert all(
        options["mode"] == (c.RTLD_GLOBAL if index % 2 == 0 else c.RTLD_LOCAL)
        for index, (_, options) in enumerate(loads)
    )
    assert not RELEASES.intersection(names(sdk))
    drained = [owner.drain(graphs_retired=True) for owner in owners]
    closed_imports = [owner.release_imports(drained) for owner in owners]
    for owner in owners:
        owner.close(closed_imports, graphs_retired=True)
        assert owner.state == "closed" and owner.plan.value is None
        assert all(not edge["imported"] and edge["handle"].value is None for edge in owner.edges.values())
    calls = names(sdk)
    assert calls.count("sidecard_exchange_create") == calls.count("sidecard_exchange_destroy") == 2 * (len(members) - 1)
    assert calls.count("aclrtCreateNotify") == 4 * (len(members) - 1)
    assert calls.count("aclrtFree") == sum(len(owner.allocations) for owner in owners)
    assert all(
        type(args[1]) is int and args[1] == 1
        for name, args in sdk.calls
        if name in ("sidecard_group_destroy", "sidecard_exchange_destroy")
    )
    assert calls.index("sidecard_group_destroy") < calls.index("aclrtFree")
    assert not any("Reset" in name or "Init" in name or "SetDevice" in name for name in calls)


@pytest.mark.parametrize(
    "name",
    (
        "aclrtCreateStream",
        "aclrtMalloc",
        "aclrtCreateNotify",
        "aclrtIpcMemGetExportKey",
        "aclrtNotifyGetExportKey",
        "aclrtIpcMemSetImportPid",
        "aclrtNotifySetImportPid",
    ),
)
@pytest.mark.parametrize("partial", (False, True))
def test_prepare_failure_retains_every_attached_owner_without_automatic_cleanup(factory, name, partial):
    sdk, make, _, _ = factory
    owner = make()
    sdk.fail, sdk.partial = name, partial
    with pytest.raises(RuntimeError, match="77"):
        prepared(owner)
    assert owner.state == "quarantined"
    assert not RELEASES.intersection(names(sdk))
    if name == "aclrtCreateStream":
        assert bool(owner.stream.value) is partial
    elif name == "aclrtMalloc":
        assert len(owner.allocations) == 1 and bool(owner.allocations[0].value) is partial
    else:
        assert len(owner.allocations) >= 4 and owner.edges[1]["key_buffers"]
        if name == "aclrtCreateNotify":
            assert bool(owner.edges[1]["local"]["ready0"].value) is partial
        elif name in ("aclrtIpcMemGetExportKey", "aclrtNotifyGetExportKey"):
            key = "inbox0" if name == "aclrtIpcMemGetExportKey" else "ready0"
            assert bool(owner.edges[1]["key_buffers"][key].value) is partial
    with pytest.raises(RuntimeError, match="quarantined"):
        owner.prepare({1: 1001})


@pytest.mark.parametrize(
    "name", ("aclrtIpcMemImportByKey", "aclrtNotifyImportByKey", "sidecard_exchange_create", "sidecard_group_create")
)
@pytest.mark.parametrize("partial", (False, True))
def test_connect_failure_retains_partial_imports_and_native_handles(factory, name, partial):
    sdk, make, _, _ = factory
    owner = make()
    prepared(owner)
    sdk.fail, sdk.partial = name, partial
    with pytest.raises(RuntimeError, match="77"):
        owner.connect(packets(owner))
    assert owner.state == "quarantined" and len(owner.allocations) == 5
    edge = owner.edges[1]
    if name in ("aclrtIpcMemImportByKey", "aclrtNotifyImportByKey"):
        key = "inbox0" if name == "aclrtIpcMemImportByKey" else "ready0"
        assert bool(edge["imported"][key][0].value) is partial
        assert edge["imported"][key][1] == packets(owner)[1]["keys"][key]
    elif name == "sidecard_exchange_create":
        assert bool(edge["handle"].value) is partial and len(edge["imported"]) == 4
    else:
        assert bool(owner.plan.value) is partial and edge["handle"].value
    assert edge["key_buffers"] and not RELEASES.intersection(names(sdk))


def test_null_successful_allocation_is_quarantined_and_retained(factory):
    sdk, make, _, _ = factory
    owner = make()
    sdk.null_resource = "aclrtMalloc"
    with pytest.raises(RuntimeError, match="null allocation"):
        prepared(owner)
    assert owner.state == "quarantined" and len(owner.allocations) == 1
    assert not RELEASES.intersection(names(sdk))


@pytest.mark.parametrize("name", ("sidecard_group_enqueue", "aclrtSynchronizeStreamWithTimeout"))
def test_execution_failure_quarantines_group_without_count_advance_or_free(factory, name):
    sdk, make, _, _ = factory
    owner = active(make())
    sdk.fail = name
    with pytest.raises(RuntimeError, match="77"):
        owner.enqueue(0x100000, 0x200000) if name == "sidecard_group_enqueue" else owner.drain(graphs_retired=True)
    assert owner.state == "quarantined" and owner.calls == 0
    assert not RELEASES.intersection(names(sdk))


@pytest.mark.parametrize("foreign", (True, 1.0, "1", 3, None))
def test_foreign_or_coerced_rank_rejected_before_any_library_load(factory, foreign):
    sdk, make, _, loads = factory
    with pytest.raises(ValueError):
        make(rank=foreign) if foreign is not None else make(rank=-1)
    assert not sdk.calls and not loads


@pytest.mark.parametrize("bad_links", ({True: "hccs"}, {1: "remote-read"}, {1: "other"}, {}, {1: "hccs", 3: "pcie"}))
def test_link_mode_and_exact_peer_key_types_rejected_before_loading(factory, bad_links):
    sdk, make, _, loads = factory
    with pytest.raises(ValueError):
        make(links=bad_links)
    assert not sdk.calls and not loads


@pytest.mark.parametrize(
    "bad_pids", ({True: 1001}, {1: True}, {1: 0}, {1: -1}, {1: 2**31}, {1: "1001"}, {}, {1: 1001, 2: 1002})
)
def test_invalid_peer_pid_or_key_rejected_before_stream_and_allocation(factory, bad_pids):
    sdk, make, _, _ = factory
    owner = make()
    before = list(sdk.calls)
    with pytest.raises(ValueError):
        owner.prepare(bad_pids)
    assert sdk.calls == before and owner.state == "new" and not owner.allocations


@pytest.mark.parametrize(
    "option",
    (
        dict(elements=16 * 1024 * 1024 + 1),
        dict(device_index=True),
        dict(device_index=-1),
        dict(stream=True),
        dict(stream=0),
        dict(stream="123"),
        dict(session="z" * 64),
    ),
)
def test_invalid_geometry_device_stream_or_session_rejected_before_loading(factory, option):
    sdk, make, _, loads = factory
    with pytest.raises(ValueError):
        make(**option)
    assert not sdk.calls and not loads


@pytest.mark.parametrize("library_index", (0, 1))
def test_library_drift_before_prepare_rejected_before_resource_allocation(factory, library_index):
    sdk, make, paths, _ = factory
    owner = make()
    paths[library_index].write_bytes(b"changed after constructor authentication")
    before = list(sdk.calls)
    with pytest.raises(ValueError, match="SHA256|library"):
        prepared(owner)
    assert sdk.calls == before and owner.state == "new" and not owner.allocations


@pytest.mark.parametrize("getter", ("sidecard_exchange_abi", "sidecard_exchange_bindings_bytes", "sidecard_group_abi"))
def test_native_abi_mismatch_fails_before_sdk_context_or_resources(factory, getter):
    sdk, make, _, _ = factory
    sdk.abis[getter] += 1
    with pytest.raises(ValueError, match="ABI"):
        make()
    assert not any(name.startswith("acl") for name in names(sdk))


@pytest.mark.parametrize("library_index", (0, 1))
def test_binary_authentication_rejects_bad_hash_before_that_library_load(factory, library_index):
    sdk, make, _, loads = factory
    key = "acl_sha256" if library_index == 0 else "native_sha256"
    with pytest.raises(ValueError, match="SHA256"):
        make(**{key: "0" * 64})
    assert len(loads) == library_index and not sdk.calls


@pytest.mark.parametrize("identity", ("context", "process", "thread"))
def test_owner_identity_change_blocks_native_submission_and_cleanup(factory, monkeypatch, identity):
    sdk, make, _, _ = factory
    owner = active(make())
    before = sdk.counts.get("sidecard_group_enqueue", 0)
    if identity == "context":
        sdk.context += 1
    elif identity == "process":
        monkeypatch.setattr(module.os, "getpid", lambda: owner.pid + 1)
    else:
        monkeypatch.setattr(module.threading, "get_ident", lambda: owner.thread + 1)
    with pytest.raises(RuntimeError, match="context changed|another process or thread"):
        owner.enqueue(0x100000, 0x200000)
    assert sdk.counts.get("sidecard_group_enqueue", 0) == before
    assert owner.calls == 0 and not RELEASES.intersection(names(sdk))


@pytest.mark.parametrize(
    "mutation",
    (
        "session",
        "schedule",
        "rank-bool",
        "peer",
        "missing-key",
        "duplicate-key",
        "duplicate-notify-key",
        "nul-key",
        "foreign-peer",
        "bool-peer-key",
        "unknown-field",
    ),
)
def test_malformed_peer_packet_rejected_before_any_import(factory, mutation):
    sdk, make, _, _ = factory
    owner = make()
    prepared(owner)
    packet = packets(owner)
    if mutation == "session":
        packet[1]["session"] = "a" * 64
    elif mutation == "schedule":
        packet[1]["schedule_sha256"] = BinomialSchedule((0, 1), "float16", 4096).identity
    elif mutation == "rank-bool":
        packet[1]["rank"] = True
    elif mutation == "peer":
        packet[1]["peer"] = 2
    elif mutation == "missing-key":
        del packet[1]["keys"]["inbox1"]
    elif mutation == "duplicate-key":
        packet[1]["keys"]["inbox1"] = packet[1]["keys"]["inbox0"]
    elif mutation == "duplicate-notify-key":
        packet[1]["keys"]["ready1"] = packet[1]["keys"]["ready0"]
    elif mutation == "nul-key":
        packet[1]["keys"]["inbox0"] = "invalid\0key"
    elif mutation == "foreign-peer":
        packet[2] = packet.pop(1)
    elif mutation == "bool-peer-key":
        packet = {True: packet[1]}
    else:
        packet[1]["extra"] = 1
    before = list(sdk.calls)
    with pytest.raises(ValueError):
        owner.connect(packet)
    assert sdk.calls == before and owner.state == "prepared"
    assert all(not edge["imported"] for edge in owner.edges.values())


def test_memory_and_notify_exports_may_share_key_text(factory):
    sdk, make, _, _ = factory
    owner = make()
    prepared(owner)
    packet = packets(owner)
    for export in packet.values():
        export["keys"]["ready0"] = export["keys"]["inbox0"]
        export["keys"]["ready1"] = export["keys"]["inbox1"]
    owner.connect(packet)
    assert owner.state == "active"
    assert sdk.counts["aclrtIpcMemImportByKey"] == 2 * len(owner.peers)
    assert sdk.counts["aclrtNotifyImportByKey"] == 2 * len(owner.peers)


def test_shutdown_phase_order_requires_local_drain_before_import_release(factory):
    sdk, make, _, _ = factory
    owner = active(make())
    before = list(sdk.calls)
    with pytest.raises(RuntimeError, match="state"):
        owner.release_imports(receipts(owner))
    with pytest.raises(RuntimeError, match="state"):
        owner.close(receipts(owner, imports_closed=True), graphs_retired=True)
    assert sdk.calls == before
    owner.drain(graphs_retired=True)
    before = list(sdk.calls)
    with pytest.raises(RuntimeError, match="state"):
        owner.close(receipts(owner, imports_closed=True), graphs_retired=True)
    assert sdk.calls == before and owner.state == "drained"


@pytest.mark.parametrize("phase", ("drained", "imports-closed"))
@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "duplicate",
        "rank-bool",
        "rank-foreign",
        "calls-bool",
        "calls-stale",
        "session",
        "schedule",
        "graphs-int",
        "extra",
        "imports-int",
    ),
)
def test_shutdown_requires_exact_all_rank_typed_receipts_before_releasing_resources(factory, phase, mutation):
    sdk, make, _, _ = factory
    owner = active(make())
    owner.enqueue(0x100000, 0x200000)
    owner.drain(graphs_retired=True)
    if phase == "imports-closed":
        owner.release_imports(receipts(owner))
    proof = copy.deepcopy(receipts(owner, imports_closed=phase == "imports-closed"))
    if mutation == "missing":
        proof.pop()
    elif mutation == "duplicate":
        proof[1] = copy.deepcopy(proof[0])
    elif mutation == "rank-bool":
        proof[1]["rank"] = True
    elif mutation == "rank-foreign":
        proof[1]["rank"] = 9
    elif mutation == "calls-bool":
        proof[1]["calls"] = True
    elif mutation == "calls-stale":
        proof[1]["calls"] = 0
    elif mutation == "session":
        proof[1]["session"] = "a" * 64
    elif mutation == "schedule":
        proof[1]["schedule_sha256"] = "a" * 64
    elif mutation == "graphs-int":
        proof[1]["graphs_retired"] = 1
    elif mutation == "extra":
        proof[1]["extra"] = 1
    elif phase == "imports-closed":
        proof[1]["imports_closed"] = 1
    else:
        proof[1]["imports_closed"] = True
    before = list(sdk.calls)
    with pytest.raises(ValueError):
        owner.release_imports(proof) if phase == "drained" else owner.close(proof, graphs_retired=True)
    assert sdk.calls == before and owner.state == phase


@pytest.mark.parametrize("truthy", (False, 1, "true", None))
def test_graph_retirement_requires_exact_true_before_drain_and_close(factory, truthy):
    sdk, make, _, _ = factory
    owner = active(make())
    before = list(sdk.calls)
    with pytest.raises(ValueError, match="graph"):
        owner.drain(graphs_retired=truthy)
    assert sdk.calls == before and owner.state == "active"
    owner.drain(graphs_retired=True)
    owner.release_imports(receipts(owner))
    before = list(sdk.calls)
    with pytest.raises(ValueError, match="graph"):
        owner.close(receipts(owner, imports_closed=True), graphs_retired=truthy)
    assert sdk.calls == before and owner.state == "imports-closed"


def test_borrowed_stream_is_used_and_never_created_or_destroyed(factory):
    sdk, make, _, _ = factory
    owner = active(make(stream=0xBEEF))
    owner.enqueue(0x100000, 0x200000)
    enqueue = next(args for name, args in sdk.calls if name == "sidecard_group_enqueue")
    assert enqueue[3].value == 0xBEEF
    owner.drain(graphs_retired=True)
    owner.release_imports(receipts(owner))
    owner.close(receipts(owner, imports_closed=True), graphs_retired=True)
    assert "aclrtCreateStream" not in names(sdk) and "aclrtDestroyStream" not in names(sdk)


@pytest.mark.parametrize(
    "failure",
    (
        "aclrtIpcMemClose",
        "aclrtDestroyNotify",
        "sidecard_group_destroy",
        "sidecard_exchange_destroy",
        "aclrtFree",
        "aclrtDestroyStream",
    ),
)
def test_shutdown_failure_retains_remaining_owners_and_never_resets_or_retries(factory, failure):
    sdk, make, _, _ = factory
    owner = active(make())
    owner.drain(graphs_retired=True)
    if failure in ("aclrtIpcMemClose", "aclrtDestroyNotify"):
        sdk.fail = failure
        action = lambda: owner.release_imports(receipts(owner))
    else:
        owner.release_imports(receipts(owner))
        sdk.fail = failure
        action = lambda: owner.close(receipts(owner, imports_closed=True), graphs_retired=True)
    with pytest.raises(RuntimeError, match="77"):
        action()
    assert owner.state == "quarantined" and owner.allocations and owner.edges
    assert sdk.counts[failure] >= 1
    before = list(sdk.calls)
    with pytest.raises(RuntimeError, match="quarantined"):
        action()
    assert sdk.calls == before
    assert not any("Reset" in name for name in names(sdk))
