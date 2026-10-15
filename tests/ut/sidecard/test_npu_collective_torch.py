# SPDX-License-Identifier: Apache-2.0
"""Dependency-injected eager facade tests; no accelerator qualification."""

import copy
import ctypes
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.sidecard.npu_collective_acl import NpuCollectiveOwner
from tools.sidecard.npu_collective_torch import CollectiveSetupError, TorchCollective, create, validate_config


class FakeTensor:
    def __init__(self, dtype="float16", elements=32, *, contiguous=True, device=0, device_type="npu"):
        self.dtype, self.elements, self.contiguous = dtype, elements, contiguous
        self.device = SimpleNamespace(type=device_type, index=device)

    def numel(self):
        return self.elements

    def is_contiguous(self):
        return self.contiguous


class FakeStream:
    def __init__(self, stream_id=10, handle=9000, device=0):
        self.stream_id, self.handle = stream_id, handle
        self.device = SimpleNamespace(type="npu", index=device)
        self.handle_reads = 0

    @property
    def npu_stream(self):
        self.handle_reads += 1
        return self.handle


class FakeOwner:
    def __init__(self, env, schedule, rank, session, links, acl_library, native_library, **kwargs):
        self.env, self.schedule, self.rank, self.session = env, schedule, rank, session
        self.peers, self.links = schedule.peers(rank), links
        self.stream = ctypes.c_void_p(kwargs["stream"])
        self.plan = ctypes.c_void_p(123456 + len(env.owners))
        self.calls, self.state = 0, "new"
        self.graph_leases = []
        self.acl_library, self.native_library, self.kwargs = acl_library, native_library, kwargs
        env.owners.append(self)
        env.events.append(("owner", schedule.identity))

    def acquire_graph_lease(self):
        lease = object()
        self.graph_leases.append(lease)
        return lease

    def release_graph_lease(self, lease):
        self.graph_leases.remove(lease)

    def bare_pid(self):
        return 100 + self.env.members.index(self.rank)

    def exports(self, rank):
        return {
            peer: dict(
                session=self.session,
                schedule_sha256=self.schedule.identity,
                rank=rank,
                peer=peer,
                keys="IPC_PRIVATE_DO_NOT_PUBLISH",
            )
            for peer in self.schedule.peers(rank)
        }

    def prepare(self, peer_pids):
        assert peer_pids == {peer: 100 + self.env.members.index(peer) for peer in self.peers}
        self.state = "prepared"
        self.env.events.append(("prepare", self.schedule.identity))
        if self.env.prepare_error:
            raise RuntimeError("fake uncertain preparation containing IPC_PRIVATE_DO_NOT_PUBLISH")
        return self.exports(self.rank)

    def connect(self, exports):
        assert exports == {peer: self.exports(peer)[self.rank] for peer in self.peers}
        self.state = "active"
        self.env.events.append(("connect", self.schedule.identity))

    def drain(self, *, graphs_retired):
        assert graphs_retired is True
        self.env.events.append(("drain", self.schedule.identity))
        self.state = "drained"
        return dict(
            rank=self.rank,
            session=self.session,
            schedule_sha256=self.schedule.identity,
            calls=self.calls,
            graphs_retired=True,
        )

    def release_imports(self, receipts):
        NpuCollectiveOwner._receipts(self, receipts, imports_closed=False)
        self.env.events.append(("release", self.schedule.identity))
        self.state = "imports-closed"
        return dict(receipts[self.env.members.index(self.rank)], imports_closed=True)

    def close(self, receipts, *, graphs_retired):
        assert graphs_retired is True
        NpuCollectiveOwner._receipts(self, receipts, imports_closed=True)
        self.env.events.append(("close", self.schedule.identity))
        self.state = "closed"


class Environment:
    def __init__(self, tmp_path, members=(0, 1), rank=None, algorithm="binomial"):
        self.members, self.rank = members, members[0] if rank is None else rank
        self.devices = list(range(len(members)))
        self.device = self.devices[members.index(self.rank)]
        self.group = object()
        self.events, self.owners, self.op_calls, self.gathers = [], [], [], []
        self.current_stream = FakeStream(device=self.device)
        self.capturing, self.prepare_error, self.op_error, self.sync_error = False, False, False, False
        self.group_size, self.group_rank, self.world_rank = len(members), members.index(self.rank), self.rank
        self.mutate_gather = None
        self.config = dict(
            schema_version=1,
            algorithm=algorithm,
            geometries=[dict(dtype="float16", elements=32), dict(dtype="float32", elements=64)],
            physical_devices=self.devices,
            link_groups=[self.devices[:2]] + ([self.devices[2:]] if len(members) > 2 else []),
            output_root=str(tmp_path / "receipts"),
        )
        for name in ("acl", "native", "torch"):
            path = tmp_path / (name + ".so")
            path.write_bytes(("fake pinned " + name).encode())
            self.config[name + "_library"] = str(path)
            self.config[name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.torch = SimpleNamespace(
            Tensor=FakeTensor,
            float16="float16",
            float32="float32",
            npu=SimpleNamespace(
                current_device=lambda: self.device,
                device_count=lambda: len(self.devices),
                is_current_stream_capturing=lambda: self.capturing,
                synchronize=self.synchronize,
            ),
            ops=SimpleNamespace(
                load_library=lambda path: self.events.append(("load", path)),
                sidecard_comm=SimpleNamespace(all_reduce=self.native_op),
            ),
        )
        self.dist = SimpleNamespace(
            get_world_size=lambda *, group: self.group_size,
            get_rank=lambda group=None: self.world_rank if group is None else self.group_rank,
            all_gather_object=self.all_gather,
        )

    def synchronize(self, device):
        assert device == self.device
        self.events.append(("torch-synchronize", device))
        if self.sync_error:
            raise RuntimeError("fake delayed Torch queue error")

    def native_op(self, tensor, plan, timeout):
        self.op_calls.append((tensor, plan, timeout))
        if self.op_error:
            raise RuntimeError("fake possibly partial native operation")
        return FakeTensor(tensor.dtype, tensor.elements, device=tensor.device.index)

    def all_gather(self, result, local, *, group):
        assert group is self.group
        self.gathers.append(copy.deepcopy(local))
        phase = local["phase"]
        owner = self.owners[-1] if self.owners else None
        root_nonce = local["payload"]["nonce"] if phase == "bootstrap" and self.rank == self.members[0] else "b" * 64
        for position, rank in enumerate(self.members):
            if phase == "bootstrap":
                payload = dict(
                    members=list(self.members),
                    device=self.devices[position],
                    nonce=root_nonce if position == 0 else None,
                )
            elif phase.startswith("pids:"):
                payload = 100 + position
            elif phase.startswith("exports:"):
                payload = owner.exports(rank)
            elif phase.startswith("active:"):
                payload = "active"
            elif phase.startswith(("drain:", "imports-closed:")):
                payload = dict(local["payload"], rank=rank)
            else:
                pytest.fail("unexpected CPU collective phase " + phase)
            result[position] = dict(local, rank=rank, payload=payload)
        if self.mutate_gather is not None:
            self.mutate_gather(phase, result)
        self.events.append(("gather", phase))

    def create(self):
        return create(
            self.config,
            self.members,
            self.rank,
            self.group,
            self.devices[self.members.index(self.rank)],
            torch=self.torch,
            dist=self.dist,
            stream_getter=lambda: self.current_stream,
            owner_factory=lambda *args, **kwargs: FakeOwner(self, *args, **kwargs),
        )

    def receipts(self, phase):
        return list(Path(self.config["output_root"]).glob(phase + "-*.json"))


@pytest.fixture
def env(tmp_path):
    return Environment(tmp_path)


@pytest.mark.parametrize("members", ((0,), (4, 2), (8, 3, 5), (9, 2, 7, 4)))
@pytest.mark.parametrize("member_position", (0, -1))
def test_setup_all_ranks_authenticate_ordered_identity_and_activate_each_geometry(tmp_path, members, member_position):
    env = Environment(tmp_path, members, members[member_position])
    facade = env.create()
    assert facade.state == "active" and len(facade.owners) == 2
    assert all(owner.state == "active" for owner in env.owners)
    assert len({owner.session for owner in env.owners}) == 1
    assert len(env.gathers) == 1 + 3 * len(env.owners)
    assert len([event for event in env.events if event[0] == "load"]) == 1
    assert env.current_stream.handle_reads == 1
    for owner in env.owners:
        assert owner.kwargs["device_index"] == env.device
        assert owner.kwargs["stream"] == env.current_stream.handle
        for peer, link in owner.links.items():
            wanted = "hccs" if env.device // 2 == env.devices[env.members.index(peer)] // 2 else "pcie"
            assert link == wanted
    assert len(env.receipts("setup")) == 1
    receipt = json.loads(env.receipts("setup")[0].read_text())
    assert receipt["members"] == list(members)
    assert receipt["config_sha256"] == facade.identity
    assert receipt["geometries"][0]["calls"] == 0
    assert "IPC_PRIVATE" not in env.receipts("setup")[0].read_text()
    assert facade.session not in env.receipts("setup")[0].read_text()


@pytest.mark.parametrize("ranks", (1, 2, 4))
def test_recursive_doubling_owners_use_same_eager_facade_contract(tmp_path, ranks):
    env = Environment(tmp_path, tuple(range(ranks)), algorithm="recursive-doubling")
    facade = env.create()
    assert all(owner.schedule.algorithm == "recursive-doubling" for owner in env.owners)
    result = facade.all_reduce(FakeTensor(), lambda _: pytest.fail("unexpected fallback"))
    assert isinstance(result, FakeTensor)


def test_first_eligible_call_claims_actual_stream_and_hot_path_never_reads_native_handle_or_syncs(env):
    initial = env.current_stream
    facade = env.create()
    env.current_stream = FakeStream(20, 9100)
    claimed = env.current_stream
    before = len(env.gathers)
    for _ in range(10):
        facade.all_reduce(FakeTensor(), lambda _: pytest.fail("unexpected fallback"))
    assert initial.handle_reads == 1 and claimed.handle_reads == 1
    assert all(owner.stream.value == 9100 for owner in env.owners)
    assert facade.claimed_stream is claimed
    assert env.owners[0].calls == 10 and env.owners[1].calls == 0
    assert len(env.gathers) == before
    assert not any(event[0] == "torch-synchronize" for event in env.events)
    assert all(call[1:] == (env.owners[0].plan.value, 30000) for call in env.op_calls)
    # Torch may return a fresh Python wrapper for the same actual stream.
    env.current_stream = FakeStream(20, 9999)
    facade.all_reduce(FakeTensor("float32", 64), lambda _: pytest.fail("unexpected fallback"))
    assert env.current_stream.handle_reads == 0
    assert env.owners[1].calls == 1


@pytest.mark.parametrize(
    "tensor", (FakeTensor(elements=2048), FakeTensor(dtype="bfloat16"), FakeTensor(contiguous=False), object())
)
def test_unsupported_prefill_geometry_falls_back_before_custom_admission(env, tensor):
    facade = env.create()
    env.current_stream = FakeStream(30, 9300)
    called = []
    assert facade.all_reduce(tensor, lambda value: called.append(value) or "fallback") == "fallback"
    assert called == [tensor] and not env.op_calls
    assert facade.claimed_stream_id is None and env.current_stream.handle_reads == 0
    assert all(owner.calls == 0 for owner in env.owners)
    assert sum(facade.fallbacks.values()) == 1


@pytest.mark.parametrize("fault", ("stream", "device", "input-device", "cpu-input", "capture", "process", "thread"))
def test_unsafe_context_change_fails_closed_without_hccl_fallback(env, fault):
    facade = env.create()
    facade.all_reduce(FakeTensor(), lambda _: pytest.fail("unexpected fallback"))
    tensor = FakeTensor(elements=2048)
    if fault == "stream":
        env.current_stream = FakeStream(90, 9900)
    elif fault == "device":
        env.device = 1
    elif fault == "input-device":
        tensor.device.index = 1
    elif fault == "cpu-input":
        tensor.device.type = "cpu"
    elif fault == "capture":
        env.capturing = True
    elif fault == "process":
        facade.pid = -1
    else:
        facade.thread = -1
    with pytest.raises(RuntimeError):
        facade.all_reduce(tensor, lambda _: pytest.fail("unsafe fallback"))
    assert facade.state == "quarantined"
    assert len(env.op_calls) == 1
    assert all(owner.state == "quarantined" for owner in env.owners)
    assert not any(event[0] == "close" for event in env.events)


def test_native_error_never_selects_fallback_and_retains_every_owner(env):
    facade = env.create()
    env.op_error = True
    with pytest.raises(RuntimeError, match="partial"):
        facade.all_reduce(FakeTensor(), lambda _: pytest.fail("unsafe native-error fallback"))
    assert facade.owners and all(owner.state == "quarantined" for owner in env.owners)
    with pytest.raises(RuntimeError):
        facade.all_reduce(FakeTensor(), lambda _: pytest.fail("fallback after quarantine"))
    assert len(env.op_calls) == 1
    receipt = json.loads(env.receipts("quarantine")[0].read_text())
    assert receipt["error_type"] == "RuntimeError" and receipt["automatic_cleanup"] is False


@pytest.mark.parametrize("ranks", (1, 2, 3, 4))
def test_explicit_close_drains_torch_before_all_rank_import_and_export_release(tmp_path, ranks):
    env = Environment(tmp_path, tuple(range(ranks)))
    facade = env.create()
    facade.all_reduce(FakeTensor(), lambda _: pytest.fail("unexpected fallback"))
    facade.all_reduce(FakeTensor(elements=2048), lambda _: "fallback")
    before = len(env.events)
    facade.close(graphs_retired=True)
    events = env.events[before:]
    assert events[0] == ("torch-synchronize", env.device)
    assert [event[0] for event in events[1:]] == ["drain", "gather", "release", "gather", "close"] * 2
    assert facade.state == "closed" and all(owner.state == "closed" for owner in env.owners)
    receipt = json.loads(env.receipts("shutdown")[0].read_text())
    assert receipt["ownership_released"] is True and receipt["graphs_retired"] is True
    assert receipt["geometries"][0]["calls"] == 1
    assert receipt["fallback_schemas"] == [dict(dtype="float16", elements=2048, contiguous=True, calls=1)]
    assert "IPC_PRIVATE" not in env.receipts("shutdown")[0].read_text()
    assert "123456" not in env.receipts("shutdown")[0].read_text()


def test_close_requires_explicit_handler_retirement_and_has_no_automatic_cleanup(env):
    facade = env.create()
    assert "__del__" not in TorchCollective.__dict__
    for value in (False, None, 1):
        with pytest.raises(ValueError, match="retire"):
            facade.close(graphs_retired=value)
    assert facade.state == "active"
    assert not any(event[0] in ("torch-synchronize", "close") for event in env.events)


def test_delayed_torch_queue_failure_retains_native_owners(env):
    facade = env.create()
    env.sync_error = True
    with pytest.raises(RuntimeError, match="queue"):
        facade.close(graphs_retired=True)
    assert facade.state == "quarantined"
    assert not any(event[0] in ("drain", "release", "close") for event in env.events)


def test_partial_setup_error_attaches_strong_facade_before_propagating(env):
    env.prepare_error = True
    with pytest.raises(CollectiveSetupError) as failure:
        env.create()
    facade = failure.value.collective_owner
    assert facade is failure.value.collective
    assert len(facade.owners) == 1 and list(facade.owners.values())[0] is env.owners[0]
    assert facade.state == "quarantined" and env.owners[0].state == "quarantined"
    assert facade.torch is env.torch
    assert not any(event[0] == "close" for event in env.events)
    assert "IPC_PRIVATE" not in env.receipts("quarantine")[0].read_text()


@pytest.mark.parametrize("mismatch", ("identity", "phase", "rank", "members", "device", "nonce", "root-nonce"))
def test_all_rank_bootstrap_must_agree_before_loading_or_allocating(env, mismatch):
    def mutate(phase, packets):
        if phase != "bootstrap":
            return
        if mismatch == "identity":
            packets[1]["config_sha256"] = "0" * 64
        elif mismatch == "phase":
            packets[1]["phase"] = "other"
        elif mismatch == "rank":
            packets[0]["rank"] = False
        elif mismatch == "members":
            packets[1]["payload"]["members"] = [1, 0]
        elif mismatch == "device":
            packets[1]["payload"]["device"] = False
        elif mismatch == "nonce":
            packets[1]["payload"]["nonce"] = "b" * 64
        else:
            packets[0]["payload"]["nonce"] = "not a nonce"

    env.mutate_gather = mutate
    with pytest.raises(CollectiveSetupError) as failure:
        env.create()
    assert not env.owners and not any(event[0] == "load" for event in env.events)
    assert failure.value.collective_owner.state == "quarantined"


@pytest.mark.parametrize("field", ("group_size", "group_rank", "world_rank"))
def test_local_cpu_group_mapping_must_match_configuration(env, field):
    setattr(env, field, 999)
    with pytest.raises(CollectiveSetupError):
        env.create()
    assert not env.gathers and not env.owners


def test_shutdown_call_count_disagreement_retains_exports(env):
    facade = env.create()

    def mutate(phase, packets):
        if phase.startswith("drain:"):
            packets[1]["payload"]["calls"] += 1

    env.mutate_gather = mutate
    with pytest.raises(ValueError, match="call sequence"):
        facade.close(graphs_retired=True)
    assert facade.state == "quarantined"
    assert not any(event[0] in ("release", "close") for event in env.events)


@pytest.mark.parametrize("phase_name", ("pids:", "exports:", "active:"))
def test_partial_geometry_setup_failure_quarantines_attached_owner(env, phase_name):
    def mutate(phase, packets):
        if not phase.startswith(phase_name):
            return
        if phase_name == "pids:":
            packets[1]["payload"] = True
        elif phase_name == "exports:":
            packets[1]["payload"] = {}
        else:
            packets[1]["payload"] = "prepared"

    env.mutate_gather = mutate
    with pytest.raises(CollectiveSetupError) as failure:
        env.create()
    assert len(failure.value.collective_owner.owners) == 1
    assert env.owners[0].state == "quarantined"
    assert not any(event[0] == "close" for event in env.events)


def test_all_rank_import_closure_disagreement_prevents_export_release(env):
    facade = env.create()

    def mutate(phase, packets):
        if phase.startswith("imports-closed:"):
            packets[1]["payload"]["imports_closed"] = 1

    env.mutate_gather = mutate
    with pytest.raises(ValueError):
        facade.close(graphs_retired=True)
    assert facade.state == "quarantined"
    assert not any(event[0] == "close" for event in env.events)


@pytest.mark.parametrize("capturing,visible", ((True, 2), (False, 1)))
def test_capture_or_incomplete_single_host_visibility_rejected_before_setup(env, capturing, visible):
    env.capturing = capturing
    env.torch.npu.device_count = lambda: visible
    with pytest.raises(CollectiveSetupError):
        env.create()
    assert not env.gathers and not env.owners


def test_missing_capture_query_fails_closed_before_load(env):
    del env.torch.npu.is_current_stream_capturing
    with pytest.raises(CollectiveSetupError):
        env.create()
    assert not env.gathers and not env.owners


@pytest.mark.parametrize("field", ("torch_library", "acl_sha256", "geometries", "algorithm", "link_groups"))
def test_missing_required_configuration_field_rejected_before_side_effects(env, field):
    del env.config[field]
    with pytest.raises(ValueError, match="configuration fields"):
        env.create()
    assert not env.events


@pytest.mark.parametrize("members,rank,device", (((0, 0), 0, 0), ([0, 1], 0, 0), ((0, 1), False, 0), ((0, 1), 0, True)))
def test_exact_member_rank_and_physical_device_types(env, members, rank, device):
    with pytest.raises(ValueError):
        validate_config(env.config, members, rank, device)


@pytest.mark.parametrize(
    "field,value",
    (
        ("schema_version", True),
        ("schema_version", 2),
        ("algorithm", "ring"),
        ("algorithm", []),
        ("physical_devices", [0, 0]),
        ("physical_devices", [False, 1]),
        ("physical_devices", [0]),
        ("physical_devices", [[0], 1]),
        ("link_groups", []),
        ("link_groups", [[0], [0, 1]]),
        ("link_groups", [[0]]),
        ("link_groups", [[False, 1]]),
        ("link_groups", [[0, 1], []]),
        ("geometries", []),
        ("geometries", [dict(dtype="bfloat16", elements=32)]),
        ("geometries", [dict(dtype="float16", elements=True)]),
        ("geometries", [dict(dtype="float16", elements=0)]),
        ("geometries", [dict(dtype="float16", elements=16 * 1024 * 1024 + 1)]),
        ("geometries", [dict(dtype="float16", elements=32), dict(dtype="float16", elements=32)]),
        ("geometries", [dict(dtype="float16", elements=32, extra=True)]),
        ("output_root", "relative"),
        ("output_root", None),
        ("unknown", True),
    ),
)
def test_malformed_configuration_is_rejected_before_load_or_allocation(env, field, value):
    env.config[field] = value
    with pytest.raises(ValueError):
        env.create()
    assert not env.gathers and not env.owners and not env.events


@pytest.mark.parametrize("name", ("acl", "native", "torch"))
@pytest.mark.parametrize("fault", ("missing", "drift", "relative", "path-type", "sha-type"))
def test_all_pinned_libraries_are_validated_before_bootstrap(env, name, fault):
    path = Path(env.config[name + "_library"])
    if fault == "missing":
        path.unlink()
    elif fault == "drift":
        path.write_bytes(b"changed")
    elif fault == "relative":
        env.config[name + "_library"] = path.name
    elif fault == "path-type":
        env.config[name + "_library"] = [str(path)]
    else:
        env.config[name + "_sha256"] = None
    with pytest.raises(ValueError):
        env.create()
    assert not env.events


def test_torch_dependency_must_share_pinned_private_native_directory(env, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    path = other / "torch.so"
    path.write_bytes(b"different private directory")
    env.config.update(torch_library=str(path), torch_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="RPATH"):
        env.create()


def test_recursive_doubling_rejects_three_ranks_before_any_setup(tmp_path):
    env = Environment(tmp_path, (0, 1, 2), algorithm="recursive-doubling")
    with pytest.raises(ValueError, match="power-of-two"):
        env.create()
    assert not env.events


def test_pin_mutation_during_bootstrap_is_rechecked_before_loading(env):
    def mutate(phase, packets):
        if phase == "bootstrap":
            Path(env.config["torch_library"]).write_bytes(b"drifted during agreement")

    env.mutate_gather = mutate
    with pytest.raises(CollectiveSetupError) as failure:
        env.create()
    assert isinstance(failure.value.__cause__, ValueError)
    assert not env.owners and not any(event[0] == "load" for event in env.events)


def test_group_identity_is_stable_and_receipt_names_separate_tp_groups(env, tmp_path):
    original = copy.deepcopy(env.config)
    normalized = validate_config(env.config, env.members, env.rank, env.device)
    assert normalized == env.config and env.config == original
    first = env.create()
    other = Environment(tmp_path, (2, 3))
    second = other.create()
    assert first.identity != second.identity
    assert len(env.receipts("setup")) == 2


def test_import_does_not_load_torch_or_native_libraries():
    root = Path(__file__).resolve().parents[3]
    code = (
        "import ctypes,sys; "
        "ctypes.CDLL=lambda *a,**k: (_ for _ in ()).throw(AssertionError('native load')); "
        "import tools.sidecard.npu_collective_torch; "
        "assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True, capture_output=True, text=True)


@pytest.mark.parametrize("mode", (None, True, "auto", "GRAPH", 1))
def test_invalid_execution_mode_rejected_before_loading(env, mode):
    env.config["execution_mode"] = mode
    with pytest.raises(ValueError, match="execution_mode"):
        env.create()
    assert not env.events


def test_graph_mode_rejects_host_alternating_recursive_doubling(env):
    env.config.update(execution_mode="graph", algorithm="recursive-doubling")
    with pytest.raises(ValueError, match="fixed-address"):
        env.create()
    assert not env.events


def test_graph_capture_uses_actual_stream_without_flushing_or_replacing_eager_owner(env):
    env.config.update(execution_mode="graph", algorithm="recursive-doubling-fixed", graph_replay_tracking=True)
    facade = env.create()
    eager = env.current_stream
    facade.all_reduce(FakeTensor(), lambda _: pytest.fail("fallback"))
    graph = SimpleNamespace(reset=lambda: None, replay=lambda: None)
    facade.graph_observer.capture_begin(graph)
    env.capturing = True
    env.current_stream = FakeStream(77, 9777)
    facade.all_reduce(FakeTensor(), lambda _: pytest.fail("fallback"))
    assert env.current_stream.handle_reads == 0
    assert facade.claimed_stream is eager
    assert all(owner.stream.value == eager.handle for owner in env.owners)
    env.capturing = False
    env.current_stream = eager
    facade.graph_observer.capture_end(graph)
    for _ in range(3):
        facade.graph_observer.before_replay(graph)
        facade.graph_observer.replayed(graph)
    assert env.owners[0].calls == 5
    facade.graph_observer.retire(lambda: env.synchronize(env.device))
    facade.close(graphs_retired=True)
    assert facade.state == "closed"


def test_unregistered_capture_quarantines_without_native_submission(env):
    env.config.update(execution_mode="graph", algorithm="recursive-doubling-fixed", graph_replay_tracking=True)
    facade = env.create()
    facade.all_reduce(FakeTensor(), lambda _: None)
    env.capturing = True
    with pytest.raises(RuntimeError):
        facade.all_reduce(FakeTensor(), lambda _: None)
    assert len(env.op_calls) == 1 and facade.state == "quarantined"


def test_graph_close_requires_observer_retirement_even_with_caller_assertion(env):
    env.config.update(execution_mode="graph", algorithm="recursive-doubling-fixed", graph_replay_tracking=True)
    facade = env.create()
    with pytest.raises(RuntimeError, match="observer retirement"):
        facade.close(graphs_retired=True)
    assert facade.state == "quarantined"
    assert not any(e[0] == "drain" for e in env.events)


def test_graph_capture_before_eager_claim_never_reads_native_stream(env):
    env.config.update(execution_mode="graph", algorithm="recursive-doubling-fixed", graph_replay_tracking=True)
    facade = env.create()
    facade.graph_observer.capture_begin(SimpleNamespace(reset=lambda: None, replay=lambda: None))
    env.capturing = True
    env.current_stream = FakeStream(99, 9999)
    with pytest.raises(RuntimeError, match="eager warmup"):
        facade.all_reduce(FakeTensor(), lambda _: None)
    assert env.current_stream.handle_reads == 0 and not env.op_calls


@pytest.mark.parametrize("tracking", (None, 1, "false", []))
def test_diagnostic_replay_tracking_requires_exact_boolean(env, tracking):
    env.config.update(execution_mode="graph", graph_replay_tracking=tracking)
    with pytest.raises(ValueError, match="graph_replay_tracking"):
        env.create()
    assert not env.events


def test_graph_mode_defaults_to_unobserved_native_replay(env):
    env.config.update(execution_mode="graph", algorithm="recursive-doubling-fixed")
    facade = env.create()
    assert facade.graph_observer.track_replays is False
    facade.graph_observer.retire(lambda: env.synchronize(env.device))
    facade.close(graphs_retired=True)
    assert facade.state == "closed"
