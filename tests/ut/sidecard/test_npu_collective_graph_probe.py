# SPDX-License-Identifier: Apache-2.0
"""CPU graph recorder with the real observer; no accelerator imports/calls."""

import contextlib
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.sidecard import npu_collective_graph_probe as probe
from tools.sidecard.npu_collective_graph import CollectiveGraphObserver


@pytest.fixture
def config(tmp_path):
    result = dict(
        schema_version=1,
        algorithm="recursive-doubling-fixed",
        execution_mode="graph",
        geometries=[dict(dtype="float16", elements=4), dict(dtype="float32", elements=7)],
        physical_devices=[0, 1, 2, 3],
        link_groups=[[0, 1], [2, 3]],
        output_root=str(tmp_path / "receipts"),
    )
    for name in ("acl", "native", "torch"):
        path = tmp_path / (name + ".so")
        path.write_bytes(("mock pinned " + name).encode())
        result[name + "_library"] = str(path)
        result[name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def test_six_rank_binomial_graph_config(config):
    config.update(algorithm="binomial", physical_devices=list(range(6)), link_groups=[[0, 1], [2, 3], [4, 5]])
    assert probe.validate(config) == config
    config["algorithm"] = "recursive-doubling-fixed"
    with pytest.raises(ValueError, match="power"):
        probe.validate(config)
    config["algorithm"] = "recursive-doubling"
    with pytest.raises(ValueError, match="fixed-address"):
        probe.validate(config)


class Tensor:
    def __init__(self, env, raw, dtype, case, device="cpu"):
        self.env, self.raw, self.dtype, self.case, self.device = env, raw, dtype, case, device
        self.pointer = env.pointer
        env.pointer += len(raw) + 64

    def clone(self):
        output = Tensor(self.env, self.raw, self.dtype, self.case, self.device)

        def copy():
            output.raw, output.case = self.raw, self.case

        self.env.record("clone", copy)
        return output

    def to(self, *, device):
        return Tensor(self.env, self.raw, self.dtype, self.case, device)

    def copy_(self, other):
        assert self.env.capturing is None
        self.raw, self.case = other.raw, other.case
        self.env.input_updates += 1
        return self

    def zero_(self):
        assert self.env.capturing is None
        self.raw = b"\x00" * len(self.raw)
        self.env.output_invalidations += 1
        return self

    def data_ptr(self):
        return self.pointer

    def detach(self):
        return self

    def cpu(self):
        assert self.env.capturing is None
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self

    def tobytes(self):
        return self.raw


class Graph:
    def __init__(self, env):
        self.env, self.tasks, self.kinds = env, [], []
        self.replays, self.resets = 0, 0
        env.graphs.append(self)

    def replay(self):
        assert self.resets == 0 and self.env.capturing is None
        if self.env.fail_replay:
            raise RuntimeError("private replay failure")
        self.replays += 1
        self.env.replay_order.append(self.env.graphs.index(self))
        target = self.env.latest_graphs[(self.mode, self.geometry)] if self.env.rebind_old_graphs else self
        for task in target.tasks:
            task()

    def reset(self):
        assert self.env.capturing is None and self.resets == 0
        if self.env.fail_reset:
            raise RuntimeError("private graph reset failure")
        self.resets += 1
        self.tasks.clear()
        self.env.events.append("graph-reset")


class Stream:
    def __init__(self, env, device, *, initial=False):
        self.env, self.device, self.initial = env, device, initial

    def wait_stream(self, other):
        assert other is self.env.default_stream and not self.initial
        self.env.events.append("wait-default-stream")


class Collective:
    def __init__(self, env):
        self.env, self.state, self.fallbacks = env, "active", {}
        self.owners = {
            (geometry["dtype"], geometry["elements"]): SimpleNamespace(calls=0, state="active")
            for geometry in env.config["geometries"]
        }
        for owner in self.owners.values():
            owner.leases = []

            def acquire(owner=owner):
                lease = object()
                owner.leases.append(lease)
                return lease

            owner.acquire_graph_lease = acquire
            owner.release_graph_lease = owner.leases.remove
        self.graph_observer = CollectiveGraphObserver(
            self, track_replays=env.config.get("graph_replay_tracking", False)
        )
        self.claimed_stream = None

    def _context(self):
        assert self.env.capturing is None

    def validate_graph_replay(self):
        assert self.env.capturing is None and self.env.current_stream is self.claimed_stream

    def all_reduce(self, source, fallback):
        assert fallback is probe.eager._fallback and self.state == "active"
        assert self.env.current_stream is not self.env.default_stream
        if self.claimed_stream is None:
            assert self.env.capturing is None, "capture preceded eager warmup"
            self.claimed_stream = self.env.current_stream
        assert self.claimed_stream is self.env.current_stream
        if self.env.capturing is not None:
            assert self.graph_observer.require_capture() is self.env.capturing
            if self.env.fail_capture:
                raise RuntimeError("private capture failure")
        elements = len(source.raw) // (2 if source.dtype == "float16" else 4)
        self.owners[(source.dtype, elements)].calls += 1
        output = Tensor(self.env, b"\x00" * len(source.raw), source.dtype, source.case, source.device)
        self.env.reduce(source, output, "custom")
        return output

    def close(self, *, graphs_retired):
        assert graphs_retired is True and self.graph_observer.retired
        assert all(graph.resets == 1 for graph in self.env.graphs)
        self.env.events.append("facade-close")
        self.state = "closed"


class Environment:
    def __init__(self, config, rank):
        self.config, self.rank, self.events, self.graphs = config, rank, [], []
        self.pointer, self.input_updates, self.capturing, self.facade = 4096, 0, None, None
        self.output_invalidations, self.replay_order, self.latest_graphs = 0, [], {}
        self.rebind_old_graphs = False
        self.fail_capture, self.fail_replay, self.fail_reset, self.corrupt = False, False, False, False
        self.default_stream = Stream(self, config["physical_devices"][rank], initial=True)
        self.current_stream = self.default_stream
        self.cpu_group, self.hccl_group = object(), object()
        self.torch = SimpleNamespace(
            float16="float16",
            float32="float32",
            frombuffer=self.frombuffer,
            set_num_threads=lambda n: self.events.append(("threads", n)),
            npu=SimpleNamespace(
                set_device=lambda n: self.events.append(("device", n)),
                synchronize=self.synchronize,
                current_stream=lambda n: self.current_stream,
                Stream=lambda *, device: Stream(self, device),
                NPUGraph=lambda: Graph(self),
                stream=self.stream,
                graph=self.capture,
            ),
        )
        self.dist = SimpleNamespace(
            group=SimpleNamespace(WORLD=self.cpu_group),
            ReduceOp=SimpleNamespace(SUM="sum"),
            init_process_group=self.init,
            new_group=self.new_group,
            all_reduce=self.all_reduce,
            barrier=lambda *, group: self.events.append("cpu-barrier"),
            destroy_process_group=self.destroy,
        )

    def frombuffer(self, raw, *, dtype):
        first = struct.unpack_from("<e" if dtype == "float16" else "<f", raw)[0]
        case = round((first + 15 - self.rank / 4) * 8)
        return Tensor(self, bytes(raw), dtype, case)

    def init(self, backend, **kwargs):
        assert backend == "gloo" and kwargs["rank"] == self.rank
        assert kwargs["world_size"] == len(self.config["physical_devices"])
        assert kwargs["init_method"].startswith("file://")
        self.events.append("gloo-init")

    def new_group(self, **kwargs):
        assert kwargs["backend"] == "hccl" and kwargs["ranks"] == list(range(len(self.config["physical_devices"])))
        return self.hccl_group

    def create(self, config, members, rank, group, device, **kwargs):
        assert config == self.config and rank == self.rank and group is self.cpu_group
        assert members == tuple(range(len(config["physical_devices"]))) and device == config["physical_devices"][rank]
        assert kwargs["torch"] is self.torch and kwargs["dist"] is self.dist
        self.facade = Collective(self)
        return self.facade

    @contextlib.contextmanager
    def stream(self, stream):
        previous = self.current_stream
        self.current_stream = stream
        try:
            yield
        finally:
            self.current_stream = previous

    @contextlib.contextmanager
    def capture(self, graph, *, stream):
        assert self.capturing is None and self.current_stream is stream and stream is not self.default_stream
        self.capturing = graph
        try:
            yield
        finally:
            self.capturing = None
            if hasattr(graph, "geometry"):
                graph.mode = "custom" if "custom" in graph.kinds else "hccl"
                self.latest_graphs[(graph.mode, graph.geometry)] = graph

    def record(self, kind, task):
        if self.capturing is None:
            task()
        else:
            self.capturing.tasks.append(task)
            self.capturing.kinds.append(kind)

    def reduce(self, source, output, kind):
        if self.capturing is not None:
            self.capturing.geometry = source.dtype, len(source.raw)

        def task():
            elements = len(source.raw) // (2 if source.dtype == "float16" else 4)
            output.raw = probe.expected(elements, len(self.config["physical_devices"]), source.case, source.dtype)
            output.case = source.case
            if self.corrupt:
                output.raw = b"!" * len(output.raw)

        self.record(kind, task)

    def all_reduce(self, tensor, *, op, group):
        assert op == "sum" and group is self.hccl_group
        self.reduce(tensor, tensor, "hccl")

    def synchronize(self, device):
        assert self.capturing is None and device == self.config["physical_devices"][self.rank]
        self.events.append("synchronize")

    def destroy(self, group):
        assert "facade-close" in self.events
        self.events.append(("destroy", group))


def exercise(config, tmp_path, rank, monkeypatch, *, env=None, retained=None):
    env = Environment(config, rank) if env is None else env
    phases = []
    monkeypatch.setattr(probe, "gather", lambda connection, phase, payload: phases.append(phase))
    retained = (
        dict(facade=None, observer=None, groups=[], tensors=[], graphs=[], streams=[]) if retained is None else retained
    )
    result = probe.exercise(config, tmp_path, rank, None, retained, torch=env.torch, dist=env.dist, create=env.create)
    return result, env, retained, phases


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
@pytest.mark.parametrize("tracking", [False, True])
def test_real_observer_captures_changed_independent_sums_and_retires_before_close(
    config, tmp_path, monkeypatch, rank, tracking
):
    config["graph_replay_tracking"] = tracking
    result, env, retained, phases = exercise(config, tmp_path, rank, monkeypatch)
    assert result["passed"] is result["graphs_retired"] is result["ownership_released"] is True
    assert result["graph_observer"] == probe.observer_receipt(config)
    assert len(result["captures"]) == 4 and len(retained["graphs"]) == len(env.graphs) == 8
    assert len(retained["streams"]) == 1 and retained["groups"] == []
    assert all(graph.resets == 1 and graph.replays == 217 for graph in env.graphs)
    assert result["final_recapture_history"] == probe.final_history(config)
    assert env.replay_order[-16:] == [6, 7, 4, 5, 2, 3, 0, 1, 0, 1, 2, 3, 4, 5, 6, 7]
    assert env.output_invalidations == 32
    assert [graph.kinds for graph in env.graphs] == [
        ["custom"],
        ["clone", "hccl"],
        ["custom"] * 3,
        ["clone", "hccl"] * 3,
    ] * len(config["geometries"])
    assert env.input_updates == 5 * 2 * (1 + 3) * len(config["geometries"])
    assert phases[-3:] == ["graphs-completed", "all-graphs-retired", "facade-closed"]
    assert max(index for index, event in enumerate(env.events) if event == "graph-reset") < env.events.index(
        "facade-close"
    )
    for capture in result["captures"]:
        assert len(capture["changed_replays"]) == len(capture["eager_interleaves"]) == 5
        assert len(capture["timings"]) == 14
        assert all(sample["count"] == 30 * capture["operations"] for sample in capture["timings"])
        assert all(history["custom"] == history["hccl"] for history in capture["changed_replays"])


def test_final_sweep_detects_older_executor_rebound_to_later_capture_addresses(config, tmp_path, monkeypatch):
    env = Environment(config, 0)
    env.rebind_old_graphs = True
    retained = dict(facade=None, observer=None, groups=[], tensors=[], graphs=[], streams=[])
    with pytest.raises(AssertionError, match="complete Torch result differs"):
        exercise(config, tmp_path, 0, monkeypatch, env=env, retained=retained)
    # Every graph passed its initial changed histories and timing batches. The
    # newly added post-capture sweep exposes mutable executor-address reuse.
    assert len(env.graphs) == 8 and min(graph.replays for graph in env.graphs) >= 215
    assert env.output_invalidations > 0 and "facade-close" not in env.events
    assert all(graph.resets == 0 for graph in retained["graphs"])


@pytest.mark.parametrize("flag", ["fail_capture", "fail_replay", "fail_reset", "corrupt"])
def test_graph_failure_preserves_graph_stream_tensor_and_facade_owners(config, tmp_path, monkeypatch, flag):
    env = Environment(config, 0)
    setattr(env, flag, True)
    retained = dict(facade=None, observer=None, groups=[], tensors=[], graphs=[], streams=[])
    with pytest.raises((RuntimeError, AssertionError)):
        exercise(config, tmp_path, 0, monkeypatch, env=env, retained=retained)
    assert retained["facade"] is env.facade and retained["tensors"] and len(retained["groups"]) == 2
    assert retained["streams"] and "facade-close" not in env.events
    if flag != "corrupt":
        assert retained["graphs"]
        assert env.facade.state == "quarantined"


def test_public_graph_reports_and_cross_rank_timing_are_authenticated(config, tmp_path, monkeypatch):
    ranks = [exercise(config, tmp_path, rank, monkeypatch)[0] for rank in range(4)]
    comparison = probe.comparisons(config, ranks)
    assert len(comparison) == len(config["geometries"]) * 2
    assert all(item["comparison"]["clock"] == "shared-host CLOCK_MONOTONIC" for item in comparison)
    assert all(item["comparison"]["pure_transit_measured"] is False for item in comparison)


@pytest.fixture
def rank_reports(config, tmp_path, monkeypatch):
    return [exercise(config, tmp_path, rank, monkeypatch)[0] for rank in range(4)]


@pytest.mark.parametrize(
    "mutation", ["rank", "device", "retired", "observer", "counts", "capture", "history", "eager", "timing"]
)
def test_missing_graph_or_numerical_coverage_cannot_pass(config, rank_reports, mutation):
    report = rank_reports[0]
    if mutation in ("rank", "device", "retired", "observer", "counts"):
        report[
            dict(
                rank="rank",
                device="physical_device",
                retired="graphs_retired",
                observer="graph_observer",
                counts="graph_counts",
            )[mutation]
        ] = "changed"
    elif mutation == "capture":
        report["captures"].pop()
    else:
        report["captures"][0][
            dict(history="changed_replays", eager="eager_interleaves", timing="timings")[mutation]
        ].pop()
    with pytest.raises(ValueError):
        probe.comparisons(config, rank_reports)


@pytest.mark.parametrize("mutation", ["missing", "order", "hash", "integer-type"])
def test_controller_requires_exact_final_old_graph_replay_history(config, rank_reports, mutation):
    report = rank_reports[0]
    history = report["final_recapture_history"]
    if mutation == "missing":
        del report["final_recapture_history"]
    elif mutation == "order":
        history[0], history[1] = history[1], history[0]
    elif mutation == "hash":
        history[0]["custom"][0] = "0" * 64
    else:
        history[0]["capture_index"] = float(history[0]["capture_index"])
    with pytest.raises(ValueError):
        probe.comparisons(config, rank_reports)


class Control:
    def __init__(self, fd):
        self.fd, self.closed = fd, False

    def fileno(self):
        return self.fd

    def settimeout(self, value):
        assert value == probe.CONTROL_TIMEOUT_SECONDS

    def close(self):
        self.closed = True

    def sendall(self, raw):
        assert json.loads(raw[4:])["phase"] == "done"


def mock_controller(rank_reports, output, monkeypatch, *, fail_rank=None, bad_done=False):
    controls, processes, commands = [], [], []

    def socketpair():
        pair = [Control(100 + len(controls) + index) for index in range(2)]
        controls.extend(pair)
        return pair

    def popen(command, **kwargs):
        rank = int(command[command.index("--worker") + 1])
        commands.append(command)
        assert command[:3] == [sys.executable, "-m", "tools.sidecard.npu_collective_graph_probe"]
        assert kwargs["pass_fds"] == (controls[-1].fd,)
        if rank == fail_rank:
            raise OSError("mock graph spawn failure")
        (output / f"rank{rank}.json").write_text(json.dumps(rank_reports[rank]))
        process = SimpleNamespace(pid=7000 + rank, waited=False)

        def wait(timeout):
            assert timeout == 10
            process.waited = True
            return 0

        process.wait = wait
        processes.append(process)
        return process

    def receive(connection):
        rank = controls.index(connection) // 2
        checksum = hashlib.sha256((output / f"rank{rank}.json").read_bytes()).hexdigest()
        return dict(
            phase="done", payload=dict(rank=rank, passed=True, result_sha256="0" * 64 if bad_done else checksum)
        )

    monkeypatch.setattr(probe.socket, "socketpair", socketpair)
    monkeypatch.setattr(probe.subprocess, "Popen", popen)
    monkeypatch.setattr(probe, "receive", receive)
    return controls, processes, commands


def test_graph_controller_uses_new_private_directory_and_matching_device_count(
    config, rank_reports, tmp_path, monkeypatch
):
    output = tmp_path / "gate"
    controls, processes, commands = mock_controller(rank_reports, output, monkeypatch)
    result = probe.run(config, output)
    assert result["passed"] is result["graphs_retired"] is True
    assert len(commands) == 4 and all(process.waited for process in processes)
    assert all(control.closed for control in controls) and not (output / "quarantine.json").exists()


@pytest.mark.parametrize("fail_rank,bad_done", [(0, False), (2, False), (None, True)])
def test_graph_controller_failure_retains_pids_without_process_kill(
    config, rank_reports, tmp_path, monkeypatch, fail_rank, bad_done
):
    output = tmp_path / "gate"
    controls, processes, _ = mock_controller(rank_reports, output, monkeypatch, fail_rank=fail_rank, bad_done=bad_done)
    with pytest.raises((ValueError, OSError)):
        probe.run(config, output)
    assert json.loads((output / "quarantine.json").read_text()) == dict(
        pids=[process.pid for process in processes], owners_retained=True, automatic_kill=False
    )
    assert all(control.closed for control in controls) and all(not process.waited for process in processes)


@pytest.mark.parametrize(
    "field,value",
    [("execution_mode", "eager"), ("algorithm", "recursive-doubling"), ("torch_sha256", "0" * 64), ("extra", True)],
)
def test_graph_configuration_rejected_before_side_effects(config, tmp_path, monkeypatch, field, value):
    config[field] = value
    output = tmp_path / "gate"
    monkeypatch.setattr(probe.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("invalid graph config launched"))
    with pytest.raises(ValueError):
        probe.run(config, output)
    assert not output.exists()


def test_worker_quarantines_graphs_and_private_error_details(config, tmp_path, monkeypatch, capsys):
    marker, saved = object(), []

    def broken(config, output, rank, connection, retained, **kwargs):
        retained["graphs"].append(marker)
        retained["streams"].append(marker)
        saved.append(retained)
        raise RuntimeError("private graph pointer 0xDEAD")

    class Paused(BaseException):
        pass

    def pause():
        assert saved[0]["graphs"] == saved[0]["streams"] == [marker]
        assert isinstance(saved[0]["error"], RuntimeError)
        raise Paused()

    monkeypatch.setattr(probe.eager, "_backend", lambda: (None, None))
    monkeypatch.setattr(probe, "exercise", broken)
    monkeypatch.setattr(probe, "send", lambda *args: None)
    monkeypatch.setattr(probe.signal, "pause", pause)
    with pytest.raises(Paused):
        probe.worker(config, tmp_path, 0, None)
    raw = (tmp_path / "rank0-failure.json").read_text()
    assert "0xDEAD" not in raw and json.loads(raw)["owners_retained"] is True
    assert "private graph pointer 0xDEAD" in capsys.readouterr().err


def test_graph_import_and_help_never_import_accelerator_modules():
    root = Path(__file__).resolve().parents[3]
    code = (
        "import sys,ctypes; ctypes.CDLL=lambda *a,**k: (_ for _ in ()).throw(AssertionError('device load')); "
        "import tools.sidecard.npu_collective_graph_probe; "
        "assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True, capture_output=True)
    result = subprocess.run(
        [sys.executable, "-m", "tools.sidecard.npu_collective_graph_probe", "--help"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--config" in result.stdout and "--worker" in result.stdout


def test_unobserved_probe_replay_never_calls_python_tracking():
    calls = []

    def forbidden(*args):
        pytest.fail("unobserved native replay invoked Python tracking")

    observer = SimpleNamespace(track_replays=False, before_replay=forbidden, replayed=forbidden)
    graph = SimpleNamespace(replay=lambda: calls.append("native-replay"))
    probe._replay(graph, observer)
    assert calls == ["native-replay"]
