# SPDX-License-Identifier: Apache-2.0
"""Injected Torch/controller tests; never import accelerator runtimes."""

import hashlib
import json
import subprocess
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.sidecard import npu_collective_torch_probe as probe


@pytest.fixture
def config(tmp_path):
    result = dict(
        schema_version=1,
        algorithm="recursive-doubling",
        geometries=[dict(dtype="float16", elements=4), dict(dtype="float32", elements=7)],
        physical_devices=[0, 1, 2, 3],
        link_groups=[[0, 1], [2, 3]],
        output_root=str(tmp_path / "receipts"),
    )
    for name in ("acl", "native", "torch"):
        path = tmp_path / (name + ".so")
        path.write_bytes(("pinned mock " + name).encode())
        result[name + "_library"] = str(path)
        result[name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class Tensor:
    def __init__(self, environment, raw, dtype, case, device="cpu"):
        self.env, self.raw, self.dtype, self.case, self.device = environment, raw, dtype, case, device
        self.pointer = environment.pointer
        environment.pointer += len(raw) + 64

    def clone(self):
        self.env.events.append("clone")
        return Tensor(self.env, self.raw, self.dtype, self.case, self.device)

    def to(self, *, device):
        return Tensor(self.env, self.raw, self.dtype, self.case, device)

    def data_ptr(self):
        return self.pointer

    def detach(self):
        return self

    def cpu(self):
        assert not self.env.pending
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self

    def tobytes(self):
        return self.raw


class Environment:
    def __init__(self, config, rank):
        self.config, self.rank, self.events, self.pending = config, rank, [], []
        self.pointer, self.inputs, self.max_pending = 4096, 0, 0
        self.facade, self.retained = None, None
        self.fail_submit, self.fail_sync, self.corrupt = False, False, False
        self.torch = SimpleNamespace(
            float16="float16",
            float32="float32",
            frombuffer=self.frombuffer,
            set_num_threads=lambda n: self.events.append(("threads", n)),
            npu=SimpleNamespace(
                set_device=lambda n: self.events.append(("device", n)),
                synchronize=self.synchronize,
                current_stream=lambda n: ("stream", n),
            ),
        )
        self.cpu_group, self.hccl_group = object(), object()
        self.dist = SimpleNamespace(
            group=SimpleNamespace(WORLD=self.cpu_group),
            ReduceOp=SimpleNamespace(SUM="sum"),
            init_process_group=self.init,
            new_group=self.new_group,
            all_reduce=self.all_reduce,
            barrier=lambda *, group: self.events.append(("barrier", group)),
            destroy_process_group=self.destroy,
        )

    def frombuffer(self, raw, *, dtype):
        case = self.inputs % (probe.CORRECTNESS_CASES + probe.STRESS_CALLS)
        self.inputs += 1
        return Tensor(self, bytes(raw), dtype, case)

    def init(self, backend, **kwargs):
        assert backend == "gloo" and kwargs["rank"] == self.rank
        assert kwargs["world_size"] == len(self.config["physical_devices"])
        assert kwargs["init_method"].startswith("file://") and kwargs["init_method"].endswith("gloo-rendezvous")
        self.events.append("gloo-init")

    def new_group(self, **kwargs):
        assert kwargs["backend"] == "hccl" and kwargs["ranks"] == list(range(len(self.config["physical_devices"])))
        self.events.append("hccl-init")
        return self.hccl_group

    def create(self, config, members, rank, group, device, **kwargs):
        assert config == self.config and members == tuple(range(len(config["physical_devices"])))
        assert rank == self.rank and group is self.cpu_group and device == config["physical_devices"][rank]
        assert kwargs["torch"] is self.torch and kwargs["dist"] is self.dist
        self.facade = SimpleNamespace(fallbacks={}, all_reduce=self.custom, close=self.close)
        return self.facade

    def enqueue(self, source, output, mode):
        self.events.append(mode)
        if self.fail_submit:
            raise RuntimeError("private SDK address 0xBAD")
        self.pending.append((weakref.ref(source), weakref.ref(output)))
        self.max_pending = max(self.max_pending, len(self.pending))

    def custom(self, source, fallback):
        assert fallback is probe._fallback
        output = Tensor(self, b"\x00" * len(source.raw), source.dtype, source.case, source.device)
        self.enqueue(source, output, "custom")
        return output

    def all_reduce(self, tensor, *, op, group):
        assert group is self.hccl_group and op == "sum"
        self.enqueue(tensor, tensor, "hccl")

    def synchronize(self, device):
        assert device == self.config["physical_devices"][self.rank]
        self.events.append("synchronize")
        if self.fail_sync:
            raise RuntimeError("private delayed stream failure")
        for source_ref, output_ref in self.pending:
            source, output = source_ref(), output_ref()
            assert source is not None and output is not None, "queued tensor owner was dropped"
            elements = len(source.raw) // (2 if source.dtype == "float16" else 4)
            output.raw = probe.expected(elements, len(self.config["physical_devices"]), source.case, source.dtype)
            if self.corrupt:
                output.raw = b"!" * len(output.raw)
        self.pending.clear()

    def close(self, *, graphs_retired):
        assert graphs_retired is True and not self.pending
        self.events.append("facade-close")

    def destroy(self, group):
        assert self.events.index("facade-close") < len(self.events)
        self.events.append(("destroy", group))


def injected_exercise(config, output, rank, monkeypatch, **flags):
    env = Environment(config, rank)
    for name, value in flags.items():
        setattr(env, name, value)
    phases = []

    def gather(connection, phase, payload):
        assert connection == "private-control"
        phases.append(phase)
        return [payload] * len(config["physical_devices"])

    monkeypatch.setattr(probe, "gather", gather)
    retained = dict(facade=None, groups=[], tensors=[])
    env.retained = retained
    result = probe.exercise(
        config, output, rank, "private-control", retained, torch=env.torch, dist=env.dist, create=env.create
    )
    return result, env, phases


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_worker_registered_geometries_queue_distinct_live_tensors_and_close_all_ranks(
    config, tmp_path, monkeypatch, rank
):
    result, env, phases = injected_exercise(config, tmp_path, rank, monkeypatch)
    assert result["passed"] is True and result["physical_device"] == rank
    assert result["ownership_released"] is True and result["fallback_calls"] == 0
    assert env.events[:2] == [("threads", 1), ("device", rank)]
    assert env.max_pending == probe.STRESS_CALLS == probe.TIMING_BATCH
    assert env.events[-3:] == ["facade-close", ("destroy", env.hccl_group), ("destroy", env.cpu_group)]
    assert env.retained["groups"] == [] and env.retained["tensors"] == []
    assert phases[-2:] == ["completed", "facade-closed"]
    for index, geometry in enumerate(result["geometries"]):
        assert geometry["changed_input_stress"]["custom"] == geometry["changed_input_stress"]["hccl"]
        assert geometry["changed_input_stress"]["custom"]["calls"] == 30
        timing_phases = [p for p in phases if p.startswith(f"geometry-{index}-timed-")]
        assert timing_phases[:4] == [
            f"geometry-{index}-timed-0-custom",
            f"geometry-{index}-timed-0-hccl",
            f"geometry-{index}-timed-1-hccl",
            f"geometry-{index}-timed-1-custom",
        ]
        assert len(geometry["timings"]) == 14 and all(sample["count"] == 30 for sample in geometry["timings"])


@pytest.mark.parametrize("flag", ["fail_submit", "fail_sync", "corrupt"])
def test_exercise_failure_retains_owners_and_never_destroys_groups(config, tmp_path, monkeypatch, flag):
    env = Environment(config, 0)
    setattr(env, flag, True)
    monkeypatch.setattr(probe, "gather", lambda *args: [{}] * 4)
    retained = dict(facade=None, groups=[], tensors=[])
    with pytest.raises((RuntimeError, AssertionError)):
        probe.exercise(config, tmp_path, 0, None, retained, torch=env.torch, dist=env.dist, create=env.create)
    assert retained["facade"] is env.facade and len(retained["groups"]) == 2 and retained["tensors"]
    assert "facade-close" not in env.events and not any(
        isinstance(event, tuple) and event[0] == "destroy" for event in env.events
    )


def test_worker_failure_receipt_is_sanitized_but_private_traceback_and_owners_survive(
    config, tmp_path, monkeypatch, capsys
):
    marker, saved = object(), []

    def exercise(config, output, rank, connection, retained, **kwargs):
        retained["facade"] = marker
        retained["tensors"] = [marker]
        saved.append(retained)
        raise RuntimeError("private IPC key and pointer 0xABC")

    class Paused(BaseException):
        pass

    def pause():
        assert saved[0]["facade"] is marker and saved[0]["tensors"] == [marker]
        assert isinstance(saved[0]["error"], RuntimeError)
        raise Paused()

    sent = []
    monkeypatch.setattr(probe, "_backend", lambda: (None, None))
    monkeypatch.setattr(probe, "exercise", exercise)
    monkeypatch.setattr(probe, "send", lambda connection, payload: sent.append(payload))
    monkeypatch.setattr(probe.signal, "pause", pause)
    with pytest.raises(Paused):
        probe.worker(config, tmp_path, 0, None)
    raw = (tmp_path / "rank0-failure.json").read_text()
    assert "0xABC" not in raw and "IPC" not in raw
    assert sent[0]["phase"] == "error" and sent[0]["payload"]["owners_retained"] is True
    assert "private IPC key and pointer 0xABC" in capsys.readouterr().err


def reports(config):
    result = []
    for rank in range(len(config["physical_devices"])):
        geometries = []
        for geometry in config["geometries"]:
            cases = []
            for case in range(probe.CORRECTNESS_CASES):
                raw = probe.expected(geometry["elements"], len(config["physical_devices"]), case, geometry["dtype"])
                checksum = hashlib.sha256(raw).hexdigest()
                cases.append(dict(case=case, bytes=len(raw), sha256=checksum, hccl_sha256=checksum))
            stress = dict(
                calls=30,
                distinct_inputs=30,
                distinct_outputs=30,
                sha256=[
                    hashlib.sha256(
                        probe.expected(
                            geometry["elements"], len(config["physical_devices"]), 3 + case, geometry["dtype"]
                        )
                    ).hexdigest()
                    for case in range(30)
                ],
            )
            timings = [
                dict(
                    pair=pair,
                    mode=mode,
                    count=30,
                    started_ns=pair * 1_000_000 + rank * 1000,
                    submitted_ns=pair * 1_000_000 + 10_000 + rank * 1000,
                    completed_ns=pair * 1_000_000 + 100_000 + rank * 1000,
                )
                for pair in range(7)
                for mode in ("custom", "hccl")
            ]
            geometries.append(
                dict(
                    **geometry,
                    finite_cases=cases,
                    changed_input_stress=dict(custom=stress, hccl=stress),
                    timings=timings,
                )
            )
        result.append(
            dict(
                rank=rank,
                physical_device=config["physical_devices"][rank],
                config_sha256=probe.identity(config),
                passed=True,
                ownership_released=True,
                fallback_calls=0,
                geometries=geometries,
            )
        )
    return result


@pytest.mark.parametrize(
    "mutation", ["rank", "device", "identity", "release", "fallback", "geometry", "case", "stress", "timing"]
)
def test_summary_rejects_missing_or_forged_coverage(config, mutation):
    ranks = reports(config)
    target = ranks[0]
    if mutation in ("rank", "device", "identity", "release", "fallback"):
        key = dict(
            rank="rank",
            device="physical_device",
            identity="config_sha256",
            release="ownership_released",
            fallback="fallback_calls",
        )[mutation]
        target[key] = "wrong"
    elif mutation == "geometry":
        target["geometries"].pop()
    elif mutation == "case":
        target["geometries"][0]["finite_cases"].pop()
    elif mutation == "stress":
        target["geometries"][0]["changed_input_stress"]["custom"] = {}
    else:
        target["geometries"][0]["timings"].pop()
    with pytest.raises(ValueError):
        probe.comparisons(config, ranks)


class Control:
    def __init__(self, fd):
        self.fd, self.closed, self.frames = fd, False, []

    def fileno(self):
        return self.fd

    def settimeout(self, value):
        assert value == probe.CONTROL_TIMEOUT_SECONDS

    def close(self):
        self.closed = True

    def sendall(self, raw):
        self.frames.append(json.loads(raw[4:]))


def launch_mocks(config, output, monkeypatch, *, fail_rank=None, bad_done=False):
    controls, processes, commands = [], [], []
    rank_reports = reports(config)

    def socketpair():
        pair = [Control(100 + len(controls) + index) for index in range(2)]
        controls.extend(pair)
        return pair

    def popen(command, **kwargs):
        rank = int(command[command.index("--worker") + 1])
        commands.append(command)
        assert command[:3] == [sys.executable, "-m", "tools.sidecard.npu_collective_torch_probe"]
        assert kwargs["pass_fds"] == (controls[-1].fd,) and not kwargs["stdout"].closed
        if rank == fail_rank:
            raise OSError("injected spawn failure")
        (output / f"rank{rank}.json").write_text(json.dumps(rank_reports[rank]))
        process = SimpleNamespace(pid=9000 + rank, waited=False)

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


@pytest.mark.parametrize("ranks", [1, 2, 3, 4])
def test_controller_spawns_exact_device_count_and_summarizes_private_receipts(config, tmp_path, monkeypatch, ranks):
    config["physical_devices"] = list(range(ranks))
    config["link_groups"] = [list(range(ranks))]
    if ranks == 3:
        config["algorithm"] = "binomial"
    output = tmp_path / "gate"
    controls, processes, commands = launch_mocks(config, output, monkeypatch)
    result = probe.run(config, output)
    assert len(commands) == ranks and all(process.waited for process in processes)
    assert all(control.closed for control in controls)
    assert len(result["comparison"]) == 2 and result["graphs_qualified"] is False
    for geometry in result["comparison"]:
        assert geometry["comparison"]["clock"] == "shared-host CLOCK_MONOTONIC"
        assert geometry["comparison"]["pure_transit_measured"] is False
    assert not (output / "quarantine.json").exists()
    assert json.loads((output / "result.json").read_text()) == result


@pytest.mark.parametrize("fail_rank,bad_done", [(0, False), (2, False), (None, True)])
def test_controller_failures_retain_live_pids_without_kill_or_wait(config, tmp_path, monkeypatch, fail_rank, bad_done):
    output = tmp_path / "gate"
    controls, processes, _ = launch_mocks(config, output, monkeypatch, fail_rank=fail_rank, bad_done=bad_done)
    with pytest.raises((OSError, ValueError)):
        probe.run(config, output)
    assert json.loads((output / "quarantine.json").read_text()) == dict(
        pids=[process.pid for process in processes], owners_retained=True, automatic_kill=False
    )
    assert all(control.closed for control in controls) and all(not process.waited for process in processes)


@pytest.mark.parametrize("mutation", ["unknown", "pin", "devices", "geometry"])
def test_validation_uses_exact_facade_schema_before_output_or_launch(config, tmp_path, monkeypatch, mutation):
    if mutation == "unknown":
        config["warmup"] = 1
    elif mutation == "pin":
        config["torch_sha256"] = "0" * 64
    elif mutation == "devices":
        config["physical_devices"] = [0, 0]
    else:
        config["geometries"] = []
    output = tmp_path / "gate"
    monkeypatch.setattr(probe.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("invalid config launched"))
    with pytest.raises(ValueError):
        probe.run(config, output)
    assert not output.exists()


def test_fresh_receipt_directory_required_before_any_launch(config, tmp_path, monkeypatch):
    Path(config["output_root"]).mkdir()
    output = tmp_path / "gate"
    monkeypatch.setattr(
        probe.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("existing receipt root launched")
    )
    with pytest.raises(ValueError, match="fresh"):
        probe.run(config, output)
    assert not output.exists()


def test_setup_error_keeps_partially_created_facade(config, tmp_path, monkeypatch):
    env = Environment(config, 0)
    marker = object()
    retained = dict(facade=None, groups=[], tensors=[])

    def create(*args, **kwargs):
        error = RuntimeError("private partial setup")
        error.collective_owner = marker
        raise error

    with pytest.raises(RuntimeError, match="partial setup"):
        probe.exercise(config, tmp_path, 0, None, retained, torch=env.torch, dist=env.dist, create=create)
    assert retained["facade"] is marker and len(retained["groups"]) == 2


def test_import_and_help_do_not_load_torch_or_accelerator_libraries():
    root = Path(__file__).resolve().parents[3]
    code = (
        "import sys,ctypes; ctypes.CDLL=lambda *a,**k: (_ for _ in ()).throw(AssertionError('device load')); "
        "import tools.sidecard.npu_collective_torch_probe; "
        "assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True, capture_output=True)
    result = subprocess.run(
        [sys.executable, "-m", "tools.sidecard.npu_collective_torch_probe", "--help"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--config" in result.stdout and "--worker" in result.stdout
