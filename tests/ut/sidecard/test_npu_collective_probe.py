# SPDX-License-Identifier: Apache-2.0
"""Portable probe contract tests; fake controllers do not qualify devices."""

import copy
import hashlib
import json
import socket
import struct
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from tools.sidecard import npu_collective_probe as probe
from tools.sidecard.pair_collective_probe import FRAME_BYTES


@pytest.fixture
def config(tmp_path):
    result = dict(physical_chips=[0, 2], elements=32, dtype="float16")
    for name in ("acl", "native", "reference"):
        path = tmp_path / (name + ".so")
        path.write_bytes(("fake pinned library " + name).encode())
        if name != "reference":
            result[name + "_library"] = str(path)
            result[name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


@pytest.mark.parametrize("chips", ([0], [3], [0, 1], [0, 2], [2, 0, 3], [3, 2, 1, 0]))
@pytest.mark.parametrize("dtype", ("float16", "float32"))
def test_pinned_configuration_defaults_and_explicit_physical_order(config, chips, dtype):
    config.update(physical_chips=chips, dtype=dtype)
    original = copy.deepcopy(config)
    actual = probe.validate(config)
    assert actual == dict(original, warmup=10, pairs=7, batch=30, cases=3, stress=True, algorithm="binomial")
    assert config == original


@pytest.mark.parametrize("raw", (None, [], "configuration", 1, True))
def test_configuration_must_be_an_object(raw):
    with pytest.raises(ValueError, match="fields"):
        probe.validate(raw)


@pytest.mark.parametrize(
    "field,value",
    (
        ("physical_chips", []),
        ("physical_chips", [0, 1, 2, 3, 4]),
        ("physical_chips", [0, 0]),
        ("physical_chips", [-1]),
        ("physical_chips", [4]),
        ("physical_chips", [True]),
        ("physical_chips", [0.0]),
        ("physical_chips", ["0"]),
        ("physical_chips", [[0]]),
        ("physical_chips", (0, 1)),
        ("physical_chips", None),
        ("dtype", "bfloat16"),
        ("dtype", "FLOAT16"),
        ("dtype", []),
        ("dtype", None),
        ("dtype", True),
        ("elements", 0),
        ("elements", -1),
        ("elements", True),
        ("elements", 1.0),
        ("elements", "32"),
        ("elements", None),
        ("elements", 16 * 1024 * 1024 + 1),
        ("unknown", "unsupported"),
        ("stress", 1),
        ("stress", "false"),
        ("stress", None),
    ),
)
def test_invalid_configuration_rejected_before_launch(config, field, value):
    config[field] = value
    with pytest.raises(ValueError):
        probe.validate(config)


@pytest.mark.parametrize("field", ("physical_chips", "elements", "dtype", "acl_library", "native_sha256"))
def test_required_configuration_fields(config, field):
    del config[field]
    with pytest.raises(ValueError, match="fields"):
        probe.validate(config)


@pytest.mark.parametrize("field,upper", (("warmup", 100), ("pairs", 50), ("batch", 60), ("cases", 20)))
@pytest.mark.parametrize("bad", (0, -1, True, 1.0, "1", None, "above-limit"))
def test_probe_work_limits_require_bounded_integers(config, field, upper, bad):
    config[field] = upper + 1 if bad == "above-limit" else bad
    with pytest.raises(ValueError, match=field):
        probe.validate(config)


def test_maximum_work_limits_are_admitted_without_allocating_payload(config):
    config.update(elements=16 * 1024 * 1024, warmup=100, pairs=50, batch=60, cases=20, stress=False)
    assert probe.validate(config) == dict(config, algorithm="binomial")


@pytest.mark.parametrize("algorithm", (None, True, 1, [], {}, "ring", "recursive", "Recursive-doubling", ""))
def test_algorithm_requires_an_exact_supported_name(config, algorithm):
    config["algorithm"] = algorithm
    with pytest.raises(ValueError, match="algorithm"):
        probe.validate(config)


@pytest.mark.parametrize("chips", ([0], [3], [0, 2], [3, 2, 1, 0]))
@pytest.mark.parametrize("dtype", ("float16", "float32"))
def test_recursive_doubling_admits_only_actual_power_of_two_groups(config, chips, dtype):
    config.update(physical_chips=chips, dtype=dtype, algorithm="recursive-doubling")
    result = probe.validate(config)
    assert result["physical_chips"] == chips
    assert result["algorithm"] == "recursive-doubling"
    assert result["dtype"] == dtype


def test_recursive_doubling_rejects_three_ranks_without_padding_or_fallback(config):
    config.update(physical_chips=[2, 0, 3], algorithm="recursive-doubling")
    with pytest.raises(ValueError, match="power-of-two"):
        probe.validate(config)
    assert config["physical_chips"] == [2, 0, 3]
    config["algorithm"] = "binomial"
    assert probe.validate(config)["algorithm"] == "binomial"


@pytest.mark.parametrize("name", ("acl", "native", "reference"))
@pytest.mark.parametrize(
    "bad", ("changed-file", "wrong-sha", "relative", "missing", "directory", "path-type", "sha-type")
)
def test_all_libraries_require_existing_absolute_sha_pins(config, tmp_path, name, bad):
    path = tmp_path / (name + ".so")
    config[name + "_library"] = str(path)
    config[name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if bad == "changed-file":
        path.write_bytes(b"changed after manifest")
    elif bad == "wrong-sha":
        config[name + "_sha256"] = "0" * 64
    elif bad == "relative":
        config[name + "_library"] = path.name
    elif bad == "missing":
        path.unlink()
    elif bad == "directory":
        config[name + "_library"] = str(tmp_path)
    elif bad == "path-type":
        config[name + "_library"] = [str(path)]
    else:
        config[name + "_sha256"] = None
    with pytest.raises(ValueError, match="library"):
        probe.validate(config)


@pytest.mark.parametrize("provided", ("reference_library", "reference_sha256"))
def test_optional_hccl_reference_is_all_or_none(config, provided):
    config[provided] = "unpaired reference"
    with pytest.raises(ValueError, match="requires its SHA256"):
        probe.validate(config)


def test_optional_raw_hccl_reference_is_authenticated(config, tmp_path):
    path = tmp_path / "reference.so"
    config.update(reference_library=str(path), reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert probe.validate(config)["reference_library"] == str(path)


@pytest.mark.parametrize("ranks", (1, 2, 3, 4))
@pytest.mark.parametrize("dtype,code,width", (("float16", "e", 2), ("float32", "f", 4)))
@pytest.mark.parametrize("elements", (1, 31, 32, 129))
@pytest.mark.parametrize("case", (0, 1, 19, 49, 79, 109))
def test_complete_exact_patterns_include_changed_input_and_stress_cases(ranks, dtype, code, width, elements, case):
    # Quantize every tree addition independently, rather than summing only in
    # Python float64. These eighth-integer cases must stay exact in both dtypes.
    unpack = lambda raw: struct.unpack("<" + code * elements, raw)
    pack = lambda values: struct.pack("<" + code * elements, *values)
    partial = [0.0] * elements
    inputs = []
    for rank in range(ranks):
        raw = probe.pattern(elements, rank, case, dtype)
        assert len(raw) == elements * width
        values = unpack(raw)
        wanted = [float(Fraction(i % 31 - 15) + Fraction(rank, 4) + Fraction(case, 8)) for i in range(elements)]
        assert list(values) == wanted
        partial = unpack(pack([left + right for left, right in zip(partial, values, strict=True)]))
        inputs.append(raw)
    result = probe.expected(elements, ranks, case, dtype)
    independent = [
        float(sum(Fraction(i % 31 - 15) + Fraction(rank, 4) + Fraction(case, 8) for rank in range(ranks)))
        for i in range(elements)
    ]
    assert len(result) == elements * width
    assert result == pack(independent) == pack(partial)
    assert len(set(inputs)) == ranks
    assert result != probe.expected(elements, ranks, case + 1, dtype)


def test_private_control_gather_roundtrip():
    local, peer = socket.socketpair()
    with local, peer:
        reports = [{"rank": 0}, {"rank": 1}]
        probe.send(peer, dict(phase="connected", payloads=reports))
        assert probe.gather(local, "connected", {"schedule_sha256": "private"}) == reports
        assert probe.receive(peer) == dict(phase="connected", payload={"schedule_sha256": "private"})


@pytest.mark.parametrize(
    "reply",
    (
        {},
        {"phase": "wrong", "payloads": []},
        {"phase": "connected"},
        {"phase": "connected", "payloads": {}},
        {"phase": "connected", "payloads": None},
    ),
)
def test_control_gather_rejects_phase_or_payload_shape(reply):
    local, peer = socket.socketpair()
    with local, peer:
        probe.send(peer, reply)
        with pytest.raises(RuntimeError, match="phase differs"):
            probe.gather(local, "connected", {})


@pytest.mark.parametrize("size", (0, FRAME_BYTES + 1, 2**32 - 1))
def test_control_receive_rejects_unbounded_frame_lengths(size):
    local, peer = socket.socketpair()
    with local, peer:
        peer.sendall(struct.pack("!I", size))
        with pytest.raises(ValueError, match="frame size"):
            probe.receive(local)


@pytest.mark.parametrize("body", (b"[]", b"null", b"3", b"invalid json"))
def test_control_receive_requires_json_object(body):
    local, peer = socket.socketpair()
    with local, peer:
        peer.sendall(struct.pack("!I", len(body)) + body)
        with pytest.raises(ValueError):
            probe.receive(local)


@pytest.mark.parametrize("prefix", (b"", b"\x00\x00", struct.pack("!I", 12) + b"{}"))
def test_control_disconnect_does_not_return_partial_packet(prefix):
    local, peer = socket.socketpair()
    with local, peer:
        peer.sendall(prefix)
        peer.shutdown(socket.SHUT_WR)
        with pytest.raises(EOFError):
            probe.receive(local)


def test_control_send_rejects_oversize_and_nonfinite_values():
    local, peer = socket.socketpair()
    with local, peer:
        for payload in ({"data": "x" * FRAME_BYTES}, {"value": float("nan")}):
            with pytest.raises(ValueError):
                probe.send(local, payload)


def timing_reports(ranks, pairs=2, modes=("custom", "hccl")):
    return [
        dict(
            rank=rank,
            timings=[
                dict(
                    pair=pair,
                    mode=mode,
                    count=10,
                    started_ns=1_000_000 + pair * 1_000_000 + rank * 1_000,
                    submitted_ns=1_010_000 + pair * 1_000_000 + rank * 2_000,
                    completed_ns=1_100_000 + pair * 1_000_000 + rank * 3_000 + (mode == "hccl") * 100_000,
                )
                for pair in range(pairs)
                for mode in modes
            ],
        )
        for rank in range(ranks)
    ]


@pytest.mark.parametrize("ranks", (1, 2, 3, 4))
@pytest.mark.parametrize("modes", (("custom",), ("custom", "hccl")))
def test_summary_matches_host_clock_envelope_without_transit_or_confidence_claims(ranks, modes):
    reports = timing_reports(ranks, modes=modes)
    result = probe.summarize(reports)
    assert result["clock"] == "shared-host CLOCK_MONOTONIC"
    assert result["scope"] == "completed batch including host dispatch/arrival spread"
    assert result["pure_transit_measured"] is False
    assert result["confidence_interval_computed"] is False
    assert set(result["modes"]) == set(modes)
    assert len(result["samples"]) == 2 * len(modes)
    for value in result["samples"]:
        wanted = 10 + (ranks - 1) * 0.3 + (value["mode"] == "hccl") * 10
        assert value["completed_batch_us_per_call"] == pytest.approx(wanted)
        assert value["arrival_spread_us"] == ranks - 1
        assert value["rank_submission_us_per_call"] == [1 + rank * 0.1 for rank in range(ranks)]
        assert value["count"] == 10
        aggregate = result["modes"][value["mode"]]
        assert aggregate == dict(median_us=wanted, min_us=wanted, max_us=wanted, samples_us=[wanted, wanted])


@pytest.mark.parametrize("raw", (None, {}, [], [None], [{"rank": True}], [{"rank": 1}], [{"rank": 0}, {"rank": 0}]))
def test_summary_requires_exact_distinct_rank_set(raw):
    with pytest.raises(ValueError):
        probe.summarize(raw)


@pytest.mark.parametrize("raw", (None, {}, [], [None], [1]))
def test_summary_requires_nonempty_sample_lists(raw):
    with pytest.raises(ValueError):
        probe.summarize([dict(rank=0, timings=raw)])


@pytest.mark.parametrize("field", ("pair", "count", "started_ns", "submitted_ns", "completed_ns"))
@pytest.mark.parametrize("bad", (True, -1, 1.0, "1", None, float("nan"), float("inf")))
def test_summary_rejects_clock_and_count_coercions_on_any_rank(field, bad):
    reports = timing_reports(2)
    reports[1]["timings"][0][field] = bad
    with pytest.raises(ValueError):
        probe.summarize(reports)


@pytest.mark.parametrize(
    "mutation",
    (
        "duplicate",
        "missing",
        "different-mode",
        "unknown-mode",
        "different-pair",
        "different-count",
        "zero-count",
        "clock-order",
        "missing-field",
        "extra-field",
    ),
)
def test_summary_rejects_unmatched_samples_and_invalid_host_intervals(mutation):
    reports = timing_reports(3)
    values = reports[2]["timings"]
    if mutation == "duplicate":
        values.append(copy.deepcopy(values[0]))
    elif mutation == "missing":
        values.pop()
    elif mutation == "different-mode":
        values[0]["mode"] = "hccl"
    elif mutation == "unknown-mode":
        values[0]["mode"] = "kernel-event-only"
    elif mutation == "different-pair":
        values[0]["pair"] = 99
    elif mutation == "different-count":
        values[0]["count"] += 1
    elif mutation == "zero-count":
        values[0]["count"] = 0
    elif mutation == "clock-order":
        values[0]["completed_ns"] = values[0]["submitted_ns"] - 1
    elif mutation == "missing-field":
        del values[0]["started_ns"]
    else:
        values[0]["device_ns"] = 1
    with pytest.raises(ValueError):
        probe.summarize(reports)


class FakeControl:
    """Record framed controller writes without opening any device or socket."""

    def __init__(self, fd):
        self.fd = fd
        self.closed = False
        self.frames = []
        self.timeout = None

    def fileno(self):
        return self.fd

    def settimeout(self, timeout):
        self.timeout = timeout

    def close(self):
        self.closed = True

    def sendall(self, frame):
        assert not self.closed
        size = struct.unpack("!I", frame[:4])[0]
        assert size == len(frame) - 4 <= FRAME_BYTES
        self.frames.append(json.loads(frame[4:]))


class FakeProcess:
    def __init__(self, pid, returncode=0):
        self.pid = pid
        self.returncode = returncode
        self.waited = False

    def wait(self, timeout):
        assert timeout == 10
        self.waited = True
        return self.returncode


def install_fake_launch(monkeypatch, reports, output, *, fail_rank=None, done_mutation=None, phase="done"):
    controls, processes, commands = [], [], []

    def socketpair():
        parent, child = FakeControl(len(controls) + 100), FakeControl(len(controls) + 101)
        controls.extend((parent, child))
        return parent, child

    def launch(command, *, pass_fds, stdout, stderr):
        rank = int(command[command.index("--worker") + 1])
        commands.append(command)
        assert command[:3] == [sys.executable, "-m", "tools.sidecard.npu_collective_probe"]
        assert pass_fds == (controls[-1].fileno(),)
        assert not stdout.closed and stderr == subprocess.STDOUT
        if rank == fail_rank:
            raise OSError("fake spawn failed")
        selected = json.loads(Path(command[command.index("--config") + 1]).read_text())
        reports[rank].setdefault("physical_chip", selected["physical_chips"][rank])
        reports[rank].setdefault("passed", True)
        reports[rank].setdefault("ownership_released", True)
        (output / f"rank{rank}.json").write_text(json.dumps(reports[rank]))
        process = FakeProcess(7000 + rank)
        processes.append(process)
        return process

    def receive(connection):
        rank = controls.index(connection) // 2
        raw = (output / f"rank{rank}.json").read_bytes()
        payload = dict(rank=rank, passed=True, result_sha256=hashlib.sha256(raw).hexdigest())
        if done_mutation is not None:
            done_mutation(rank, payload)
        return dict(phase=phase, payload=payload)

    monkeypatch.setattr(probe.socket, "socketpair", socketpair)
    monkeypatch.setattr(probe.subprocess, "Popen", launch)
    monkeypatch.setattr(probe, "receive", receive)
    return controls, processes, commands


def test_partial_launch_failure_preserves_pids_and_closes_all_control_fds(config, tmp_path, monkeypatch):
    output = tmp_path / "probe"
    controls, processes, _ = install_fake_launch(monkeypatch, timing_reports(2), output, fail_rank=1)
    with pytest.raises(OSError, match="spawn failed"):
        probe.run(config, output)
    assert json.loads((output / "quarantine.json").read_text()) == dict(
        pids=[7000], owners_retained=True, automatic_kill=False
    )
    assert json.loads((output / "processes.json").read_text()) == dict(pids=[7000])
    assert len(controls) == 4 and all(connection.closed for connection in controls)
    assert not processes[0].waited
    assert not (output / "result.json").exists()


def test_first_launch_failure_also_records_quarantine(config, tmp_path, monkeypatch):
    output = tmp_path / "probe"
    controls, _, _ = install_fake_launch(monkeypatch, timing_reports(2), output, fail_rank=0)
    with pytest.raises(OSError):
        probe.run(config, output)
    assert json.loads((output / "quarantine.json").read_text())["pids"] == []
    assert all(connection.closed for connection in controls)


@pytest.mark.parametrize("ranks", (1, 2, 3, 4))
def test_controller_authenticates_private_reports_and_uses_compact_done_frames(config, tmp_path, monkeypatch, ranks):
    config.update(physical_chips=list(range(ranks)), pairs=50)
    output = tmp_path / "probe"
    reports = timing_reports(ranks, pairs=50)
    controls, processes, commands = install_fake_launch(monkeypatch, reports, output)
    result = probe.run(config, output)
    assert result["passed"] is True and result["ranks"] == reports
    assert result["graphs_qualified"] is False and result["notify_timeout_units_verified"] is False
    assert "not model numerical qualification" in result["numerical_scope"]
    assert result["physical_chips"] == list(range(ranks))
    assert result["algorithm"] == "binomial"
    assert all(process.waited for process in processes)
    assert len(commands) == ranks and all(connection.closed for connection in controls)
    assert all(connection.timeout == probe.CONTROL_TIMEOUT_SECONDS for connection in controls[::2])
    for connection in controls[::2]:
        assert len(connection.frames) == 1
        payloads = connection.frames[0]["payloads"]
        assert [packet["rank"] for packet in payloads] == list(range(ranks))
        assert all(set(packet) == {"rank", "passed", "result_sha256"} for packet in payloads)
    assert len(json.dumps(controls[0].frames[0]).encode()) < len(json.dumps(reports).encode()) / 10
    assert not (output / "quarantine.json").exists()


@pytest.mark.parametrize("ranks", (1, 2, 4))
def test_controller_result_identifies_recursive_doubling(config, tmp_path, monkeypatch, ranks):
    config.update(physical_chips=list(range(ranks)), algorithm="recursive-doubling")
    output = tmp_path / "probe"
    install_fake_launch(monkeypatch, timing_reports(ranks), output)
    result = probe.run(config, output)
    assert result["algorithm"] == "recursive-doubling"
    assert json.loads((output / "config.json").read_text())["algorithm"] == "recursive-doubling"
    assert json.loads((output / "result.json").read_text())["algorithm"] == "recursive-doubling"


@pytest.mark.parametrize("bad", ("rank", "hash", "extra", "boolean-rank", "integer-pass"))
def test_controller_rejects_drifted_done_receipts_and_retains_owners(config, tmp_path, monkeypatch, bad):
    def mutate(rank, payload):
        if rank == 0 and bad == "boolean-rank":
            payload["rank"] = False
        elif rank == 1 and bad == "integer-pass":
            payload["passed"] = 1
        elif rank == 1 and bad not in ("boolean-rank", "integer-pass"):
            payload[{"rank": "rank", "hash": "result_sha256", "extra": "unexpected"}[bad]] = "changed"

    output = tmp_path / "probe"
    controls, processes, _ = install_fake_launch(monkeypatch, timing_reports(2), output, done_mutation=mutate)
    with pytest.raises(ValueError, match="receipt differs"):
        probe.run(config, output)
    assert json.loads((output / "quarantine.json").read_text())["pids"] == [7000, 7001]
    assert all(connection.closed for connection in controls)
    assert all(not process.waited for process in processes)
    assert not (output / "result.json").exists()


@pytest.mark.parametrize(
    "field,bad",
    (
        ("rank", False),
        ("rank", 1),
        ("physical_chip", False),
        ("physical_chip", 3),
        ("passed", False),
        ("passed", 1),
        ("ownership_released", False),
        ("ownership_released", 1),
    ),
)
def test_authenticated_report_must_confirm_actual_rank_chip_success_and_release(
    config, tmp_path, monkeypatch, field, bad
):
    output = tmp_path / "probe"
    reports = timing_reports(2)
    reports[0][field] = bad
    controls, processes, _ = install_fake_launch(monkeypatch, reports, output)
    with pytest.raises(ValueError):
        probe.run(config, output)
    assert json.loads((output / "quarantine.json").read_text())["owners_retained"] is True
    assert all(connection.closed for connection in controls)
    assert all(not process.waited for process in processes)
    assert not (output / "result.json").exists()


def test_worker_error_phase_quarantines_without_automatic_process_kill(config, tmp_path, monkeypatch):
    output = tmp_path / "probe"
    controls, processes, _ = install_fake_launch(monkeypatch, timing_reports(2), output, phase="error")
    with pytest.raises(RuntimeError, match="rank phase failed"):
        probe.run(config, output)
    assert json.loads((output / "quarantine.json").read_text()) == dict(
        pids=[7000, 7001], owners_retained=True, automatic_kill=False
    )
    assert all(connection.closed and not connection.frames for connection in controls)
    assert all(not process.waited for process in processes)


def test_invalid_configuration_has_no_output_or_launch_side_effects(config, tmp_path, monkeypatch):
    output = tmp_path / "probe"

    def forbidden(*args, **kwargs):
        pytest.fail("invalid configuration reached process or socket creation")

    monkeypatch.setattr(probe.subprocess, "Popen", forbidden)
    monkeypatch.setattr(probe.socket, "socketpair", forbidden)
    config["physical_chips"] = [0, 0]
    with pytest.raises(ValueError):
        probe.run(config, output)
    assert not output.exists()


def test_import_and_cli_help_do_not_load_accelerator_libraries():
    root = Path(__file__).resolve().parents[3]
    code = (
        "import ctypes, sys; "
        "ctypes.CDLL = lambda *a, **k: (_ for _ in ()).throw(AssertionError('device library load')); "
        "import tools.sidecard.npu_collective_probe; "
        "assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True, capture_output=True, text=True)
    result = subprocess.run(
        [sys.executable, "-m", "tools.sidecard.npu_collective_probe", "--help"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--config" in result.stdout and "--worker" in result.stdout
