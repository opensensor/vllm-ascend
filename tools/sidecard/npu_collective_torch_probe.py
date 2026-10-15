# SPDX-License-Identifier: Apache-2.0
"""Explicit eager Torch collective gate; imports no accelerator until worker setup.

Configuration is exactly the pinned Torch facade schema. The controller owns
private socket coordination and a fresh file-store rendezvous. Failures retain
workers and every uncertain owner; the controller never kills or resets them.
"""

import argparse
import contextlib
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

from . import npu_collective_torch as facade
from .npu_collective_probe import CONTROL_TIMEOUT_SECONDS, expected, gather, pattern, receive, send, summarize

CORRECTNESS_CASES = 3
STRESS_CALLS = 30
WARMUP_CALLS = 10
TIMING_PAIRS = 7
TIMING_BATCH = 30


def identity(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate(config):
    if type(config) is not dict or type(config.get("physical_devices")) is not list:
        raise ValueError("explicit Torch facade configuration is required")
    devices = config["physical_devices"]
    if not devices:
        raise ValueError("declare a nonempty physical device list")
    members = tuple(range(len(devices)))
    return facade.validate_config(config, members, 0, devices[0])


def _backend():
    # Worker-only imports are deliberate: validation, help and controller setup
    # must not initialize accelerator runtimes or load an unpinned extension.
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    return torch, dist


def _fallback(_input):
    raise RuntimeError("registered geometry unexpectedly selected fallback")


def _bytes(tensor):
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def _input(torch, geometry, rank, case, device):
    dtype = getattr(torch, geometry["dtype"])
    raw = bytearray(pattern(geometry["elements"], rank, case, geometry["dtype"]))
    return torch.frombuffer(raw, dtype=dtype).clone().to(device=device)


def _verify(tensor, wanted):
    raw = _bytes(tensor)
    if raw != wanted:
        raise AssertionError("complete Torch result differs from exact finite reference")
    return hashlib.sha256(raw).hexdigest()


def exercise(config, output, rank, connection, retained, *, torch, dist, create=facade.create):
    """Run matched operations; retained keeps active tensors/owners on failure."""
    devices, members = config["physical_devices"], tuple(range(len(config["physical_devices"])))
    device_index = devices[rank]
    device = f"npu:{device_index}"
    torch.set_num_threads(1)
    torch.npu.set_device(device_index)
    dist.init_process_group(
        "gloo",
        init_method=(output / "gloo-rendezvous").as_uri(),
        rank=rank,
        world_size=len(members),
        timeout=timedelta(seconds=CONTROL_TIMEOUT_SECONDS),
    )
    cpu_group = dist.group.WORLD
    retained["groups"].append(cpu_group)
    hccl_group = dist.new_group(ranks=list(members), backend="hccl", timeout=timedelta(seconds=CONTROL_TIMEOUT_SECONDS))
    retained["groups"].append(hccl_group)
    try:
        collective = create(
            config,
            members,
            rank,
            cpu_group,
            device_index,
            torch=torch,
            dist=dist,
            stream_getter=lambda: torch.npu.current_stream(device_index),
        )
    except BaseException as error:
        retained["facade"] = getattr(error, "collective_owner", None)
        raise
    retained["facade"] = collective
    reports = []
    for index, geometry in enumerate(config["geometries"]):
        prefix = f"geometry-{index}"

        def synchronize():
            torch.npu.synchronize(device_index)

        def barrier(phase, prefix=prefix):
            gather(connection, prefix + "-" + phase, {})
            dist.barrier(group=cpu_group)

        def submit(mode, source):
            if mode == "custom":
                result = collective.all_reduce(source, _fallback)
                retained["tensors"].append(result)
            else:
                result = source.clone()
                # Retain the clone before a potentially partially submitted SDK
                # collective raises. No CPU coordination occurs in this loop.
                retained["tensors"].append(result)
                dist.all_reduce(result, op=dist.ReduceOp.SUM, group=hccl_group)
            return result

        receipts = []
        for case in range(CORRECTNESS_CASES):
            source = _input(torch, geometry, rank, case, device)
            retained["tensors"] = [source]
            synchronize()
            wanted = expected(geometry["elements"], len(members), case, geometry["dtype"])
            hashes = {}
            for mode in ("custom", "hccl"):
                barrier(f"case-{case}-{mode}")
                result = submit(mode, source)
                synchronize()
                hashes[mode] = _verify(result, wanted)
            _verify(source, pattern(geometry["elements"], rank, case, geometry["dtype"]))
            receipts.append(dict(case=case, bytes=len(wanted), sha256=hashes["custom"], hccl_sha256=hashes["hccl"]))
        # Distinct live tensors test queued-handler lifetimes and descriptor
        # snapshots. There is no synchronize between these thirty submissions.
        sources = [_input(torch, geometry, rank, CORRECTNESS_CASES + case, device) for case in range(STRESS_CALLS)]
        retained["tensors"] = sources.copy()
        synchronize()
        stress = {}
        for mode in ("custom", "hccl"):
            barrier("stress-" + mode)
            outputs = [submit(mode, source) for source in sources]
            pointers = [tensor.data_ptr() for tensor in sources + outputs]
            if len(set(pointers)) != STRESS_CALLS * 2:
                raise AssertionError("stress requires distinct live input/output allocations")
            synchronize()
            hashes = []
            for case, result in enumerate(outputs):
                wanted = expected(geometry["elements"], len(members), CORRECTNESS_CASES + case, geometry["dtype"])
                hashes.append(_verify(result, wanted))
                _verify(sources[case], pattern(geometry["elements"], rank, CORRECTNESS_CASES + case, geometry["dtype"]))
            stress[mode] = dict(
                calls=STRESS_CALLS, distinct_inputs=STRESS_CALLS, distinct_outputs=STRESS_CALLS, sha256=hashes
            )
        source = sources[-1]
        wanted = expected(geometry["elements"], len(members), CORRECTNESS_CASES + STRESS_CALLS - 1, geometry["dtype"])
        for mode in ("custom", "hccl"):
            retained["tensors"] = [source]
            barrier("warmup-" + mode)
            outputs = [submit(mode, source) for _ in range(WARMUP_CALLS)]
            synchronize()
            for result in outputs:
                _verify(result, wanted)
        timings = []
        for pair in range(TIMING_PAIRS):
            modes = ("custom", "hccl") if pair % 2 == 0 else ("hccl", "custom")
            for mode in modes:
                retained["tensors"] = [source]
                barrier(f"timed-{pair}-{mode}")
                started = time.monotonic_ns()
                outputs = [submit(mode, source) for _ in range(TIMING_BATCH)]
                submitted = time.monotonic_ns()
                synchronize()
                completed = time.monotonic_ns()
                for result in outputs:
                    _verify(result, wanted)
                timings.append(
                    dict(
                        pair=pair,
                        mode=mode,
                        count=TIMING_BATCH,
                        started_ns=started,
                        submitted_ns=submitted,
                        completed_ns=completed,
                    )
                )
        reports.append(dict(**geometry, finite_cases=receipts, changed_input_stress=stress, timings=timings))
        retained["tensors"] = []
    if collective.fallbacks:
        raise AssertionError("registered geometry fell back during Torch qualification")
    gather(connection, "completed", {})
    collective.close(graphs_retired=True)
    gather(connection, "facade-closed", dict(rank=rank, closed=True))
    dist.destroy_process_group(hccl_group)
    dist.destroy_process_group(cpu_group)
    retained["groups"] = []
    return dict(
        rank=rank,
        physical_device=device_index,
        config_sha256=identity(config),
        passed=True,
        ownership_released=True,
        fallback_calls=0,
        geometries=reports,
    )


def worker(config, output, rank, connection):
    retained = dict(facade=None, groups=[], tensors=[])
    try:
        torch, dist = _backend()
        result = exercise(config, output, rank, connection, retained, torch=torch, dist=dist)
        path = output / f"rank{rank}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        gather(
            connection,
            "done",
            dict(rank=rank, passed=True, result_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        )
    except BaseException as error:
        retained["error"] = error
        # stderr is the controller's private rank log. Do not expose messages,
        # locals, IPC descriptors or addresses in public failure receipts.
        traceback.print_exception(error, file=sys.stderr)
        chain, cause = [], error
        while cause is not None:
            chain.append(type(cause).__name__)
            cause = cause.__cause__
        result = dict(
            rank=rank, passed=False, pid=os.getpid(), owners_retained=True, automatic_cleanup=False, error_types=chain
        )
        with contextlib.suppress(OSError):
            (output / f"rank{rank}-failure.json").write_text(json.dumps(result, indent=2) + "\n")
            send(connection, dict(phase="error", payload=result))
        while True:
            signal.pause()


def comparisons(config, ranks):
    if type(ranks) is not list or len(ranks) != len(config["physical_devices"]):
        raise ValueError("exactly one Torch report per assigned rank is required")
    result = []
    for index, geometry in enumerate(config["geometries"]):
        samples = []
        for rank, report in enumerate(ranks):
            expected_identity = dict(
                rank=rank,
                physical_device=config["physical_devices"][rank],
                config_sha256=identity(config),
                passed=True,
                ownership_released=True,
                fallback_calls=0,
            )
            if (
                type(report) is not dict
                or set(report) != set(expected_identity) | {"geometries"}
                or any(
                    type(report.get(key)) is not type(value) or report[key] != value
                    for key, value in expected_identity.items()
                )
            ):
                raise ValueError("Torch rank identity, ownership or fallback receipt differs")
            values = report.get("geometries")
            if type(values) is not list or len(values) != len(config["geometries"]):
                raise ValueError("Torch geometry coverage differs")
            selected = values[index]
            if (
                type(selected) is not dict
                or set(selected) != set(geometry) | {"finite_cases", "changed_input_stress", "timings"}
                or any(
                    type(selected.get(key)) is not type(value) or selected[key] != value
                    for key, value in geometry.items()
                )
            ):
                raise ValueError("Torch geometry identity differs")
            cases = selected.get("finite_cases")
            stress = selected.get("changed_input_stress")
            if (
                type(cases) is not list
                or any(
                    type(case) is not dict or type(case.get("case")) is not int or type(case.get("bytes")) is not int
                    for case in cases
                )
                or [case.get("case") for case in cases] != list(range(CORRECTNESS_CASES))
            ):
                raise ValueError("Torch finite case coverage differs")
            for case in cases:
                wanted = expected(geometry["elements"], len(ranks), case["case"], geometry["dtype"])
                checksum = hashlib.sha256(wanted).hexdigest()
                if case != dict(case=case["case"], bytes=len(wanted), sha256=checksum, hccl_sha256=checksum):
                    raise ValueError("Torch finite result receipt differs")
            wanted_stress = [
                hashlib.sha256(
                    expected(geometry["elements"], len(ranks), CORRECTNESS_CASES + case, geometry["dtype"])
                ).hexdigest()
                for case in range(STRESS_CALLS)
            ]
            wanted_stress = dict(
                calls=STRESS_CALLS, distinct_inputs=STRESS_CALLS, distinct_outputs=STRESS_CALLS, sha256=wanted_stress
            )
            if (
                type(stress) is not dict
                or set(stress) != {"custom", "hccl"}
                or any(
                    type(stress[mode]) is not dict
                    or any(
                        type(stress[mode].get(key)) is not int
                        for key in ("calls", "distinct_inputs", "distinct_outputs")
                    )
                    for mode in stress
                )
                or stress != dict(custom=wanted_stress, hccl=wanted_stress)
            ):
                raise ValueError("Torch lifetime stress coverage or result differs")
            timings = selected.get("timings")
            if (
                type(timings) is not list
                or any(type(sample) is not dict for sample in timings)
                or len(timings) != TIMING_PAIRS * 2
                or {(sample.get("pair"), sample.get("mode"), sample.get("count")) for sample in timings}
                != {(pair, mode, TIMING_BATCH) for pair in range(TIMING_PAIRS) for mode in ("custom", "hccl")}
            ):
                raise ValueError("Torch timing coverage differs")
            samples.append(dict(rank=rank, timings=timings))
        result.append(dict(**geometry, comparison=summarize(samples)))
    return result


def run(config, output):
    config, output = validate(config), Path(output).absolute()
    if Path(config["output_root"]).exists():
        raise ValueError("Torch facade receipts require a fresh private output_root")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    path = output / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    connections, processes = [], []

    def quarantine():
        (output / "quarantine.json").write_text(
            json.dumps(dict(pids=[p.pid for p in processes], owners_retained=True, automatic_kill=False), indent=2)
            + "\n"
        )

    try:
        for rank in range(len(config["physical_devices"])):
            parent, child = socket.socketpair()
            connections.append(parent)
            parent.settimeout(CONTROL_TIMEOUT_SECONDS)
            command = [
                sys.executable,
                "-m",
                "tools.sidecard.npu_collective_torch_probe",
                "--config",
                str(path),
                "--output",
                str(output),
                "--worker",
                str(rank),
                "--fd",
                str(child.fileno()),
            ]
            try:
                with (output / f"rank{rank}.log").open("w") as log:
                    processes.append(
                        subprocess.Popen(command, pass_fds=(child.fileno(),), stdout=log, stderr=subprocess.STDOUT)
                    )
                (output / "processes.json").write_text(
                    json.dumps(dict(pids=[p.pid for p in processes]), indent=2) + "\n"
                )
            finally:
                child.close()
        while True:
            packets = [receive(connection) for connection in connections]
            phase = packets[0].get("phase")
            if type(phase) is not str or phase == "error" or any(packet.get("phase") != phase for packet in packets):
                raise RuntimeError("Torch collective rank control phase failed")
            with (output / "progress.jsonl").open("a") as log:
                log.write(json.dumps(dict(phase=phase, at_ns=time.monotonic_ns())) + "\n")
            payloads = [packet["payload"] for packet in packets]
            for connection in connections:
                send(connection, dict(phase=phase, payloads=payloads))
            if phase == "done":
                reports = []
                for rank, packet in enumerate(payloads):
                    raw = (output / f"rank{rank}.json").read_bytes()
                    wanted = dict(rank=rank, passed=True, result_sha256=hashlib.sha256(raw).hexdigest())
                    if (
                        type(packet) is not dict
                        or set(packet) != set(wanted)
                        or any(
                            type(packet[key]) is not type(value) or packet[key] != value
                            for key, value in wanted.items()
                        )
                    ):
                        raise ValueError("Torch rank result receipt differs")
                    reports.append(json.loads(raw))
                break
        result = dict(
            passed=True,
            ranks=reports,
            physical_devices=config["physical_devices"],
            algorithm=config["algorithm"],
            comparison=comparisons(config, reports),
            graphs_qualified=False,
            numerical_scope="synthetic exactly representable finite Torch inputs; not model qualification",
            timing_scope="out-of-place custom vs input.clone plus Torch HCCL SUM; CPU barriers excluded",
        )
        for process in processes:
            if process.wait(timeout=10):
                raise RuntimeError("completed Torch worker did not exit successfully")
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    except BaseException:
        quarantine()
        raise
    finally:
        for connection in connections:
            connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--fd", type=int)
    args = parser.parse_args()
    config = validate(json.loads(args.config.read_text()))
    if args.worker is None:
        if args.fd is not None:
            parser.error("private FD requires an assigned worker")
        result = run(config, args.output)
        print(json.dumps(dict(passed=result["passed"], ranks=len(result["ranks"]))))
    else:
        if args.fd is None or args.worker not in range(len(config["physical_devices"])):
            parser.error("worker requires its assigned rank and private control FD")
        worker(config, args.output.absolute(), args.worker, socket.socket(fileno=args.fd))


if __name__ == "__main__":
    main()
