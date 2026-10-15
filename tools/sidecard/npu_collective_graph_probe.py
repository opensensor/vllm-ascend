# SPDX-License-Identifier: Apache-2.0
"""Explicit fixed-handshake Torch graph gate, separate from model qualification.

Use the pinned facade configuration with execution_mode="graph" and algorithm
"recursive-doubling-fixed". One separate stream owns eager warmup, capture,
changed-input replay and interleaved eager calls. Failed workers retain graphs,
streams, tensors and native owners without automatic recovery or process kills.
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
from . import npu_collective_torch_probe as eager
from .npu_collective_probe import CONTROL_TIMEOUT_SECONDS, expected, gather, pattern, receive, send, summarize

CAPTURE_COUNTS = (1, 3)
CHANGED_REPLAYS = 5
WARMUP = 3
TIMING_PAIRS = 7
TIMING_REPLAYS = 30
FINAL_SWEEPS = ("reverse", "forward")


def validate(config):
    if (
        type(config) is not dict
        or config.get("execution_mode") != "graph"
        or config.get("algorithm") != "recursive-doubling-fixed"
    ):
        raise ValueError("graph probe requires explicit graph mode and recursive-doubling-fixed")
    return eager.validate(config)


def _case(replay, index):
    return 3 + replay * max(CAPTURE_COUNTS) + index


def _checksum(geometry, ranks, case):
    return hashlib.sha256(expected(geometry["elements"], ranks, case, geometry["dtype"])).hexdigest()


def observer_receipt(config):
    graphs = len(config["geometries"]) * len(CAPTURE_COUNTS)
    calls = len(config["geometries"]) * sum(CAPTURE_COUNTS)
    tracking = config.get("graph_replay_tracking", False)
    replays = CHANGED_REPLAYS + TIMING_PAIRS * TIMING_REPLAYS + len(FINAL_SWEEPS) if tracking else 0
    return dict(
        registered_graphs=graphs,
        captured_graphs=graphs,
        failed_graphs=0,
        replay_submissions=graphs * replays,
        captured_calls=calls,
        replayed_calls=calls * replays,
        reset_graphs=graphs,
        active_capture=False,
        retired=True,
        lifecycle_only=True,
        tracking_enabled=tracking,
        count_semantics="capture-and-replay-submissions" if tracking else "capture-only",
        native_leases_acquired=graphs * len(config["geometries"]),
        native_leases_released=graphs * len(config["geometries"]),
    )


def final_history(config):
    """Exact ordered final replay receipts, independently derived by controller."""
    geometries = [dict(**geometry, operations=count) for geometry in config["geometries"] for count in CAPTURE_COUNTS]
    history = []
    for direction in FINAL_SWEEPS:
        indices = range(len(geometries) - 1, -1, -1) if direction == "reverse" else range(len(geometries))
        for index in indices:
            geometry = geometries[index]
            cases = [_case(CHANGED_REPLAYS - 1, offset) for offset in range(geometry["operations"])]
            hashes = [_checksum(geometry, len(config["physical_devices"]), case) for case in cases]
            history.append(
                dict(direction=direction, capture_index=index, **geometry, cases=cases, custom=hashes, hccl=hashes)
            )
    return history


def _replay(graph, observer):
    if observer is not None and not observer.track_replays:
        # The actual launch is Torch-NPU's bound native method. Ownership was
        # established during capture; no observer callback runs on this path.
        try:
            return graph.replay()
        except BaseException as error:
            observer.replay_failed(graph, error)
            raise
    try:
        if observer is not None:
            observer.before_replay(graph)
        graph.replay()
        if observer is not None:
            observer.replayed(graph)
    except BaseException as error:
        if observer is not None:
            observer.replay_failed(graph, error)
        raise


def _capture(torch, graph, stream, inputs, submit, observer, retained):
    if observer is not None:
        observer.capture_begin(graph)
    try:
        with torch.npu.graph(graph, stream=stream):
            outputs = []
            for source in inputs:
                result = submit(source)
                outputs.append(result)
                retained["tensors"].append(result)
        if observer is not None:
            observer.capture_end(graph)
        return outputs
    except BaseException as error:
        if observer is not None:
            observer.capture_failed(graph, error)
        raise


def exercise(config, output, rank, connection, retained, *, torch, dist, create=facade.create):
    """Capture independent SUMs and keep every graph owner until retirement."""
    devices = config["physical_devices"]
    members, device_index = tuple(range(len(devices))), devices[rank]
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
    observer = collective.graph_observer
    retained["observer"] = observer
    stream = torch.npu.Stream(device=device_index)
    retained["streams"].append(stream)
    # The separate stream is also the eager owner stream, so the first warmup
    # claims exactly the same stream used for capture and later eager calls.
    stream.wait_stream(torch.npu.current_stream(device_index))
    captures = []
    bundles = retained.setdefault("capture_bundles", [])
    baseline_graphs = []

    def synchronize():
        torch.npu.synchronize(device_index)

    def submit(mode, source):
        if mode == "custom":
            result = collective.all_reduce(source, eager._fallback)
            retained["tensors"].append(result)
        else:
            result = source.clone()
            retained["tensors"].append(result)
            dist.all_reduce(result, op=dist.ReduceOp.SUM, group=hccl_group)
        return result

    def barrier(phase):
        gather(connection, phase, {})
        dist.barrier(group=cpu_group)

    with torch.npu.stream(stream):
        for geometry_index, geometry in enumerate(config["geometries"]):
            for count in CAPTURE_COUNTS:
                prefix = f"geometry-{geometry_index}-ops-{count}"
                graphs, inputs, outputs = {}, {}, {}
                for mode in ("custom", "hccl"):
                    sources = [eager._input(torch, geometry, rank, index, device) for index in range(count)]
                    retained["tensors"].extend(sources)
                    inputs[mode] = sources
                    barrier(prefix + "-warmup-" + mode)
                    for _ in range(WARMUP):
                        warmed = [submit(mode, source) for source in sources]
                    synchronize()
                    for index, result in enumerate(warmed):
                        eager._verify(result, expected(geometry["elements"], len(members), index, geometry["dtype"]))
                    graph = torch.npu.NPUGraph()
                    graphs[mode] = graph
                    retained["graphs"].append(graph)
                    if mode == "hccl":
                        baseline_graphs.append(graph)
                    barrier(prefix + "-capture-" + mode)
                    outputs[mode] = _capture(
                        torch,
                        graph,
                        stream,
                        sources,
                        lambda source, mode=mode: submit(mode, source),
                        observer if mode == "custom" else None,
                        retained,
                    )
                    synchronize()
                    if len({tensor.data_ptr() for tensor in sources + outputs[mode]}) != count * 2:
                        raise AssertionError("capture requires distinct independent inputs and outputs")
                histories, interleaves = [], []
                for replay in range(CHANGED_REPLAYS):
                    cases = [_case(replay, index) for index in range(count)]
                    hashes = {}
                    for mode in ("custom", "hccl"):
                        for source, case in zip(inputs[mode], cases, strict=True):
                            replacement = eager._input(torch, geometry, rank, case, device)
                            retained["tensors"].append(replacement)
                            source.copy_(replacement)
                        synchronize()
                        barrier(prefix + f"-replay-{replay}-" + mode)
                        _replay(graphs[mode], observer if mode == "custom" else None)
                        synchronize()
                        hashes[mode] = [
                            eager._verify(result, expected(geometry["elements"], len(members), case, geometry["dtype"]))
                            for result, case in zip(outputs[mode], cases, strict=True)
                        ]
                        for source, case in zip(inputs[mode], cases, strict=True):
                            eager._verify(source, pattern(geometry["elements"], rank, case, geometry["dtype"]))
                    histories.append(dict(replay=replay, cases=cases, custom=hashes["custom"], hccl=hashes["hccl"]))
                    # An eager call after every replay checks that neither odd
                    # capture counts nor host submissions alter graph addresses.
                    case = 40 + replay
                    source = eager._input(torch, geometry, rank, case, device)
                    retained["tensors"].append(source)
                    interleaved = {}
                    for mode in ("custom", "hccl"):
                        barrier(prefix + f"-eager-{replay}-" + mode)
                        result = submit(mode, source)
                        synchronize()
                        interleaved[mode] = eager._verify(
                            result, expected(geometry["elements"], len(members), case, geometry["dtype"])
                        )
                    interleaves.append(dict(case=case, **interleaved))
                timings = []
                for pair in range(TIMING_PAIRS):
                    for mode in ("custom", "hccl") if pair % 2 == 0 else ("hccl", "custom"):
                        barrier(prefix + f"-timed-{pair}-" + mode)
                        started = time.monotonic_ns()
                        for _ in range(TIMING_REPLAYS):
                            _replay(graphs[mode], observer if mode == "custom" else None)
                        submitted = time.monotonic_ns()
                        synchronize()
                        completed = time.monotonic_ns()
                        for index, result in enumerate(outputs[mode]):
                            wanted = expected(
                                geometry["elements"], len(members), _case(CHANGED_REPLAYS - 1, index), geometry["dtype"]
                            )
                            eager._verify(result, wanted)
                        timings.append(
                            dict(
                                pair=pair,
                                mode=mode,
                                count=TIMING_REPLAYS * count,
                                started_ns=started,
                                submitted_ns=submitted,
                                completed_ns=completed,
                            )
                        )
                captures.append(
                    dict(
                        **geometry,
                        operations=count,
                        changed_replays=histories,
                        eager_interleaves=interleaves,
                        timings=timings,
                    )
                )
                bundles.append(dict(geometry=geometry, operations=count, graphs=graphs, inputs=inputs, outputs=outputs))
        final_recapture_history = []
        for direction in FINAL_SWEEPS:
            indices = range(len(bundles) - 1, -1, -1) if direction == "reverse" else range(len(bundles))
            for index in indices:
                bundle = bundles[index]
                geometry, count = bundle["geometry"], bundle["operations"]
                cases = [_case(CHANGED_REPLAYS - 1, offset) for offset in range(count)]
                hashes = {}
                for mode in ("custom", "hccl"):
                    # Invalidate prior correct results so a stale executor that
                    # writes a newer graph's addresses cannot pass as a no-op.
                    for result in bundle["outputs"][mode]:
                        result.zero_()
                    synchronize()
                    barrier(f"final-{direction}-capture-{index}-{mode}")
                    _replay(bundle["graphs"][mode], observer if mode == "custom" else None)
                    synchronize()
                    hashes[mode] = [
                        eager._verify(result, expected(geometry["elements"], len(members), case, geometry["dtype"]))
                        for result, case in zip(bundle["outputs"][mode], cases, strict=True)
                    ]
                    for source, case in zip(bundle["inputs"][mode], cases, strict=True):
                        eager._verify(source, pattern(geometry["elements"], rank, case, geometry["dtype"]))
                final_recapture_history.append(
                    dict(
                        direction=direction,
                        capture_index=index,
                        **geometry,
                        operations=count,
                        cases=cases,
                        custom=hashes["custom"],
                        hccl=hashes["hccl"],
                    )
                )
        if collective.fallbacks:
            raise AssertionError("graph geometry unexpectedly fell back")
        barrier("graphs-completed")
        observer.retire(synchronize)
        if observer.retired is not True:
            raise RuntimeError("custom graph observer did not prove retirement")
        observed = observer.snapshot()
        if observed != observer_receipt(config):
            raise RuntimeError("graph observer submission accounting differs from matched gate schedule")
        # HCCL graphs are independent of the custom observer and must also be
        # reset explicitly, after successful drain and before any owner close.
        for graph in baseline_graphs:
            graph.reset()
        barrier("all-graphs-retired")
        collective.close(graphs_retired=True)
    gather(connection, "facade-closed", dict(rank=rank, closed=True))
    dist.destroy_process_group(hccl_group)
    dist.destroy_process_group(cpu_group)
    retained["groups"] = []
    return dict(
        rank=rank,
        physical_device=device_index,
        config_sha256=eager.identity(config),
        passed=True,
        ownership_released=True,
        graphs_retired=True,
        fallback_calls=0,
        graph_counts=dict(custom=len(captures), hccl=len(baseline_graphs)),
        graph_observer=observed,
        final_recapture_history=final_recapture_history,
        captures=captures,
    )


def comparisons(config, ranks):
    if type(ranks) is not list or len(ranks) != len(config["physical_devices"]):
        raise ValueError("one graph report per assigned rank is required")
    result = []
    geometries = [dict(**geometry, operations=count) for geometry in config["geometries"] for count in CAPTURE_COUNTS]
    for index, geometry in enumerate(geometries):
        samples = []
        for rank, report in enumerate(ranks):
            required = dict(
                rank=rank,
                physical_device=config["physical_devices"][rank],
                config_sha256=eager.identity(config),
                passed=True,
                ownership_released=True,
                graphs_retired=True,
                fallback_calls=0,
                graph_counts=dict(custom=len(geometries), hccl=len(geometries)),
                graph_observer=observer_receipt(config),
            )
            if (
                type(report) is not dict
                or set(report) != set(required) | {"captures", "final_recapture_history"}
                or any(
                    type(report.get(key)) is not type(value) or report[key] != value for key, value in required.items()
                )
            ):
                raise ValueError("graph rank identity, ownership or retirement differs")
            if json.dumps(report["final_recapture_history"], sort_keys=True, allow_nan=False) != json.dumps(
                final_history(config), sort_keys=True, allow_nan=False
            ):
                raise ValueError("final graph recapture replay order, coverage or numerical result differs")
            captures = report["captures"]
            if type(captures) is not list or len(captures) != len(geometries):
                raise ValueError("graph shape coverage differs")
            capture = captures[index]
            if (
                type(capture) is not dict
                or set(capture) != set(geometry) | {"changed_replays", "eager_interleaves", "timings"}
                or any(
                    type(capture.get(key)) is not type(value) or capture[key] != value
                    for key, value in geometry.items()
                )
            ):
                raise ValueError("graph operation/geometry identity differs")
            histories, interleaves = [], []
            for replay in range(CHANGED_REPLAYS):
                cases = [_case(replay, offset) for offset in range(geometry["operations"])]
                checksums = [_checksum(geometry, len(ranks), case) for case in cases]
                histories.append(dict(replay=replay, cases=cases, custom=checksums, hccl=checksums))
                checksum = _checksum(geometry, len(ranks), 40 + replay)
                interleaves.append(dict(case=40 + replay, custom=checksum, hccl=checksum))
            if capture["changed_replays"] != histories or capture["eager_interleaves"] != interleaves:
                raise ValueError("graph changed-input replay or eager coverage differs")
            timings = capture["timings"]
            if (
                type(timings) is not list
                or len(timings) != TIMING_PAIRS * 2
                or any(type(sample) is not dict for sample in timings)
                or {(sample.get("pair"), sample.get("mode"), sample.get("count")) for sample in timings}
                != {
                    (pair, mode, TIMING_REPLAYS * geometry["operations"])
                    for pair in range(TIMING_PAIRS)
                    for mode in ("custom", "hccl")
                }
            ):
                raise ValueError("graph timing coverage differs")
            samples.append(dict(rank=rank, timings=timings))
        result.append(dict(**geometry, comparison=summarize(samples)))
    return result


def worker(config, output, rank, connection):
    retained = dict(facade=None, observer=None, groups=[], tensors=[], graphs=[], streams=[])
    try:
        torch, dist = eager._backend()
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
        traceback.print_exception(error, file=sys.stderr)
        chain, cause = [], error
        while cause is not None:
            chain.append(type(cause).__name__)
            cause = cause.__cause__
        failure = dict(
            rank=rank, pid=os.getpid(), passed=False, owners_retained=True, automatic_cleanup=False, error_types=chain
        )
        with contextlib.suppress(OSError):
            (output / f"rank{rank}-failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            send(connection, dict(phase="error", payload=failure))
        while True:
            signal.pause()


def run(config, output):
    config, output = validate(config), Path(output).absolute()
    if Path(config["output_root"]).exists():
        raise ValueError("graph facade receipts require a fresh private output_root")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    path = output / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    connections, processes = [], []
    try:
        for rank in range(len(config["physical_devices"])):
            parent, child = socket.socketpair()
            connections.append(parent)
            parent.settimeout(CONTROL_TIMEOUT_SECONDS)
            command = [
                sys.executable,
                "-m",
                "tools.sidecard.npu_collective_graph_probe",
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
                raise RuntimeError("graph collective rank control phase failed")
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
                        raise ValueError("graph rank result receipt differs")
                    reports.append(json.loads(raw))
                break
        result = dict(
            passed=True,
            ranks=reports,
            physical_devices=config["physical_devices"],
            algorithm=config["algorithm"],
            comparison=comparisons(config, reports),
            graphs_retired=True,
            numerical_scope="synthetic exactly representable finite graph inputs; not model qualification",
            timing_scope="same captured independent SUM count; custom vs clone+Torch HCCL; CPU barriers excluded",
        )
        for process in processes:
            if process.wait(timeout=10):
                raise RuntimeError("completed graph worker did not exit successfully")
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    except BaseException:
        (output / "quarantine.json").write_text(
            json.dumps(
                dict(pids=[process.pid for process in processes], owners_retained=True, automatic_kill=False), indent=2
            )
            + "\n"
        )
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
            parser.error("worker requires assigned rank and private control FD")
        worker(config, args.output.absolute(), args.worker, socket.socket(fileno=args.fd))


if __name__ == "__main__":
    main()
