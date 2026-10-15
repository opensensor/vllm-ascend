# SPDX-License-Identifier: Apache-2.0
"""Coordinated general-N device qualification; no model or automatic recovery."""

import argparse
import contextlib
import ctypes as c
import hashlib
import json
import os
import secrets
import signal
import socket
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

from .npu_collective_acl import NpuCollectiveOwner
from .npu_collective_schedule import BinomialSchedule
from .pair_collective_acl import _bind
from .pair_collective_probe import receive, send
from .pcie.runtime import Runtime, Stream

CONTROL_TIMEOUT_SECONDS = 120


def gather(connection, phase, payload):
    send(connection, dict(phase=phase, payload=payload))
    packet = receive(connection)
    if packet.get("phase") != phase or type(packet.get("payloads")) is not list:
        raise RuntimeError("collective control phase differs")
    return packet["payloads"]


def validate(config):
    required = {"physical_chips", "elements", "dtype", "acl_library", "acl_sha256", "native_library", "native_sha256"}
    optional = {"reference_library", "reference_sha256", "warmup", "pairs", "batch", "cases", "stress", "algorithm"}
    if type(config) is not dict or set(config) - required - optional or not required <= set(config):
        raise ValueError("invalid general-N probe fields")
    chips = config["physical_chips"]
    if (
        type(chips) is not list
        or not 1 <= len(chips) <= 4
        or any(type(i) is not int or i not in range(4) for i in chips)
        or len(set(chips)) != len(chips)
    ):
        raise ValueError("declare distinct physical chips in this authorized four-chip lab")
    schedule = BinomialSchedule(
        tuple(range(len(chips))), config["dtype"], config["elements"], algorithm=config.get("algorithm", "binomial")
    )
    if schedule.elements > 16 * 1024 * 1024:
        raise ValueError("native element limit exceeded")
    result = dict(config)
    result["algorithm"] = schedule.algorithm
    for name, default, upper in (("warmup", 10, 100), ("pairs", 7, 50), ("batch", 30, 60), ("cases", 3, 20)):
        value = config.get(name, default)
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError("invalid " + name)
        result[name] = value
    if ("reference_library" in config) != ("reference_sha256" in config):
        raise ValueError("raw HCCL reference requires its SHA256")
    for key in ("acl", "native", "reference"):
        if key + "_library" not in config:
            continue
        if type(config[key + "_library"]) is not str or type(config[key + "_sha256"]) is not str:
            raise ValueError(key + " library requires an explicit path and SHA256")
        path = Path(config[key + "_library"])
        if (
            not path.is_absolute()
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != config[key + "_sha256"]
        ):
            raise ValueError(key + " library path or SHA256 differs")
    result["stress"] = config.get("stress", True)
    if type(result["stress"]) is not bool:
        raise ValueError("stress must be boolean")
    return result


def pattern(elements, rank, case, dtype):
    # Exactly representable finite values at all four available rank counts.
    # This tests complete transport bytes, not the broad model precision policy.
    code = "e" if dtype == "float16" else "f"
    return struct.pack("<" + code * elements, *((i % 31 - 15) + rank / 4 + case / 8 for i in range(elements)))


def expected(elements, ranks, case, dtype):
    code = "e" if dtype == "float16" else "f"
    return struct.pack(
        "<" + code * elements,
        *((i % 31 - 15) * ranks + sum(range(ranks)) / 4 + ranks * case / 8 for i in range(elements)),
    )


def reference(path):
    library = c.CDLL(path)
    for name in ("abi", "root_info_bytes"):
        function = getattr(library, "sidecard_hccl_group_" + name)
        function.argtypes, function.restype = [], c.c_uint64
    if library.sidecard_hccl_group_abi() != 1:
        raise ValueError("raw HCCL group ABI differs")
    ptr = c.c_void_p
    for name, arguments in (
        ("get_root_info", [ptr, c.c_uint64]),
        ("create", [ptr, c.c_uint64, c.c_uint32, c.c_uint32, c.POINTER(ptr)]),
        ("all_reduce", [ptr, ptr, ptr, c.c_uint64, c.c_uint32, ptr]),
        ("destroy", [c.POINTER(ptr), c.c_uint32]),
    ):
        _bind(library, "sidecard_hccl_group_" + name, arguments)
    return library


def checked(function, *args):
    status = function(*args)
    if status:
        raise RuntimeError(f"raw reference operation failed: {status}")


def summarize(ranks):
    if (
        type(ranks) is not list
        or not ranks
        or any(type(p) is not dict or type(p.get("rank")) is not int for p in ranks)
        or {p["rank"] for p in ranks} != set(range(len(ranks)))
    ):
        raise ValueError("exactly one report per rank is required")
    for packet in ranks:
        values = packet.get("timings")
        if type(values) is not list or not values:
            raise ValueError("nonempty timing samples are required")
        for value in values:
            fields = {"pair", "mode", "count", "started_ns", "submitted_ns", "completed_ns"}
            if (
                type(value) is not dict
                or set(value) != fields
                or type(value["mode"]) is not str
                or value["mode"] not in ("custom", "hccl")
                or any(type(value[k]) is not int or value[k] < 0 for k in fields - {"mode"})
                or value["count"] == 0
            ):
                raise ValueError("invalid host timing fields")
    indexed = [{(s["pair"], s["mode"]): s for s in p["timings"]} for p in ranks]
    keys = set(indexed[0])
    if any(set(values) != keys or len(values) != len(p["timings"]) for values, p in zip(indexed, ranks, strict=True)):
        raise ValueError("timing samples do not match across ranks")
    samples = []
    for pair, mode in sorted(keys):
        values = [rank[(pair, mode)] for rank in indexed]
        count = values[0]["count"]
        if (
            type(count) is not int
            or count <= 0
            or any(v["count"] != count or not v["started_ns"] <= v["submitted_ns"] <= v["completed_ns"] for v in values)
        ):
            raise ValueError("invalid rank timing/count")
        starts, finishes = [v["started_ns"] for v in values], [v["completed_ns"] for v in values]
        samples.append(
            dict(
                pair=pair,
                mode=mode,
                count=count,
                completed_batch_us_per_call=(max(finishes) - min(starts)) / count / 1000,
                arrival_spread_us=(max(starts) - min(starts)) / 1000,
                rank_submission_us_per_call=[(v["submitted_ns"] - v["started_ns"]) / count / 1000 for v in values],
            )
        )
    modes = {}
    for mode in sorted({s["mode"] for s in samples}):
        values = [s["completed_batch_us_per_call"] for s in samples if s["mode"] == mode]
        modes[mode] = dict(
            median_us=statistics.median(values), min_us=min(values), max_us=max(values), samples_us=values
        )
    return dict(
        clock="shared-host CLOCK_MONOTONIC",
        scope="completed batch including host dispatch/arrival spread",
        pure_transit_measured=False,
        confidence_interval_computed=False,
        samples=samples,
        modes=modes,
    )


def worker(config, output, rank, connection):
    """Keep every rank's SDK/code owner alive after uncertain partial failure."""
    owner, allocations = None, []
    try:
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0,1,2,3":
            raise ValueError("probe requires authorized four-chip visibility")
        chips, dtype, elements = config["physical_chips"], config["dtype"], config["elements"]
        ranks = len(chips)
        schedule = BinomialSchedule(tuple(range(ranks)), dtype, elements, algorithm=config["algorithm"])
        runtime = Runtime("ascend", config["acl_library"], chips[rank])
        links = {peer: "hccs" if chips[rank] // 2 == chips[peer] // 2 else "pcie" for peer in schedule.peers(rank)}
        # This mapping is specific to the documented lab: physical 0/1 and 2/3
        # are sibling chips. The reusable owner accepts explicit link metadata.
        nonce = gather(connection, "nonce", secrets.token_hex(32) if rank == 0 else None)[0]
        owner = NpuCollectiveOwner(
            schedule,
            rank,
            nonce,
            links,
            config["acl_library"],
            config["native_library"],
            acl_sha256=config["acl_sha256"],
            native_sha256=config["native_sha256"],
            device_index=chips[rank],
        )
        hello = gather(connection, "hello", dict(rank=rank, physical_chip=chips[rank], bare_pid=owner.bare_pid()))
        if [h["physical_chip"] for h in hello] != chips or [h["rank"] for h in hello] != list(range(ranks)):
            raise ValueError("rank/device mapping differs")
        exports = owner.prepare({peer: hello[peer]["bare_pid"] for peer in owner.peers})
        published = gather(connection, "exports", exports)
        owner.connect({peer: published[peer][str(rank)] for peer in owner.peers})
        gather(connection, "connected", dict(schedule_sha256=schedule.identity))
        stream = Stream(runtime, owner.stream.value)
        upload, download = (runtime.allocate(owner.nbytes, host=True) for _ in range(2))
        source, destination = (runtime.allocate(owner.nbytes) for _ in range(2))
        allocations.extend((upload, download, source, destination))

        def write(case):
            data = pattern(elements, rank, case, dtype)
            c.memmove(upload.pointer, data, len(data))
            runtime.copy(source.pointer, upload.pointer, len(data), "h2d", stream)
            owner.synchronize()

        def read():
            runtime.copy(download.pointer, destination.pointer, owner.nbytes, "d2h", stream)
            owner.synchronize()
            return c.string_at(download.pointer, owner.nbytes)

        receipts = []
        for case in range(config["cases"]):
            write(case)
            gather(connection, f"sum-start-{case}", {})
            owner.enqueue(source.pointer, destination.pointer)
            owner.synchronize()
            data = read()
            if data != expected(elements, ranks, case, dtype):
                raise AssertionError("general-N complete result differs from exact finite CPU reference")
            receipts.append(dict(case=case, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            gathered = gather(connection, f"sum-verified-{case}", receipts[-1])
            if any(p != receipts[-1] for p in gathered):
                raise AssertionError("broadcast did not preserve identical result bytes on all ranks")

        stress = []
        if config["stress"]:
            count, windows = 30, 3
            history_bytes = owner.nbytes * count
            history_upload, history_download = (runtime.allocate(history_bytes, host=True) for _ in range(2))
            history, history_source = (runtime.allocate(history_bytes) for _ in range(2))
            allocations.extend((history_upload, history_download, history, history_source))
            for window in range(windows):
                first = config["cases"] + window * count
                data = b"".join(pattern(elements, rank, first + i, dtype) for i in range(count))
                c.memmove(history_upload.pointer, data, len(data))
                runtime.copy(history_source.pointer, history_upload.pointer, history_bytes, "h2d", stream)
                owner.synchronize()
                gather(connection, f"stress-start-{window}", {})
                if rank == window % ranks:
                    time.sleep(0.001)
                for i in range(count):
                    offset = i * owner.nbytes
                    # Distinct LOCAL input/output pointers for every call test
                    # SDK metadata snapshotting before any host drain.
                    owner.enqueue(history_source.pointer + offset, history.pointer + offset)
                owner.synchronize()
                runtime.copy(history_download.pointer, history.pointer, history_bytes, "d2h", stream)
                owner.synchronize()
                data = c.string_at(history_download.pointer, history_bytes)
                if data != b"".join(expected(elements, ranks, first + i, dtype) for i in range(count)):
                    raise AssertionError("changed-input group history differs")
                stress.append(
                    dict(window=window, calls=count, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                )
                gather(connection, f"stress-verified-{window}", stress[-1])

        library, comm, reference_receipts = None, c.c_void_p(), []
        if "reference_library" in config:
            library = reference(config["reference_library"])
            size = library.sidecard_hccl_group_root_info_bytes()
            if not 0 < size <= 16000:
                raise ValueError("HCCL root info exceeds control budget")
            root = c.create_string_buffer(size)
            if rank == 0:
                checked(library.sidecard_hccl_group_get_root_info, root, size)
            roots = gather(connection, "hccl-root", root.raw.hex() if rank == 0 else None)
            if rank:
                c.memmove(root, bytes.fromhex(roots[0]), size)
            checked(library.sidecard_hccl_group_create, root, size, ranks, rank, c.byref(comm))
            gather(connection, "hccl-connected", {})
            for case in range(config["cases"]):
                write(case)
                gather(connection, f"hccl-sum-start-{case}", {})
                checked(
                    library.sidecard_hccl_group_all_reduce,
                    comm,
                    source.pointer,
                    destination.pointer,
                    elements,
                    int(dtype == "float16"),
                    owner.stream,
                )
                owner.synchronize()
                data = read()
                if data != expected(elements, ranks, case, dtype):
                    raise AssertionError("HCCL finite reference differs")
                reference_receipts.append(dict(case=case, sha256=hashlib.sha256(data).hexdigest()))
                gather(connection, f"hccl-sum-verified-{case}", reference_receipts[-1])
        write(config["cases"] - 1)

        def submit(mode):
            if mode == "custom":
                owner.enqueue(source.pointer, destination.pointer)
            else:
                checked(
                    library.sidecard_hccl_group_all_reduce,
                    comm,
                    source.pointer,
                    destination.pointer,
                    elements,
                    int(dtype == "float16"),
                    owner.stream,
                )

        modes = ("custom", "hccl") if library else ("custom",)
        for mode in modes:
            gather(connection, "warmup-" + mode, {})
            for _ in range(config["warmup"]):
                submit(mode)
            owner.synchronize()
        timings = []
        for pair in range(config["pairs"]):
            for mode in modes if pair % 2 == 0 else tuple(reversed(modes)):
                gather(connection, f"timed-{pair}-{mode}", {})
                started = time.monotonic_ns()
                for _ in range(config["batch"]):
                    submit(mode)
                submitted = time.monotonic_ns()
                owner.synchronize()
                completed = time.monotonic_ns()
                if read() != expected(elements, ranks, config["cases"] - 1, dtype):
                    raise AssertionError("timed output differs")
                timings.append(
                    dict(
                        pair=pair,
                        mode=mode,
                        count=config["batch"],
                        started_ns=started,
                        submitted_ns=submitted,
                        completed_ns=completed,
                    )
                )
        gather(connection, "completed", {})
        if library:
            checked(library.sidecard_hccl_group_destroy, c.byref(comm), 1)
        drained = gather(connection, "drained", owner.drain(graphs_retired=True))
        closed = gather(connection, "imports-closed", owner.release_imports(drained))
        owner.close(closed, graphs_retired=True)
        for allocation in allocations:
            allocation.close()
        result = dict(
            rank=rank,
            physical_chip=chips[rank],
            passed=True,
            ownership_released=True,
            fp_cases=receipts,
            changed_input_stress=stress,
            hccl_cases=reference_receipts,
            timings=timings,
            custom_calls=owner.calls,
        )
        (output / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
        gather(
            connection,
            "done",
            dict(
                rank=rank,
                passed=True,
                result_sha256=hashlib.sha256((output / f"rank{rank}.json").read_bytes()).hexdigest(),
            ),
        )
    except BaseException as error:
        chain, cause = [], error
        while cause is not None:
            chain.append(dict(type=type(cause).__name__, message=str(cause)))
            cause = cause.__cause__
        result = dict(
            rank=rank,
            passed=False,
            pid=os.getpid(),
            owners_retained=True,
            state=getattr(owner, "state", None),
            error_chain=chain,
        )
        (output / f"rank{rank}-failure.json").write_text(json.dumps(result, indent=2) + "\n")
        with contextlib.suppress(OSError):
            send(connection, dict(phase="error", payload=result))
        while True:
            signal.pause()


def run(config, output):
    config, output = validate(config), Path(output).absolute()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    connections, processes = [], []
    try:
        for rank in range(len(config["physical_chips"])):
            parent, child = socket.socketpair()
            connections.append(parent)
            parent.settimeout(CONTROL_TIMEOUT_SECONDS)
            command = [
                sys.executable,
                "-m",
                "tools.sidecard.npu_collective_probe",
                "--config",
                str(config_path),
                "--output",
                str(output),
                "--worker",
                str(rank),
                "--fd",
                str(child.fileno()),
            ]
            try:
                with (output / f"rank{rank}.log").open("w") as log:
                    process = subprocess.Popen(
                        command, pass_fds=(child.fileno(),), stdout=log, stderr=subprocess.STDOUT
                    )
                processes.append(process)
                (output / "processes.json").write_text(
                    json.dumps(dict(pids=[p.pid for p in processes]), indent=2) + "\n"
                )
            finally:
                child.close()
    except BaseException:
        (output / "quarantine.json").write_text(
            json.dumps(dict(pids=[p.pid for p in processes], owners_retained=True, automatic_kill=False), indent=2)
            + "\n"
        )
        for connection in connections:
            connection.close()
        raise
    (output / "processes.json").write_text(json.dumps(dict(pids=[p.pid for p in processes]), indent=2) + "\n")
    try:
        while True:
            packets = [receive(connection) for connection in connections]
            phase = packets[0].get("phase")
            if phase == "error" or any(p.get("phase") != phase for p in packets):
                raise RuntimeError("general-N rank phase failed: " + str([p.get("phase") for p in packets]))
            with (output / "progress.jsonl").open("a") as log:
                log.write(json.dumps(dict(phase=phase, at_ns=time.monotonic_ns())) + "\n")
            payloads = [packet["payload"] for packet in packets]
            for connection in connections:
                send(connection, dict(phase=phase, payloads=payloads))
            if phase == "done":
                ranks = []
                for rank, packet in enumerate(payloads):
                    raw = (output / f"rank{rank}.json").read_bytes()
                    expected_packet = dict(rank=rank, passed=True, result_sha256=hashlib.sha256(raw).hexdigest())
                    if (
                        type(packet) is not dict
                        or set(packet) != set(expected_packet)
                        or any(type(packet[k]) is not type(v) or packet[k] != v for k, v in expected_packet.items())
                    ):
                        raise ValueError("rank result receipt differs")
                    report = json.loads(raw)
                    expected_report = dict(
                        rank=rank, physical_chip=config["physical_chips"][rank], passed=True, ownership_released=True
                    )
                    if type(report) is not dict or any(
                        type(report.get(k)) is not type(v) or report[k] != v for k, v in expected_report.items()
                    ):
                        raise ValueError("rank result status, identity or ownership proof differs")
                    ranks.append(report)
                break
        for process in processes:
            if process.wait(timeout=10):
                raise RuntimeError("completed worker did not exit successfully")
        result = dict(
            passed=True,
            ranks=ranks,
            physical_chips=config["physical_chips"],
            dtype=config["dtype"],
            elements=config["elements"],
            algorithm=config["algorithm"],
            comparison=summarize(ranks),
            numerical_scope="synthetic exactly representable finite inputs; not model numerical qualification",
            graphs_qualified=False,
            notify_timeout_units_verified=False,
        )
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    except BaseException:
        (output / "quarantine.json").write_text(
            json.dumps(dict(pids=[p.pid for p in processes], owners_retained=True, automatic_kill=False), indent=2)
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
        result = run(config, args.output)
        print(json.dumps(dict(passed=result["passed"], ranks=len(result["ranks"]))))
    else:
        if args.fd is None or args.worker not in range(len(config["physical_chips"])):
            parser.error("worker requires its assigned rank and private control FD")
        worker(config, args.output, args.worker, socket.socket(fileno=args.fd))


if __name__ == "__main__":
    main()
