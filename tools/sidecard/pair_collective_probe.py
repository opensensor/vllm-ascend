# SPDX-License-Identifier: Apache-2.0
"""Explicit, coordinated two-owner capability and FP32 SUM benchmark.

Both Duos (physical chips 0..3) are authorized for this follow-up. Each run uses
only its declared two-chip pair. No Torch, model, card reset, global setting or
automatic process kill is used. Workers retain owners on uncertain failure.
"""

import argparse
import array
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

from .pair_add import LocalAddSpec
from .pair_collective import BOUNDARIES, PairConfig, PairEndpoint
from .pair_collective_acl import AclPairBackend, _bind
from .pcie.runtime import Runtime, Stream

FRAME_BYTES = 65536
CONTROL_TIMEOUT_SECONDS = 120
PROBE_PROTOCOL = "sidecard-pair-probe-v1"


def send(connection, value):
    data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if not 0 < len(data) <= FRAME_BYTES:
        raise ValueError("probe control frame too large")
    connection.sendall(struct.pack("!I", len(data)) + data)


def receive(connection):
    def read(size):
        data = bytearray()
        while len(data) < size:
            block = connection.recv(size - len(data))
            if not block:
                raise EOFError("probe peer closed the control channel")
            data.extend(block)
        return data

    size = struct.unpack("!I", read(4))[0]
    if not 0 < size <= FRAME_BYTES:
        raise ValueError("invalid probe frame size")
    result = json.loads(read(size))
    if type(result) is not dict:
        raise ValueError("probe frame must be an object")
    return result


def exchange(connection, phase, payload):
    send(connection, dict(phase=phase, payload=payload))
    packet = receive(connection)
    if packet.get("phase") != phase or packet.get("abort"):
        raise RuntimeError("coordinated probe phase differs or was aborted")
    return packet["payload"]


def validate(config):
    required = {"physical_chips", "acl_library", "acl_sha256", "native_library", "native_sha256", "rows"}
    optional = {
        "hccl_reference",
        "hccl_reference_sha256",
        "warmup",
        "pairs",
        "batch",
        "cases",
        "skew_us",
        "stress",
        "exchange_mode",
        "local_add",
        "handshake",
    }
    if type(config) is not dict or set(config) - required - optional or not required <= set(config):
        raise ValueError("invalid probe configuration fields")
    chips = config["physical_chips"]
    if (
        type(chips) is not list
        or len(chips) != 2
        or any(type(n) is not int or n not in range(4) for n in chips)
        or len(set(chips)) != 2
    ):
        raise ValueError("declare exactly two distinct authorized physical chips from 0..3")
    result = dict(config)
    exchange_mode = config.get("exchange_mode", "push")
    if exchange_mode not in ("push", "remote-read"):
        raise ValueError("invalid exchange_mode")
    if exchange_mode != "push":
        raise ValueError("remote-read is disabled after its first 310P vector reduction timed out")
    result["exchange_mode"] = exchange_mode
    handshake = config.get("handshake", "consumed")
    if handshake not in ("consumed", "ping-pong"):
        raise ValueError("invalid handshake")
    if handshake == "ping-pong" and "local_add" in config:
        raise ValueError("ping-pong currently requires ACLNN")
    result["handshake"] = handshake
    if "local_add" in config:
        raw = config["local_add"]
        if type(raw) is not dict or set(raw) != {"library", "library_sha256", "binary", "binary_sha256", "blocks"}:
            raise ValueError("invalid local_add configuration")
        LocalAddSpec(**raw)
        if config["rows"] not in (1, 4, 32):
            raise ValueError("local add requires a supported fixed-row binary")
    skew = config.get("skew_us", 0)
    if type(skew) is not int or skew not in (0, 50, 250, 1000):
        raise ValueError("skew_us must be 0, 50, 250 or 1000")
    result["skew_us"] = skew
    stress = config.get("stress", True)
    if type(stress) is not bool:
        raise ValueError("stress requires a boolean")
    result["stress"] = stress
    for key, default, lower, upper in (
        ("rows", 1, 1, 32),
        ("warmup", 10, 1, 100),
        ("pairs", 5, 1, 50),
        ("batch", 30, 1, 60),
        ("cases", 3, 1, 30),
    ):
        value = config.get(key, default)
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError("invalid " + key)
        result[key] = value
    if ("hccl_reference" in config) != ("hccl_reference_sha256" in config):
        raise ValueError("HCCL reference requires its SHA256")
    for key in ("acl_library", "native_library", "hccl_reference"):
        if key not in config:
            continue
        path = Path(config[key])
        sha_key = key.replace("_library", "") + "_sha256" if key.endswith("_library") else key + "_sha256"
        if (
            not path.is_absolute()
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != config[sha_key]
        ):
            raise ValueError(key + " path or SHA256 differs")
    return result


def pattern(elements, rank, case):
    if case == 0:
        return array.array("f", ((i % 31 - 15) + rank / 4 for i in range(elements))).tobytes()
    # Normal finite FP32 values with varied signs/exponents/mantissas, including
    # cancellation and rounding-sensitive pairs. Avoid subnormals/nonfinites;
    # their exact stack policy requires separate qualification.
    return array.array(
        "I",
        (
            (((i + rank) % 2) << 31)
            | (((i * 5 + case * 7 + rank * (20 if i % 3 == 0 else 0)) % 160 + 40) << 23)
            | ((i * 1103515245 + case * 12345 + rank * 1907) & 0x7FFFFF)
            for i in range(elements)
        ),
    ).tobytes()


def expected_sum(elements, case):
    left, right = (array.array("f") for _ in range(2))
    left.frombytes(pattern(elements, 0, case))
    right.frombytes(pattern(elements, 1, case))
    return array.array("f", (a + b for a, b in zip(left, right, strict=True))).tobytes()


def hccl_reference(path):
    library = c.CDLL(str(path))
    for name in ("sidecard_hccl_abi", "sidecard_hccl_root_info_bytes"):
        function = getattr(library, name)
        function.argtypes, function.restype = [], c.c_uint64
    if library.sidecard_hccl_abi() != 1:
        raise ValueError("unsupported HCCL reference ABI")
    pointer = c.c_void_p
    _bind(library, "sidecard_hccl_get_root_info", [pointer, c.c_uint64])
    _bind(library, "sidecard_hccl_create", [pointer, c.c_uint64, c.c_uint32, c.POINTER(pointer)])
    _bind(library, "sidecard_hccl_all_reduce", [pointer, pointer, pointer, c.c_uint64, pointer])
    _bind(library, "sidecard_hccl_destroy", [c.POINTER(pointer), c.c_uint32])
    return library


def checked(function, *args):
    status = function(*args)
    if status:
        raise RuntimeError(f"native probe operation failed: {status}")


def summarize(ranks):
    """Use matched host clocks; no device-event or pure-transit attribution."""
    if len(ranks) != 2 or {r["rank"] for r in ranks} != {0, 1}:
        raise ValueError("exactly two distinct rank reports are required")
    indexed = []
    for rank in ranks:
        rows = {}
        for value in rank["timings"]:
            key = (value["pair"], value["mode"])
            if key in rows or type(value["count"]) is not int or value["count"] <= 0:
                raise ValueError("duplicate or invalid timing sample")
            if not value["started_ns"] <= value["submitted_ns"] <= value["completed_ns"]:
                raise ValueError("invalid host timing order")
            rows[key] = value
        indexed.append(rows)
    if indexed[0].keys() != indexed[1].keys():
        raise ValueError("both ranks must cover the same paired samples")
    samples = []
    for key in sorted(indexed[0]):
        left, right = (rows[key] for rows in indexed)
        if left["count"] != right["count"]:
            raise ValueError("paired rank iteration counts differ")
        count = left["count"]
        starts = [left["started_ns"], right["started_ns"]]
        finish = max(left["completed_ns"], right["completed_ns"])
        samples.append(
            dict(
                pair=key[0],
                mode=key[1],
                count=count,
                arrival_spread_us=abs(starts[0] - starts[1]) / 1000,
                completed_batch_us_per_call=(finish - min(starts)) / count / 1000,
                completion_after_latest_start_us_per_call=(finish - max(starts)) / count / 1000,
                rank_submission_us_per_call=[
                    (v["submitted_ns"] - v["started_ns"]) / count / 1000 for v in (left, right)
                ],
            )
        )
    modes = {}
    for mode in sorted({s["mode"] for s in samples}):
        values = [s["completed_batch_us_per_call"] for s in samples if s["mode"] == mode]
        modes[mode] = dict(
            samples_us=values, median_us=statistics.median(values), min_us=min(values), max_us=max(values)
        )
    return dict(
        clock="shared-host CLOCK_MONOTONIC",
        scope="completed batch including host dispatch and rank arrival spread",
        pure_transit_measured=False,
        confidence_interval_computed=False,
        samples=samples,
        modes=modes,
    )


def worker(config, output, rank, connection):
    """Keep every SDK owner alive if a native operation fails or a peer vanishes."""
    owner = endpoint = None
    allocations = []
    try:
        visibility = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        if visibility != "0,1,2,3" and not (visibility == "0,1" and config["physical_chips"] == [0, 1]):
            raise ValueError("probe requires the explicitly authorized four-chip visibility")
        physical = config["physical_chips"][rank]
        runtime = Runtime("ascend", config["acl_library"], physical)
        local_add = LocalAddSpec(**config["local_add"]) if "local_add" in config else None
        owner = AclPairBackend(
            config["acl_library"],
            config["native_library"],
            acl_sha256=config["acl_sha256"],
            native_sha256=config["native_sha256"],
            local_add=local_add,
        )
        peer = exchange(connection, "hello", dict(pid=os.getpid(), bare_pid=owner.bare_pid(), physical_chip=physical))
        if peer["physical_chip"] != config["physical_chips"][1 - rank]:
            raise ValueError("peer device allocation differs")
        session = exchange(connection, "nonce", {"nonce": secrets.token_hex(32) if rank == 0 else None})
        nonce = session["nonce"] if rank == 1 else session["own_nonce"]
        link_mode = "hccs" if config["physical_chips"][0] // 2 == config["physical_chips"][1] // 2 else "pcie"
        endpoint = PairEndpoint(
            PairConfig(
                nonce,
                config["rows"],
                link_mode,
                config["exchange_mode"],
                reducer_identity=local_add.identity if local_add else "aclnn",
                notify_timeout_raw=30000,
                handshake=config["handshake"],
            ),
            rank,
            owner,
        )
        exported = endpoint.prepare(peer["bare_pid"])
        peer_export = exchange(connection, "exported", exported)
        connected = endpoint.connect(peer_export)
        endpoint.activate(exchange(connection, "connected", connected))
        stream = Stream(runtime, owner.stream.value)
        upload, download = (runtime.allocate(endpoint.config.nbytes, host=True) for _ in range(2))
        allocations.extend((upload, download))

        def write_input(case):
            data = pattern(endpoint.config.elements, rank, case)
            c.memmove(upload.pointer, data, len(data))
            runtime.copy(owner.input.value, upload.pointer, len(data), "h2d", stream)
            owner.synchronize()

        def read(pointer):
            runtime.copy(download.pointer, pointer.value, endpoint.config.nbytes, "d2h", stream)
            owner.synchronize()
            return c.string_at(download.pointer, endpoint.config.nbytes)

        write_input(0)
        exchange(connection, "copy-start", {})
        if config["exchange_mode"] == "push":
            runtime.copy(owner.imports["data"][0].value, owner.input.value, endpoint.config.nbytes, "d2d", stream)
        else:
            runtime.copy(owner.inbox.value, owner.imports["data"][0].value, endpoint.config.nbytes, "d2d", stream)
        owner.synchronize()
        exchange(connection, "copy-completed", {})
        if read(owner.inbox) != pattern(endpoint.config.elements, 1 - rank, 0):
            raise AssertionError("peer copy bytes differ")
        exchange(connection, "copy-verified", {})
        _bind(owner.acl, "aclrtRecordNotify", [c.c_void_p, c.c_void_p])
        _bind(owner.acl, "aclrtWaitAndResetNotify", [c.c_void_p, c.c_void_p, c.c_uint32])
        for name in ("ready", "consumed"):
            owner._call("aclrtRecordNotify", owner.imports[name][0], owner.stream)
            owner._call("aclrtWaitAndResetNotify", owner.notifications[name], owner.stream, 30000)
        owner.synchronize()
        exchange(connection, "notify-verified", {})
        receipts = []
        for case in range(config["cases"]):
            write_input(case)
            exchange(connection, f"sum-start-{case}", {})
            endpoint.submit(endpoint.sequence, BOUNDARIES[endpoint.sequence % len(BOUNDARIES)])
            endpoint.synchronize()
            data = read(owner.output)
            if data != expected_sum(endpoint.config.elements, case):
                raise AssertionError("complete FP32 SUM differs from CPU reference")
            receipts.append(dict(case=case, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            exchange(connection, f"sum-verified-{case}", receipts[-1])

        stress_receipts = []
        if config["stress"]:
            count = len(BOUNDARIES)
            history_bytes = endpoint.config.nbytes * count
            history_upload, history_download = (runtime.allocate(history_bytes, host=True) for _ in range(2))
            history = runtime.allocate(history_bytes)
            allocations.extend((history_upload, history_download, history))
            for window in range(3):
                first_case = config["cases"] + window * count
                data = b"".join(pattern(endpoint.config.elements, rank, first_case + i) for i in range(count))
                c.memmove(history_upload.pointer, data, history_bytes)
                exchange(connection, f"stress-start-{window}", {})
                if rank == window % 2:
                    time.sleep(0.001)
                for index in range(count):
                    offset = index * endpoint.config.nbytes
                    runtime.copy(
                        owner.input.value, history_upload.pointer + offset, endpoint.config.nbytes, "h2d", stream
                    )
                    endpoint.submit(endpoint.sequence, BOUNDARIES[endpoint.sequence % len(BOUNDARIES)])
                    runtime.copy(history.pointer + offset, owner.output.value, endpoint.config.nbytes, "d2d", stream)
                endpoint.synchronize()
                runtime.copy(history_download.pointer, history.pointer, history_bytes, "d2h", stream)
                owner.synchronize()
                observed = c.string_at(history_download.pointer, history_bytes)
                expected = b"".join(expected_sum(endpoint.config.elements, first_case + i) for i in range(count))
                if observed != expected:
                    raise AssertionError("changed-input asynchronous output history differs")
                stress_receipts.append(
                    dict(window=window, calls=count, bytes=history_bytes, sha256=hashlib.sha256(observed).hexdigest())
                )
                exchange(connection, f"stress-verified-{window}", stress_receipts[-1])
            write_input(config["cases"] - 1)

        reference, comm = None, c.c_void_p()
        reference_receipts = []
        if "hccl_reference" in config:
            reference = hccl_reference(config["hccl_reference"])
            root_bytes = reference.sidecard_hccl_root_info_bytes()
            if not 0 < root_bytes <= FRAME_BYTES // 2:
                raise ValueError("HCCL root-info size exceeds control budget")
            root = c.create_string_buffer(root_bytes)
            if rank == 0:
                checked(reference.sidecard_hccl_get_root_info, root, root_bytes)
            peer_root = exchange(connection, "hccl-root", {"root": root.raw.hex() if rank == 0 else None})
            if rank == 1:
                c.memmove(root, bytes.fromhex(peer_root["root"]), root_bytes)
            checked(reference.sidecard_hccl_create, root, root_bytes, rank, c.byref(comm))
            exchange(connection, "hccl-connected", {})
            for case in range(config["cases"]):
                write_input(case)
                exchange(connection, f"hccl-sum-start-{case}", {})
                checked(
                    reference.sidecard_hccl_all_reduce,
                    comm,
                    owner.input,
                    owner.output,
                    endpoint.config.elements,
                    owner.stream,
                )
                owner.synchronize()
                data = read(owner.output)
                checksum = hashlib.sha256(data).hexdigest()
                if data != expected_sum(endpoint.config.elements, case) or checksum != receipts[case]["sha256"]:
                    raise AssertionError("HCCL, custom and CPU complete outputs differ")
                reference_receipts.append(dict(case=case, sha256=checksum, bytes=len(data)))
                exchange(connection, f"hccl-sum-verified-{case}", reference_receipts[-1])

        timings = []
        modes = ("custom", "hccl") if reference else ("custom",)
        for mode in modes:
            exchange(connection, f"warmup-{mode}", {})
            for _ in range(config["warmup"]):
                if mode == "custom":
                    endpoint.submit(endpoint.sequence, BOUNDARIES[endpoint.sequence % len(BOUNDARIES)])
                else:
                    checked(
                        reference.sidecard_hccl_all_reduce,
                        comm,
                        owner.input,
                        owner.output,
                        endpoint.config.elements,
                        owner.stream,
                    )
            endpoint.synchronize()
            if read(owner.output) != expected_sum(endpoint.config.elements, config["cases"] - 1):
                raise AssertionError("warmed collective output differs")
        for pair in range(config["pairs"]):
            for mode in modes if pair % 2 == 0 else tuple(reversed(modes)):
                exchange(connection, f"timing-{pair}-{mode}", {})
                if rank == 1 and config["skew_us"]:
                    time.sleep(config["skew_us"] / 1_000_000)
                started = time.monotonic_ns()
                for _ in range(config["batch"]):
                    if mode == "custom":
                        endpoint.submit(endpoint.sequence, BOUNDARIES[endpoint.sequence % len(BOUNDARIES)])
                    else:
                        checked(
                            reference.sidecard_hccl_all_reduce,
                            comm,
                            owner.input,
                            owner.output,
                            endpoint.config.elements,
                            owner.stream,
                        )
                submitted = time.monotonic_ns()
                endpoint.synchronize()
                finished = time.monotonic_ns()
                if read(owner.output) != expected_sum(endpoint.config.elements, config["cases"] - 1):
                    raise AssertionError("timed collective output differs")
                timings.append(
                    dict(
                        pair=pair,
                        mode=mode,
                        started_ns=started,
                        submitted_ns=submitted,
                        completed_ns=finished,
                        count=config["batch"],
                    )
                )
        exchange(connection, "all-work-completed", {})
        if reference:
            checked(reference.sidecard_hccl_destroy, c.byref(comm), 1)
        drained = endpoint.drain()
        imports_closed = endpoint.release_imports(exchange(connection, "drained", drained))
        endpoint.close(exchange(connection, "imports-closed", imports_closed))
        for allocation in allocations:
            allocation.close()
        result = dict(
            rank=rank,
            physical_chip=physical,
            passed=True,
            fp32_cases=receipts,
            changed_input_stress=stress_receipts,
            hccl_fp32_cases=reference_receipts,
            timings=timings,
            next_sequence=endpoint.sequence,
            ownership_released=True,
        )
        (output / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
        exchange(connection, "done", result)
    except BaseException as error:
        chain, cause = [], error
        while cause is not None:
            chain.append(dict(type=type(cause).__name__, message=str(cause)))
            cause = cause.__cause__
        result = dict(
            rank=rank,
            passed=False,
            error_type=type(error).__name__,
            error=str(error),
            cause_chain=chain,
            endpoint_state=getattr(endpoint, "state", None),
            owners_retained=True,
            pid=os.getpid(),
        )
        (output / f"rank{rank}-failure.json").write_text(json.dumps(result, indent=2) + "\n")
        with contextlib.suppress(OSError):
            send(connection, dict(phase="error", payload=result))
        # An external owner can inspect/drain these exact processes. Do not
        # unwind SDK objects or guess that a failed enqueue cancelled DMA.
        while True:
            signal.pause()


def run(config, output):
    config = validate(config)
    output = Path(output).absolute()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    config_file = output / "config.json"
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    connections, processes = [], []
    for rank in range(2):
        parent, child = socket.socketpair()
        parent.settimeout(CONTROL_TIMEOUT_SECONDS)
        command = [
            sys.executable,
            "-m",
            "tools.sidecard.pair_collective_probe",
            "--config",
            str(config_file),
            "--output",
            str(output),
            "--worker",
            str(rank),
            "--fd",
            str(child.fileno()),
        ]
        with (output / f"rank{rank}.log").open("w") as log:
            process = subprocess.Popen(command, pass_fds=(child.fileno(),), stdout=log, stderr=subprocess.STDOUT)
        child.close()
        connections.append(parent)
        processes.append(process)
    (output / "processes.json").write_text(json.dumps(dict(pids=[p.pid for p in processes]), indent=2) + "\n")
    try:
        while True:
            packets = [receive(connection) for connection in connections]
            phases = [packet.get("phase") for packet in packets]
            if phases[0] != phases[1] or phases[0] == "error":
                raise RuntimeError("pair phase failure: " + str(phases))
            phase = phases[0]
            with (output / "progress.jsonl").open("a") as log:
                log.write(json.dumps(dict(phase=phase, at_ns=time.monotonic_ns())) + "\n")
            for rank, connection in enumerate(connections):
                payload = dict(packets[1 - rank]["payload"])
                if phase == "nonce" and rank == 0:
                    payload["own_nonce"] = packets[0]["payload"]["nonce"]
                send(connection, dict(phase=phase, payload=payload))
            if phase == "done":
                ranks = [packet["payload"] for packet in packets]
                break
        for process in processes:
            if process.wait(timeout=10) != 0:
                raise RuntimeError("completed worker exited unsuccessfully")
        summary = dict(
            protocol=PROBE_PROTOCOL,
            passed=True,
            physical_chips=config["physical_chips"],
            rows=config["rows"],
            exchange_mode=config["exchange_mode"],
            handshake=config["handshake"],
            requested_arrival_skew_us=config["skew_us"],
            device_numerical_scope="synthetic finite FP32 SUM; not GLM or full transport qualification",
            notify_timeout_units_verified=False,
            ranks=ranks,
            comparison=summarize(ranks),
        )
        (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    except BaseException:
        (output / "quarantine.json").write_text(
            json.dumps(
                dict(
                    pids=[p.pid for p in processes],
                    owners_retained=True,
                    automatic_kill=False,
                ),
                indent=2,
            )
            + "\n"
        )
        raise
    finally:
        for connection in connections:
            connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=int, choices=(0, 1))
    parser.add_argument("--fd", type=int)
    args = parser.parse_args()
    config = validate(json.loads(args.config.read_text()))
    if args.worker is not None:
        if args.fd is None:
            parser.error("worker requires private control FD")
        worker(config, args.output, args.worker, socket.socket(fileno=args.fd))
    else:
        result = run(config, args.output)
        print(json.dumps(dict(passed=result["passed"], physical_chips=result["physical_chips"], rows=result["rows"])))


if __name__ == "__main__":
    main()
