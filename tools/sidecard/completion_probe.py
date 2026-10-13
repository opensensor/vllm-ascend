# SPDX-License-Identifier: Apache-2.0
"""Qualify foreign DMA-BUF ownership, packed results, and ordered CUDA merge."""

import argparse
import ctypes
import json
import os
import secrets
import signal
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

from .cold_expert_probe import receive
from .completion_protocol import (
    CONTROL_CAPACITY,
    INPUT_SEQUENCE_OFFSET,
    RESULT_OFFSET,
    SEQUENCE_OFFSET,
    RemoteResult,
    epoch,
    input_record,
    receive_fd,
    receive_result,
    result_header,
    send_fd,
)
from .completion_results import CompletionResults
from .dma_protocol import (
    CLOSE_TAG,
    COMMAND,
    INPUT_ROW_BYTES,
    AdmittedCudaSource,
    CudaByteView,
    dma_capacity,
    receive_control,
    send_control,
)
from .layer_weights import expert_batch_size
from .pcie.mailbox import RECORD, Invocation
from .pcie.peer import ASCEND_TO_CUDA, CUDA_TO_ASCEND, Driver, KernelPeer
from .pcie.runtime import Runtime
from .pcie.segmented import export_allocation, export_parts, make_peer

GUARD_POISON = 0xD3
PROBE_ROWS = (1, 4, 17, 32)
LARGE_PROBE_ROWS = (64, 96, 128, 255, 256, 511, 512, 513, 768, 1023, 1024)
WIDTH = 4096


def fixture(rank, rows):
    # Cancellation and small additions make rank addition order observable.
    values = (-1e20, -1e20, 1e20, 0.125) if rank == 0 else (1.0, 1e20, -1e20, -0.0625)
    return struct.pack("<4f", *values) * (rows * WIDTH // len(values))


def probe_rows(spec):
    limit = expert_batch_size(spec)
    return PROBE_ROWS + tuple(rows for rows in LARGE_PROBE_ROWS if rows <= limit)


class HbmSequence:
    def __init__(self, base, offset=SEQUENCE_OFFSET):
        self.base, self.runtime, self.host = base, base.runtime, False
        self.size = RESULT_OFFSET - SEQUENCE_OFFSET
        self.offset = offset

    @property
    def pointer(self):
        return self.base.pointer + self.offset

    @property
    def leases(self):
        return self.base.leases

    @leases.setter
    def leases(self, value):
        self.base.leases = value


def park():
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    while True:
        time.sleep(1)


def producer(args, spec):
    """Validate uploaded records, publish ordered DMA, and retain uncertain owners."""
    if os.geteuid() != 0 or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3":
        raise ValueError("probe producer requires root and only assigned chips 2,3")
    capacity = dma_capacity(spec)
    cuda = Runtime("cuda", spec["cuda_backend"]["cuda_runtime"])
    npu = Runtime("ascend", "/usr/local/Ascend/cann-9.1.0/x86_64-linux/lib64/libascendcl.so", args.rank)
    packet, host, guard = npu.allocate(capacity), npu.allocate(capacity, host=True), npu.allocate(capacity, host=True)
    stream, event = npu.stream(), npu.event()
    ctypes.memset(guard.pointer, GUARD_POISON, capacity)
    peer = make_peer(args.rank + 2, capacity, persistent_gpu_mappings=True)
    use_input = input_record(spec)
    admitted_rows = probe_rows(spec)
    notification = None
    incoming = None
    with socket.socket(socket.AF_UNIX) as connection:
        connection.connect(args.socket)
        pid, uid, _ = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        registration = receive_control(connection)
        if registration != dict(nonce=args.nonce, epoch=epoch(args.nonce), rank=args.rank, pid=pid, uid=uid):
            raise ValueError("foreign result registration identity differs")
        result = receive_result(cuda, connection, capacity)
        if use_input:
            control = RemoteResult(cuda, receive_fd(connection), size=CONTROL_CAPACITY)
            gpu_input = receive_result(cuda, connection, capacity)
            incoming = npu.allocate(capacity)
            notification = KernelPeer(args.rank + 2, persistent_gpu_mappings=True)
            ready = AdmittedCudaSource(cuda)
        send_control(connection, dict(ready=True, rank=args.rank))
        expected = 0
        try:
            while True:
                tag = bytearray(COMMAND.size)
                receive(connection, memoryview(tag))
                if tag == CLOSE_TAG:
                    peer.close()
                    result.close()
                    if use_input:
                        notification.close()
                        control.close()
                        gpu_input.close()
                    send_control(
                        connection,
                        dict(
                            closed=True,
                            guard_poison_exact=ctypes.string_at(guard.pointer, capacity)
                            == bytes([GUARD_POISON]) * capacity,
                        ),
                    )
                    break
                sequence, layer, rows = COMMAND.unpack(tag)
                if sequence != expected or layer != 18 or rows not in admitted_rows:
                    raise ValueError("foreign result probe command differs")
                if use_input:
                    ready.admitted = True
                    peer.copy(
                        CUDA_TO_ASCEND,
                        gpu_input,
                        incoming,
                        guard,
                        rows * INPUT_ROW_BYTES + RECORD.size,
                        source_ready=ready,
                    )
                    npu.copy(host.pointer, incoming.pointer + rows * INPUT_ROW_BYTES, RECORD.size, "d2h", stream)
                    event.record(stream)
                    event.wait()
                    record = Invocation.decode(
                        ctypes.string_at(host.pointer, RECORD.size),
                        epoch=epoch(args.nonce),
                        sequence=sequence + 1,
                        slots=2,
                        rank=args.rank,
                    )
                    if (record.layer, record.rows, record.slot) != (layer, rows, args.rank):
                        raise ValueError("uploaded CUDA record differs")
                body = (
                    b"" if use_input else result_header(epoch(args.nonce), sequence, layer, rows, args.rank)
                ) + fixture(args.rank, rows)
                ctypes.memmove(host.pointer, body, len(body))
                npu.copy(packet.pointer, host.pointer, len(body), "h2d", stream)
                event.record(stream)
                event.wait()
                payload = peer.copy(ASCEND_TO_CUDA, result, packet, guard, len(body), source_ready=event)
                signal_source = (
                    HbmSequence(incoming, rows * INPUT_ROW_BYTES + INPUT_SEQUENCE_OFFSET)
                    if use_input
                    else HbmSequence(packet)
                )
                completion = (notification if use_input else peer).copy(
                    ASCEND_TO_CUDA, control if use_input else result, signal_source, guard, 8, source_ready=event
                )
                send_control(connection, dict(passed=True, sequence=sequence, payload=payload, completion=completion))
                expected += 1
        except BaseException as error:
            uncertain = peer.failed or (notification is not None and notification.failed)
            send_control(connection, dict(passed=False, error=str(error), native_dma_uncertain=uncertain))
            if uncertain:
                park()
            peer.close()
            result.close()
            if use_input and not notification.failed:
                notification.close()
                control.close()
                gpu_input.close()
            raise
        finally:
            if not peer.failed and (notification is None or not notification.failed):
                for value in (event, stream, packet, host, guard):
                    value.close()
                if incoming is not None:
                    incoming.close()


def coordinator(args, spec):
    """Preflight the unaligned HBM sequence source before queuing future waits."""
    import torch

    capacity = dma_capacity(spec)
    torch.cuda.set_device(0)
    nonce = secrets.token_hex(32)
    admitted_rows = probe_rows(spec)
    for rows in admitted_rows:
        Invocation(epoch(nonce), 1, 18, rows, 0, 0).encode(slots=2)
    root = args.output.parent
    root.mkdir(exist_ok=True)
    results = CompletionResults(spec, nonce, Driver(), torch, max(admitted_rows))
    inputs, input_exports = [], []
    if results.input_record:
        for rank in range(2):
            allocation = results.runtime.allocate(capacity)
            inputs.append((allocation, torch.as_tensor(CudaByteView(allocation), device="cuda:0")))
            input_exports.append(export_allocation(Driver(), allocation))
    hot_cpu = torch.tensor((1e20, 1.0, -1e20, 0.03125)).repeat(max(admitted_rows) * WIDTH // 4).reshape(-1, WIDTH)
    hot = hot_cpu.cuda()
    torch.cuda.synchronize()
    peers, processes, listeners, logs = [], [], [], []
    report = dict(
        passed=False,
        state="running",
        cases=[],
        completion_input_record=results.input_record,
        source_sequence_offset=None if results.input_record else SEQUENCE_OFFSET,
        source_sequence_offset_formula="rows * 8256 + 16" if results.input_record else None,
        host_result_flushes=0,
        capacity_per_slot=capacity,
    )
    try:
        for rank in range(2):
            path = root / f"probe-rank{rank}.sock"
            listener = socket.socket(socket.AF_UNIX)
            listener.bind(str(path))
            listener.listen(1)
            listener.settimeout(120)
            listeners.append(listener)
            log = (root / f"probe-rank{rank}.log").open("x")
            logs.append(log)
            command = [
                "sudo",
                "-n",
                spec["npu_python"],
                "-m",
                "tools.sidecard.completion_probe",
                "--config",
                str(args.config),
                "--output",
                str(args.output),
                "--rank",
                str(rank),
                "--socket",
                str(path),
                "--nonce",
                nonce,
                "--bootstrap",
            ]
            process = subprocess.Popen(command, cwd=args.config.parent / "source", stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            connection, _ = listener.accept()
            peers.append(connection)
            listener.close()
            _, uid, _ = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            if uid != 0:
                raise ValueError("probe producer must be privileged")
            connection.settimeout(120)
            send_control(
                connection, dict(nonce=nonce, epoch=results.epoch, rank=rank, pid=os.getpid(), uid=os.getuid())
            )
            for part in export_parts(results.exports[rank]):
                send_fd(connection, part.fd)
            if results.input_record:
                send_fd(connection, results.control_exports[rank].fd)
                for part in export_parts(input_exports[rank]):
                    send_fd(connection, part.fd)
            if receive_control(connection) != dict(ready=True, rank=rank):
                raise ValueError("probe producer acknowledgement differs")
        for sequence in range(len(admitted_rows) * 3):
            rows = admitted_rows[sequence % len(admitted_rows)]
            pending = [(rank, COMMAND.pack(sequence, 18, rows)) for rank in range(2)]
            for view in results.views:
                view.fill_(GUARD_POISON)
            if results.input_record:
                for rank in range(2):
                    results.prepare(rank, sequence, 18, rows, inputs[rank][1])
                torch.cuda.synchronize()
            for rank, tag in pending:
                peers[rank].sendall(tag)
            # Preflight every admitted shape before submitting its future wait.
            replies = [receive_control(peer) for peer in peers] if sequence < len(admitted_rows) else None
            report["preflight_replies"] = replies if sequence == 0 else report.get("preflight_replies")
            if replies is not None and not all(value.get("passed") for value in replies):
                raise RuntimeError("packed sequence DMA preflight failed")
            output = results.merge(hot[:rows], pending, 18, rows)
            if replies is None:
                replies = [receive_control(peer) for peer in peers]
            if not all(value.get("passed") and value.get("sequence") == sequence for value in replies):
                raise RuntimeError("foreign producer did not establish completed DMA")
            actual = output.cpu()
            expected = hot_cpu[:rows].clone()
            for rank in range(2):
                partial = torch.frombuffer(bytearray(fixture(rank, rows)), dtype=torch.float32).reshape(rows, WIDTH)
                expected = expected + partial
            equal = torch.equal(actual, expected)
            result_end = rows * WIDTH * 4 + (0 if results.input_record else RESULT_OFFSET)
            tails_equal = all(bool((view[result_end:] == GUARD_POISON).all().cpu()) for view in results.views)
            results.retire()
            report["cases"].append(
                dict(
                    sequence=sequence,
                    rows=rows,
                    bitwise_equal=equal,
                    untouched_result_tail=tails_equal,
                    prequeued_before_reply=sequence >= len(admitted_rows),
                    replies=replies,
                )
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            if not equal or not tails_equal:
                raise AssertionError("packed result contract changed ordered FP32 rank addition")
        for connection in peers:
            connection.sendall(CLOSE_TAG)
            value = receive_control(connection)
            if value != dict(closed=True, guard_poison_exact=True):
                raise RuntimeError("foreign result mappings did not close cleanly")
        results.close()
        for exported in input_exports:
            exported.close()
        for allocation, view in inputs:
            allocation.close()
        report.update(passed=True, state="closed")
    except BaseException as error:
        report.update(error=str(error), state="quarantined")
        report["cuda_wait_submissions"] = results.waits
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        # Foreign DMA and queued waits may still depend on this CUDA owner.
        park()
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        for connection in peers:
            connection.close()
        for process in processes:
            process.wait(timeout=30)
        for log in logs:
            log.close()
    print(json.dumps(dict(passed=report["passed"], state=report["state"], cases=len(report["cases"]))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1))
    parser.add_argument("--socket")
    parser.add_argument("--nonce")
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.config.read_bytes())
    if args.bootstrap:
        environment = dict(os.environ)
        producer_mode = args.rank is not None
        environment.update(
            json.loads(Path(spec["npu_environment" if producer_mode else "cuda_environment"]).read_bytes())
        )
        environment["PYTHONPATH"] = str(args.config.parent / "source")
        if producer_mode:
            environment["ASCEND_RT_VISIBLE_DEVICES"] = "2,3"
        python = spec["npu_python"] if producer_mode else "/srv/ai/venvs/sidecard-cuda-20261012/bin/python"
        os.execve(
            python,
            [python, "-m", "tools.sidecard.completion_probe", *[v for v in sys.argv[1:] if v != "--bootstrap"]],
            environment,
        )
    producer(args, spec) if args.rank is not None else coordinator(args, spec)


if __name__ == "__main__":
    main()
