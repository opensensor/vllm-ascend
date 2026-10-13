# SPDX-License-Identifier: Apache-2.0
"""Complete mHC boundary tensors carried by the qualified PCIe DMA bridge.

A single serialized slot is shared by both TP2 ranks. CUDA owns its input and
output; each privileged rank owns its HBM and lifetime guard. Small authenticated
control packets carry position and reservation metadata. Tensor payloads stay on
devices. Release requires both ranks to close their bridge mappings first.
"""

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
from types import SimpleNamespace

from .completion_protocol import RemoteResult, receive_fd, send_fd
from .dma_protocol import AdmittedCudaSource, CudaByteView, CudaSourceEvent, receive_control, send_control
from .dma_worker import GUARD_POISON, TensorAllocation, TorchSourceEvent
from .layer_weights import MAX_CHUNK
from .pcie.peer import ASCEND_TO_CUDA, CUDA_TO_ASCEND, Driver, KernelPeer
from .pcie.runtime import Runtime
from .stage_options import enabled
from .stage_wire import descriptors, metadata, validate_metadata

PROTOCOL = "glm-complete-stage-dma-v1"
CAPACITY = 4 * 1024 * 1024
STARTUP_TIMEOUT = 1800
ACTIVE_TIMEOUT = 300
CLOSE_TIMEOUT = 30
PROFILE_RECORD_LIMIT = 4096


def identity(spec, nonce, rank):
    return dict(
        protocol=PROTOCOL,
        nonce=nonce,
        rank=rank,
        physical_chip=rank + 2,
        checkpoint_sha256=spec["checkpoint_sha256"],
        capacity=CAPACITY,
    )


def validate_packet(packet, spec, sequence, position, context):
    """Reject altered state geometry, replay, and cache reservation changes."""
    validate_metadata(packet)
    if packet["rows"] > MAX_CHUNK or packet["payload_bytes"] > CAPACITY:
        raise ValueError("complete-stage DMA state exceeds its dedicated slot")
    first = packet.get("first_position")  # Positions belong to a separate envelope.
    if first is not None:
        raise ValueError("position must be carried outside the canonical state metadata")
    if packet["checkpoint_sha256"] != spec["checkpoint_sha256"] or packet["sequence"] != sequence:
        raise ValueError("complete-stage DMA identity or sequence differs")
    if type(position) is not int or position < 0 or position + packet["rows"] > spec["max_context"]:
        raise ValueError("complete-stage DMA positions exceed capacity")
    reservation = packet.get("request_context_limit")
    if type(reservation) is not int or not position + packet["rows"] <= reservation <= spec["max_context"]:
        raise ValueError("complete-stage DMA reservation exceeds capacity")
    if context is not None and position != 0 and reservation != context:
        raise ValueError("complete-stage DMA changed its active reservation")
    return reservation


def views(packed, rows, torch):
    if type(rows) is not int or not 1 <= rows <= MAX_CHUNK or packed.numel() != CAPACITY:
        raise ValueError("complete-stage DMA view exceeds its fixed slot geometry")
    dtypes = {"fp16": torch.float16, "fp32": torch.float32, "int64": torch.int64}
    return tuple(
        packed[field["offset"] : field["offset"] + field["bytes"]].view(dtypes[field["dtype"]]).reshape(field["shape"])
        for field in descriptors(rows)[0]
    )


def pack(packed, state, positions, torch):
    for target, source in zip(views(packed, state[0].shape[0], torch), (*state, positions), strict=True):
        if target.shape != source.shape or target.dtype != source.dtype or target.device != source.device:
            raise ValueError("complete-stage DMA input state geometry differs")
        target.copy_(source)


def preserve(output, name, reason):
    """Keep native allocations alive when their quiescence cannot be proved."""
    (Path(output) / name).write_text(json.dumps(dict(pid=os.getpid(), reason=str(reason), automatic_kill=False)) + "\n")
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    while True:
        time.sleep(1)


class StageDMA:
    """Coordinator-owned buffers; invocations remain serialized by Generation."""

    def __init__(self, config, torch):
        self.spec = json.loads(Path(config).read_bytes())
        if not enabled(self.spec):
            raise ValueError("complete-stage DMA is not enabled")
        self.torch, self.output = torch, Path(self.spec["output"])
        self.nonce = secrets.token_hex(32)
        self.connections, self.processes, self.logs = [], [], []
        self.registered = set()
        self.closed = False
        self.runtime = Runtime("cuda", self.spec["cuda_backend"]["cuda_runtime"])
        self.driver = Driver()
        self.allocations, self.exports, self.buffers = [], [], []
        self.source_event = None
        try:
            for _ in range(2):
                self.allocations.append(self.runtime.allocate(CAPACITY))
            for value in self.allocations:
                self.exports.append(self.driver.export(value))
            self.buffers = [torch.as_tensor(CudaByteView(value), device="cuda:0") for value in self.allocations]
            self.source_event = CudaSourceEvent(self.runtime, self.driver, torch.cuda.current_stream())
            self.control = self.output / "stage-dma-control"
            self.control.mkdir(mode=0o700)
            source = str(Path(config).resolve().parent / "source")
            for rank in range(2):
                path = self.control / f"rank{rank}.sock"
                listener = socket.socket(socket.AF_UNIX)
                listener.bind(str(path))
                listener.listen(1)
                listener.settimeout(1)
                self.connections.append(listener)
                log = (self.output / f"npu-rank{rank}.log").open("x")
                self.logs.append(log)
                command = [
                    "sudo",
                    "-n",
                    self.spec["npu_python"],
                    "-m",
                    "tools.sidecard.stage_dma",
                    "--config",
                    str(Path(config).resolve()),
                    "--rank",
                    str(rank),
                    "--socket",
                    str(path),
                    "--nonce",
                    self.nonce,
                    "--parent-pid",
                    str(os.getpid()),
                    "--parent-uid",
                    str(os.getuid()),
                    "--bootstrap",
                ]
                self.processes.append(subprocess.Popen(command, cwd=source, stdout=log, stderr=subprocess.STDOUT))
        except BaseException:
            self.close()
            raise

    def ready(self):
        for rank, listener in enumerate(self.connections):
            deadline = time.monotonic() + STARTUP_TIMEOUT
            while True:
                if self.processes[rank].poll() is not None:
                    raise RuntimeError("complete-stage worker exited during preparation")
                if time.monotonic() > deadline:
                    raise TimeoutError("complete-stage DMA startup timed out")
                try:
                    connection, _ = listener.accept()
                    break
                except TimeoutError:
                    continue
            listener.close()
            self.connections[rank] = connection
            connection.settimeout(STARTUP_TIMEOUT)
            pid, uid, _ = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            hello = receive_control(connection)
            if uid != 0 or hello != dict(identity(self.spec, self.nonce, rank), pid=pid):
                raise ValueError("complete-stage DMA peer identity differs")
            send_control(connection, identity(self.spec, self.nonce, rank))
            self.registered.add(rank)
            for exported in self.exports:
                send_fd(connection, exported.fd)
            if receive_control(connection) != {"ready": True, "nonce": self.nonce}:
                raise ValueError("complete-stage DMA registration differs")
            connection.settimeout(ACTIVE_TIMEOUT)
        (self.output / "npu-processes.json").write_text(
            json.dumps(
                dict(
                    pids=[p.pid for p in self.processes],
                    physical_devices=[2, 3],
                    hccl=True,
                    transport="direct-dma",
                    payload_host_staging=False,
                    capacity_per_slot=CAPACITY,
                ),
                indent=2,
            )
            + "\n"
        )

    def compute(self, state, position, sequence, context):
        torch = self.torch
        packet = metadata(state[0].shape[0], sequence, self.spec["checkpoint_sha256"], request_context_limit=context)
        validate_packet(packet, self.spec, sequence, position, context)
        positions = torch.arange(position, position + state[0].shape[0], device="cuda", dtype=torch.int64)
        pack(self.buffers[0], state, positions, torch)
        self.source_event.record(torch.cuda.current_stream())
        self.source_event.wait()
        message = dict(state=packet, first_position=position)
        for connection in self.connections:
            send_control(connection, message)
        for connection in self.connections:
            if receive_control(connection) != message:
                raise ValueError("complete-stage DMA completion sequence differs")
        # DMA is complete on both ranks. Flush in the CUDA allocation owner's
        # context before enqueuing its consumer or reusing the input slot.
        self.driver.flush()
        return views(self.buffers[1], state[0].shape[0], torch)[:4]

    def close(self):
        if self.closed:
            return
        self.torch.cuda.synchronize()
        try:
            for rank in sorted(self.registered):
                send_control(self.connections[rank], {"close": self.nonce})
            for rank in sorted(self.registered):
                self.connections[rank].settimeout(CLOSE_TIMEOUT)
                if receive_control(self.connections[rank]) != {"closed": self.nonce}:
                    raise ValueError("complete-stage DMA mapping release differs")
        except BaseException as error:
            preserve(self.output, "stage-dma-retained-coordinator.json", error)
        self.buffers.clear()
        if self.source_event is not None:
            self.source_event.close()
        for exported in self.exports:
            exported.close()
        for allocation in self.allocations:
            allocation.close()
        for rank in sorted(self.registered):
            send_control(self.connections[rank], {"released": self.nonce})
        for connection in self.connections:
            connection.close()
        for rank, process in enumerate(self.processes):
            if rank not in self.registered and process.poll() is None:
                process.terminate()  # No DMA-BUF ownership has crossed to this rank.
            process.wait(timeout=CLOSE_TIMEOUT)
        for log in self.logs:
            log.close()
        self.closed = True


def serve(args, layers, graphs, torch):
    """Privileged HBM owner runs all fifteen layers before publishing a result.

    Keep this loop explicit so every completion and resource release follows
    the synchronous bridge contract. Both ranks receive the same state directly
    from CUDA; TP2 collectives remain inside the contiguous Ascend block.
    """
    spec = json.loads(Path(args.stage_dma).read_bytes())
    runtime = Runtime("cuda", spec["cuda_backend"]["cuda_runtime"])
    ascend = Runtime(
        "ascend", "/usr/local/Ascend/cann-9.1.0/x86_64-linux/lib64/libascendcl.so", args.rank, initialize=False
    )
    packed = torch.empty(CAPACITY, dtype=torch.uint8, device=f"npu:{args.rank}")
    output = torch.empty_like(packed)
    endpoint = TensorAllocation(ascend, packed)
    guard = ascend.allocate(CAPACITY, host=True)
    ctypes.memset(guard.pointer, GUARD_POISON, CAPACITY)
    peer = KernelPeer(args.rank + 2, persistent_gpu_mappings=True)
    source = AdmittedCudaSource(runtime)
    exports = []
    sequence, expected_position, context, calls = 0, 0, None, 0
    registered = released = False
    timings = []
    connection = socket.socket(socket.AF_UNIX)
    try:
        connection.connect(args.socket)
        pid, uid, _ = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        if (pid, uid) != (args.parent_pid, args.parent_uid):
            raise ValueError("complete-stage control socket belongs to another owner")
        send_control(connection, dict(identity(spec, args.nonce, args.rank), pid=os.getpid()))
        if receive_control(connection) != identity(spec, args.nonce, args.rank):
            raise ValueError("complete-stage owner registration differs")
        registered = True
        exports = [RemoteResult(runtime, receive_fd(connection), size=CAPACITY) for _ in range(2)]
        signal.signal(signal.SIGTERM, lambda signum, frame: None)
        signal.signal(signal.SIGINT, lambda signum, frame: None)
        send_control(connection, {"ready": True, "nonce": args.nonce})
        with torch.inference_mode():
            while True:
                message = receive_control(connection)
                if message == {"close": args.nonce}:
                    torch.npu.synchronize()
                    peer.close()
                    for exported in exports:
                        exported.close()
                    send_control(connection, {"closed": args.nonce})
                    if receive_control(connection) != {"released": args.nonce}:
                        raise ValueError("complete-stage CUDA owner release differs")
                    released = True
                    break
                if set(message) != {"state", "first_position"}:
                    raise ValueError("unexpected complete-stage DMA command")
                packet, position = message["state"], message["first_position"]
                reservation = validate_packet(packet, spec, sequence, position, context)
                if position != 0 and position != expected_position:
                    raise ValueError("complete-stage positions disagree with cache state")
                if position == 0:
                    context = reservation
                    for layer in layers:
                        layer.reset()
                        if not layer.linear:
                            layer.sparse_enabled = context > layer.config["index_topk"]
                source.admitted = True
                start = time.perf_counter()
                peer.copy(CUDA_TO_ASCEND, exports[0], endpoint, guard, packet["payload_bytes"], source_ready=source)
                uploaded = time.perf_counter()
                state = views(packed, packet["rows"], torch)[:4]
                if graphs is None:
                    for layer in layers:
                        state = layer.forward(*state)
                else:
                    state = graphs.forward(state)
                # Separate packet input and result storage prevents overwriting
                # a deferred residual view while packing the result.
                positions = torch.arange(position, position + packet["rows"], device=packed.device, dtype=torch.int64)
                pack(output, state, positions, torch)
                event = torch.npu.Event()
                event.record()
                event.synchronize()
                computed = time.perf_counter()
                if args.rank == 0:
                    returned = TensorAllocation(ascend, output)
                    peer.copy(
                        ASCEND_TO_CUDA,
                        exports[1],
                        returned,
                        guard,
                        packet["payload_bytes"],
                        source_ready=TorchSourceEvent(ascend, event),
                    )
                    returned.close()
                source.admitted = False
                send_control(connection, message)
                if len(timings) < PROFILE_RECORD_LIMIT:
                    timings.append(
                        dict(
                            sequence=sequence,
                            rows=packet["rows"],
                            upload_ms=(uploaded - start) * 1000,
                            compute_ms=(computed - uploaded) * 1000,
                            download_ms=(time.perf_counter() - computed) * 1000,
                        )
                    )
                calls += 1
                sequence += 1
                expected_position = position + packet["rows"]
    except BaseException as error:
        if peer.failed or (registered and not released):
            preserve(spec["output"], f"stage-dma-quarantine-rank{args.rank}.json", error)
        raise
    finally:
        if not peer.failed:
            peer.close()
            for exported in exports:
                exported.close()
            exact = ctypes.string_at(guard.pointer, CAPACITY) == bytes([GUARD_POISON]) * CAPACITY
            endpoint.close()
            guard.close()
            connection.close()
            (Path(spec["output"]) / f"stage-dma-rank{args.rank}.json").write_text(
                json.dumps(
                    dict(
                        state="closed",
                        calls=calls,
                        guard_poison_exact=exact,
                        profile=timings,
                        payload_host_staging=False,
                        physical_chip=args.rank + 2,
                    ),
                    indent=2,
                )
                + "\n"
            )
            if not exact:
                raise AssertionError("complete-stage lifetime guard was modified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "socket", "nonce"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--parent-uid", type=int, required=True)
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    spec = json.loads(Path(args.config).read_bytes())
    enabled(spec)
    if args.bootstrap:
        env = dict(os.environ)
        env.update(json.loads(Path(spec["npu_environment"]).read_bytes()))
        env["ASCEND_RT_VISIBLE_DEVICES"] = "2,3"
        env["PYTHONPATH"] = str(Path(args.config).resolve().parent / "source") + ":" + env.get("PYTHONPATH", "")
        command = [
            spec["npu_python"],
            "-m",
            "tools.sidecard.stage_dma",
            *[v for v in sys.argv[1:] if v != "--bootstrap"],
        ]
        os.execve(command[0], command, env)
    if os.geteuid() != 0 or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3":
        raise ValueError("complete-stage DMA requires root on assigned chips 2,3")
    from .layer_actor import npu_worker

    npu_worker(
        SimpleNamespace(
            **vars(args),
            checkpoint=spec["checkpoint"],
            identity=spec["checkpoint_sha256"],
            init_method=spec["init_method"],
            max_context=spec["max_context"],
            output=spec["output"],
            native_gate_override=spec["native_gate_override"],
            ops_library=spec.get("npu_ops_library"),
            decode_graphs=spec.get("npu_decode_graphs", False),
            stage_profile=False,
            native_bf16_round=spec.get("npu_native_bf16_round", False),
            stage_dma=args.config,
        )
    )


if __name__ == "__main__":
    main()
