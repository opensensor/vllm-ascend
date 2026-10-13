# SPDX-License-Identifier: Apache-2.0
"""CUDA coordinator using private worker-owned cross-vendor DMA slots."""

import json
import os
import secrets
import signal
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path

from .cold_expert_probe import receive
from .cold_placement import load, lookup
from .completion_protocol import CONTROL_CAPACITY, send_fd
from .completion_protocol import enabled as completion_enabled
from .completion_results import CompletionResults
from .dma_pack_graph import DmaPackGraphs
from .dma_protocol import (
    CLOSE_TAG,
    COMMAND,
    INPUT_ROW_BYTES,
    OUTPUT_ROW_BYTES,
    RELEASE,
    CudaByteView,
    CudaSourceEvent,
    dma_capacity,
    receive_control,
    send_control,
    validate_descriptor,
    validate_identity,
)
from .hot_decode_graph import HotDecodeGraphs
from .layer_weights import NPU_LAYERS, TOP_K, WIDTH, expert_batch_size
from .pcie.peer import Driver
from .pcie.runtime import Runtime
from .pcie.segmented import export_parts

STARTUP_TIMEOUT = 1800
ACTIVE_TIMEOUT = 300
CLOSE_TIMEOUT = 15


class DmaDispatch:
    def __init__(self, config, cuda, torch, *, wait=True):
        self.spec = json.loads(Path(config).read_bytes())
        self.max_chunk = expert_batch_size(self.spec)
        self.capacity = dma_capacity(self.spec)
        self.hot = load(self.spec)
        self.cuda, self.torch = cuda, torch
        self.lookups = {
            int(layer): torch.tensor(lookup(ids), dtype=torch.int32, device="cuda") for layer, ids in self.hot.items()
        }
        self.output = Path(self.spec["output"])
        self.control = self.output / "dma-control"
        self.control.mkdir(mode=0o700)
        self.nonce = secrets.token_hex(32)
        self.connections, self.processes, self.logs = [], [], []
        self.inputs, self.outputs, self.imports, self.events = [], [], [], []
        self.pack_graphs = self.hot_graphs = None
        self.sequences = [0, 0]
        self.failed = self.closed = False
        self.lock = threading.RLock()
        self.runtime = Runtime("cuda", self.spec["cuda_backend"]["cuda_runtime"])
        self.driver = Driver()
        self.completion_results = None
        self.result_registrations = set()
        try:
            if completion_enabled(self.spec):
                self.completion_results = CompletionResults(self.spec, self.nonce, self.driver, torch, self.max_chunk)
            self.start(config)
            if wait:
                self.ready()
        except BaseException:
            self.failed = True
            self.close()
            raise

    def start(self, config):
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
                "tools.sidecard.dma_worker",
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
            environment = dict(os.environ, PYTHONPATH=source)
            # sudo does not preserve PYTHONPATH; execute from the frozen package root.
            self.processes.append(
                subprocess.Popen(command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT)
            )

    def ready(self):
        sync_memops = []
        for rank, listener in enumerate(self.connections):
            deadline = time.monotonic() + STARTUP_TIMEOUT
            while True:
                if self.processes[rank].poll() is not None:
                    raise RuntimeError("direct DMA worker exited during preparation; inspect its rank log")
                if time.monotonic() >= deadline:
                    raise TimeoutError("direct DMA worker preparation timed out")
                try:
                    connection, _ = listener.accept()
                    break
                except TimeoutError:
                    continue
            listener.close()
            connection.settimeout(STARTUP_TIMEOUT)
            self.connections[rank] = connection
            pid, uid, _ = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            if uid != 0:
                raise ValueError("direct DMA worker lacks the privileged kernel endpoint")
            hello = receive_control(connection)
            if hello.get("pid") != pid:
                raise ValueError("direct worker PID differs from private socket credentials")
            validate_identity(hello, self.spec, self.nonce, rank)
            contract = self.completion_results is not None
            if hello.get("cuda_completion_contract", False) != contract:
                raise ValueError("worker and coordinator completion protocol selection differs")
            slots = (("input", self.inputs),) if contract else (("input", self.inputs), ("output", self.outputs))
            for name, targets in slots:
                validate_descriptor(hello[name], capacity=self.capacity)
                allocation = self.runtime.import_memory(hello[name])
                self.imports.append(allocation)
                before = self.driver.ensure_sync_memops(allocation)
                sync_memops.append({"rank": rank, "slot": name, "enabled_before_import_check": before})
                view = CudaByteView(allocation)
                tensor = self.torch.as_tensor(view, device="cuda:0")
                if tensor.data_ptr() != allocation.pointer or tensor.numel() != self.capacity:
                    raise ValueError("CUDA IPC tensor does not share its dedicated mapping")
                targets.append(tensor)
            event = CudaSourceEvent(self.runtime, self.driver, self.torch.cuda.current_stream())
            self.events.append(event)
            reply = {"source_readiness": "sender-event-synchronized", "nonce": self.nonce}
            if contract:
                reply.update(
                    result_owner="coordinator",
                    native_owner_ordering=True,
                    epoch=self.completion_results.epoch,
                    capacity=self.capacity,
                )
                self.outputs.append(self.completion_results.views[rank])
                if self.completion_results.input_record:
                    reply.update(input_record=True, control_capacity=CONTROL_CAPACITY)
            send_control(connection, reply)
            if contract:
                self.result_registrations.add(rank)
                for part in export_parts(self.completion_results.exports[rank]):
                    send_fd(connection, part.fd)
                if self.completion_results.input_record:
                    send_fd(connection, self.completion_results.control_exports[rank].fd)
            acknowledgement = receive_control(connection)
            if acknowledgement != {"ready": True, "nonce": self.nonce}:
                raise ValueError("direct DMA worker source-event readiness differs")
            connection.settimeout(ACTIVE_TIMEOUT)
        if self.spec.get("direct_dma", {}).get("cuda_pack_graphs", False):
            with self.torch.inference_mode():
                self.pack_graphs = DmaPackGraphs(self.torch, self.lookups, self.inputs)
        (self.output / "ipc-sync-memops.json").write_text(json.dumps(sync_memops, indent=2) + "\n")
        (self.output / "npu-processes.json").write_text(
            json.dumps(
                {
                    "pids": [p.pid for p in self.processes],
                    "physical_devices": [2, 3],
                    "hccl": False,
                    "transport": "direct-dma",
                    "payload_host_staging": False,
                    "capacity_per_slot": self.capacity,
                    "gpu_slot_owner": "coordinator-results-worker-inputs"
                    if self.completion_results
                    else "independent-ascend-worker",
                    "cuda_completion_contract": self.completion_results is not None,
                },
                indent=2,
            )
            + "\n"
        )

    def __call__(self, layer, x, ids, weights):
        return self.compute(layer, x, ids, weights)[0]

    def prepare_layers(self, layers):
        if self.spec.get("direct_dma", {}).get("cuda_hot_graphs", False):
            if self.hot_graphs is not None:
                raise RuntimeError("hot expert graphs already prepared")
            self.hot_graphs = HotDecodeGraphs(self.cuda, [layers[index] for index in NPU_LAYERS], self.torch)

    def with_shared(self, layer, x, ids, weights):
        return self.compute(layer.index, x, ids, weights, lambda: layer.dense_mlp(x, "mlp.shared_experts"))

    def batch_with_shared(self, layer, prepared):
        """Combine only cold rows; CUDA hot/shared calls keep their original shapes."""
        if layer.index not in NPU_LAYERS or not prepared:
            raise ValueError("cold batching requires prepared middle-layer chunks")
        t = self.torch
        sizes = [value[0].shape[0] for value in prepared]
        if any(size < 1 or size > self.cuda.max_chunk for size in sizes) or sum(sizes) > self.max_chunk:
            raise ValueError("cold batching exceeds CUDA chunk or DMA slot capacity")
        x = t.cat([value[0] for value in prepared])
        ids = t.cat([value[4] for value in prepared])
        weights = t.cat([value[5] for value in prepared])

        def hot_call(hot_ids, hot_weights):
            return t.cat(
                [
                    self.cuda(layer.index, value[0], local_ids, local_weights)
                    for value, local_ids, local_weights in zip(
                        prepared, hot_ids.split(sizes), hot_weights.split(sizes), strict=True
                    )
                ]
            )

        def shared_call():
            return t.cat([layer.dense_mlp(value[0], "mlp.shared_experts") for value in prepared])

        routed, shared = self.compute(layer.index, x, ids, weights, shared_call, hot_call=hot_call)
        return zip(routed.split(sizes), shared.split(sizes), strict=True)

    def compute(self, layer, x, ids, weights, shared_call=None, *, hot_call=None):
        with self.lock:
            if self.closed or self.failed:
                raise RuntimeError("direct expert transport is closed or failed")
            try:
                return self._compute(layer, x, ids, weights, shared_call, hot_call=hot_call)
            except BaseException:
                self.failed = True
                raise

    def _compute(self, layer, x, ids, weights, shared_call, *, hot_call=None):
        t, rows = self.torch, x.shape[0]
        if layer not in NPU_LAYERS:
            return self.cuda(layer, x, ids, weights), shared_call() if shared_call else None
        if not 1 <= rows <= self.max_chunk:
            raise ValueError("direct dispatch exceeds its private slot capacity")
        captured = self.pack_graphs is not None and rows == 1
        if captured:
            hot_ids, hot_weights, flags = self.pack_graphs.prepare(layer, x, ids, weights)
        else:
            local_ids = self.lookups[layer][ids.long()]
            owned = local_ids >= 0
            cold_weights = t.where(owned, 0, weights).float()
            flags = t.stack(
                [((ids >= rank * 144) & (ids < (rank + 1) * 144) & (cold_weights > 0)).any() for rank in range(2)]
            )
            hot_ids, hot_weights = t.where(owned, local_ids, 0), t.where(owned, weights, 0)
        # Only two admission flags reach the CPU; hidden rows and outputs remain on devices.
        active = flags.cpu().tolist()
        completion = getattr(self, "completion_results", None)
        if completion is not None:
            completion.retire()
        x_bytes, id_bytes = rows * WIDTH * 2, rows * TOP_K * 4
        pending = []
        for rank in range(2):
            if not active[rank]:
                continue
            if not captured:
                slot = self.inputs[rank]
                slot[:x_bytes].view(t.float16).reshape(rows, WIDTH).copy_(x)
                slot[x_bytes : x_bytes + id_bytes].view(t.int32).reshape(rows, TOP_K).copy_(ids)
                slot[x_bytes + id_bytes : rows * INPUT_ROW_BYTES].view(t.float32).reshape(rows, TOP_K).copy_(
                    cold_weights
                )
            # Includes earlier reply consumers on this CUDA stream, preventing slot reuse races.
            if completion is not None and getattr(completion, "input_record", False):
                completion.prepare(rank, self.sequences[rank], layer, rows, self.inputs[rank])
            self.events[rank].record(t.cuda.current_stream())
            tag = COMMAND.pack(self.sequences[rank], layer, rows)
            pending.append((rank, tag))
        for rank, tag in pending:
            self.events[rank].wait()
            self.connections[rank].sendall(tag)
        if self.hot_graphs is not None and rows == 1 and shared_call is not None:
            hot, shared = self.hot_graphs.compute(layer, x, hot_ids, hot_weights)
        else:
            hot = hot_call(hot_ids, hot_weights) if hot_call is not None else self.cuda(layer, x, hot_ids, hot_weights)
            shared = shared_call() if shared_call is not None else None
        if completion is not None:
            # Prequeue the exact rank-ordered consumers before receiving host
            # acknowledgements. DMA completion words release CUDA execution.
            hot = completion.merge(hot, pending, layer, rows)
        for rank, tag in pending:
            actual = bytearray(COMMAND.size)
            receive(self.connections[rank], memoryview(actual))
            if actual != tag:
                raise ValueError("direct expert completion belongs to another invocation")
            self.sequences[rank] += 1
        if pending and completion is None:
            self.driver.flush()
        # Match the qualified socket transport's fixed owner reduction order.
        if completion is None:
            for rank, _ in pending:
                hot = hot + self.outputs[rank][: rows * OUTPUT_ROW_BYTES].view(t.float32).reshape(rows, WIDTH)
        return hot, shared

    def retain_completion_owner(self, reason):
        """Keep the CUDA owner alive when pending waits or foreign DMA remain."""
        (self.output / "dma-retained-coordinator.json").write_text(
            json.dumps(dict(pid=os.getpid(), state="quarantined", reason=reason, automatic_kill=False), indent=2) + "\n"
        )
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        while True:
            time.sleep(1)

    def close(self):
        """Never kill a worker whose native DMA completion might be uncertain."""
        with self.lock:
            if self.closed:
                return
            completion = getattr(self, "completion_results", None)
            if completion is not None:
                queue = completion.queue
                if queue.failed or any(
                    not value.sealed or not value.complete.ready() for value in queue.active.values()
                ):
                    self.retain_completion_owner("CUDA completion consumers have not established quiescence")
            self.torch.cuda.synchronize()
            if self.hot_graphs is not None:
                self.hot_graphs.close()
                self.hot_graphs = None
            if self.pack_graphs is not None:
                self.pack_graphs.close()
                self.pack_graphs = None
            quiescent = []
            if completion is not None:
                close_ranks = sorted(self.result_registrations)
            else:
                close_ranks = list(range(2)) if len(self.inputs) == 2 else []
            for rank in close_ranks:
                connection = self.connections[rank]
                connection.settimeout(CLOSE_TIMEOUT)
                tag = CLOSE_TAG
                try:
                    connection.sendall(tag)
                    body = bytearray(COMMAND.size)
                    receive(connection, memoryview(body))
                    if body != tag:
                        raise ValueError("direct close acknowledgement differs")
                    quiescent.append(connection)
                except (OSError, EOFError, ValueError):
                    self.failed = True
            if completion is not None:
                if len(quiescent) != len(self.result_registrations):
                    self.retain_completion_owner("foreign result DMA mappings were not released by every worker")
                completion.close()
                (self.output / "completion-contract-receipt.json").write_text(
                    json.dumps(
                        dict(
                            calls=completion.calls,
                            device_waits=completion.waits,
                            state="closed",
                            result_owner="coordinator",
                            host_result_flushes=0,
                        ),
                        indent=2,
                    )
                    + "\n"
                )
            self.closed = True
            self.inputs.clear()
            self.outputs.clear()
            for event in self.events:
                event.close()
            self.events.clear()
            for allocation in self.imports:
                allocation.close()
            self.imports.clear()
            for connection in quiescent:
                connection.sendall(RELEASE)
            for connection in self.connections:
                connection.close()
            survivors = []
            for process in self.processes:
                try:
                    process.wait(timeout=CLOSE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    survivors.append(process.pid)
            if survivors:
                (self.output / "dma-retained-workers.json").write_text(
                    json.dumps(
                        {
                            "pids": survivors,
                            "automatic_kill": False,
                            "reason": "DMA quiescence or owner release is not established; preserve worker allocations",
                        },
                        indent=2,
                    )
                    + "\n"
                )
            for log in self.logs:
                log.close()
