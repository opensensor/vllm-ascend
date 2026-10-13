# SPDX-License-Identifier: Apache-2.0
"""CUDA coordinator using private worker-owned cross-vendor DMA slots."""

import json
import os
import secrets
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path

from .cold_expert_probe import receive
from .cold_placement import load, lookup
from .dma_pack_graph import DmaPackGraphs
from .dma_protocol import (
    CAPACITY,
    CLOSE_TAG,
    COMMAND,
    INPUT_ROW_BYTES,
    OUTPUT_ROW_BYTES,
    RELEASE,
    CudaByteView,
    CudaSourceEvent,
    receive_control,
    send_control,
    validate_descriptor,
    validate_identity,
)
from .hot_decode_graph import HotDecodeGraphs
from .layer_weights import NPU_LAYERS, TOP_K, WIDTH, prefill_chunk_size
from .pcie.peer import Driver
from .pcie.runtime import Runtime

STARTUP_TIMEOUT = 1800
ACTIVE_TIMEOUT = 300
CLOSE_TIMEOUT = 15


class DmaDispatch:
    def __init__(self, config, cuda, torch, *, wait=True):
        self.spec = json.loads(Path(config).read_bytes())
        self.max_chunk = prefill_chunk_size(self.spec)
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
        try:
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
            for name, targets in (("input", self.inputs), ("output", self.outputs)):
                validate_descriptor(hello[name])
                allocation = self.runtime.import_memory(hello[name])
                self.imports.append(allocation)
                before = self.driver.ensure_sync_memops(allocation)
                sync_memops.append({"rank": rank, "slot": name, "enabled_before_import_check": before})
                view = CudaByteView(allocation)
                tensor = self.torch.as_tensor(view, device="cuda:0")
                if tensor.data_ptr() != allocation.pointer or tensor.numel() != CAPACITY:
                    raise ValueError("CUDA IPC tensor does not share its dedicated mapping")
                targets.append(tensor)
            event = CudaSourceEvent(self.runtime, self.driver, self.torch.cuda.current_stream())
            self.events.append(event)
            send_control(connection, {"source_readiness": "sender-event-synchronized", "nonce": self.nonce})
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
                    "capacity_per_slot": CAPACITY,
                    "gpu_slot_owner": "independent-ascend-worker",
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

    def compute(self, layer, x, ids, weights, shared_call=None):
        with self.lock:
            if self.closed or self.failed:
                raise RuntimeError("direct expert transport is closed or failed")
            try:
                return self._compute(layer, x, ids, weights, shared_call)
            except BaseException:
                self.failed = True
                raise

    def _compute(self, layer, x, ids, weights, shared_call):
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
            self.events[rank].record(t.cuda.current_stream())
            tag = COMMAND.pack(self.sequences[rank], layer, rows)
            pending.append((rank, tag))
        for rank, tag in pending:
            self.events[rank].wait()
            self.connections[rank].sendall(tag)
        if self.hot_graphs is not None and rows == 1 and shared_call is not None:
            hot, shared = self.hot_graphs.compute(layer, x, hot_ids, hot_weights)
        else:
            hot = self.cuda(layer, x, hot_ids, hot_weights)
            shared = shared_call() if shared_call is not None else None
        for rank, tag in pending:
            actual = bytearray(COMMAND.size)
            receive(self.connections[rank], memoryview(actual))
            if actual != tag:
                raise ValueError("direct expert completion belongs to another invocation")
            self.sequences[rank] += 1
        if pending:
            self.driver.flush()
        # Match the qualified socket transport's fixed owner reduction order.
        for rank, _ in pending:
            hot = hot + self.outputs[rank][: rows * OUTPUT_ROW_BYTES].view(t.float32).reshape(rows, WIDTH)
        return hot, shared

    def close(self):
        """Never kill a worker whose native DMA completion might be uncertain."""
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.torch.cuda.synchronize()
            if self.hot_graphs is not None:
                self.hot_graphs.close()
                self.hot_graphs = None
            if self.pack_graphs is not None:
                self.pack_graphs.close()
                self.pack_graphs = None
            quiescent = []
            if len(self.inputs) == 2:
                for rank, connection in enumerate(self.connections):
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
