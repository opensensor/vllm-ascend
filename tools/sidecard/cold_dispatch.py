# SPDX-License-Identifier: Apache-2.0
"""Private independent Ascend expert workers and overlapped CUDA partials."""

import argparse
import json
import os
import socket
import struct
import subprocess
from pathlib import Path

from .cold_expert_probe import receive
from .cold_placement import load, lookup
from .expert_timing import WorkerTimers
from .layer_actor import guard_parent
from .layer_weights import MAX_CHUNK, NPU_LAYERS, TOP_K, WIDTH, NpuExperts

READY = b"EXPERT_READY\n"
HEADER = struct.Struct("!III")
INPUT_ROW_BYTES = WIDTH * 2 + TOP_K * 8
OUTPUT_ROW_BYTES = WIDTH * 4
ACTIVE_TIMEOUT = 300


def create_dispatch(config, cuda, torch, *, wait=True):
    specification = json.loads(Path(config).read_bytes())
    transport = specification.get("expert_transport", "socket")
    if transport == "direct-dma":
        from .dma_dispatch import DmaDispatch

        return DmaDispatch(config, cuda, torch, wait=wait)
    if transport != "socket":
        raise ValueError("unknown expert transport")
    return ColdDispatch(config, cuda, torch, wait=wait)


def worker(config, rank, fd, parent_pid):
    """Keep expert banks resident; CPU sockets permit indefinite idle time."""
    guard_parent(parent_pid)
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3":
        raise ValueError("expert worker requires only assigned physical chips 2,3")
    import torch
    import torch_npu

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.npu.set_device(rank)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    spec = json.loads(Path(config).read_bytes())
    hot = load(spec)
    layers = tuple(spec.get("cold_gate_layers", NPU_LAYERS))
    actor = NpuExperts(
        Path(spec["checkpoint"]),
        spec["checkpoint_sha256"],
        rank,
        torch,
        gate_override=spec["native_gate_override"],
        reduce_results=False,
        layers=layers,
    )
    actor.validate_repeat(Path(spec["output"]) / f"expert-repeat-rank{rank}.json", combine_results=False)
    host = torch.empty(MAX_CHUNK * INPUT_ROW_BYTES, dtype=torch.uint8, pin_memory=True)
    reply = torch.empty(MAX_CHUNK * OUTPUT_ROW_BYTES, dtype=torch.uint8, pin_memory=True)
    host_bytes, reply_bytes = memoryview(host.numpy()), memoryview(reply.numpy())
    hot_masks = {}
    for layer in layers:
        mask = torch.zeros(288, dtype=torch.bool)
        mask[hot[str(layer)]] = True
        hot_masks[layer] = mask
    profile = (
        WorkerTimers(Path(spec["output"]) / f"worker-profile-rank{rank}.jsonl", rank, torch.npu)
        if spec.get("expert_worker_profile", False)
        else None
    )
    expected = 0
    with socket.socket(fileno=fd) as connection, torch.inference_mode():
        connection.sendall(READY + spec["expert_placement_sha256"].encode())
        while True:
            tag = bytearray(HEADER.size)
            try:
                receive(connection, memoryview(tag))
            except EOFError:
                break
            sequence, layer, rows = HEADER.unpack(tag)
            if sequence != expected or layer not in layers or not 1 <= rows <= MAX_CHUNK:
                raise ValueError("invalid private expert request identity")
            receive(connection, host_bytes[: rows * INPUT_ROW_BYTES])
            if profile is not None:
                profile.begin(sequence, layer, rows)
            x_bytes, ids_bytes = rows * WIDTH * 2, rows * TOP_K * 4
            x = host[:x_bytes].view(torch.float16).reshape(rows, WIDTH)
            ids = host[x_bytes : x_bytes + ids_bytes].view(torch.int32).reshape(rows, TOP_K)
            weights = host[x_bytes + ids_bytes : rows * INPUT_ROW_BYTES].view(torch.float32).reshape(rows, TOP_K)
            if (
                bool(((ids < 0) | (ids >= 288)).any())
                or not bool(torch.isfinite(weights).all())
                or bool((weights < 0).any())
            ):
                raise ValueError("invalid cold expert route")
            if bool((weights[hot_masks[layer][ids.long()]] != 0).any()):
                raise ValueError("cold peer assigned nonzero work to a hot expert")
            device = f"npu:{rank}"
            if profile is not None:
                profile.mark("upload")
            device_x, device_ids, device_weights = x.to(device), ids.to(device), weights.to(device)
            if profile is not None:
                profile.mark("native")
            partial = actor(layer, device_x, device_ids, device_weights)
            if profile is not None:
                profile.mark("download")
            reply[: rows * OUTPUT_ROW_BYTES].view(torch.float32).reshape(rows, WIDTH).copy_(partial, non_blocking=True)
            if profile is not None:
                profile.mark("complete")
            torch.npu.synchronize()
            connection.sendall(tag)
            connection.sendall(reply_bytes[: rows * OUTPUT_ROW_BYTES])
            if profile is not None:
                profile.finish()
            expected += 1
    if profile is not None:
        profile.close()


class ColdDispatch:
    def __init__(self, config, cuda, torch, *, wait=True):
        self.spec = json.loads(Path(config).read_bytes())
        self.hot = load(self.spec)
        self.cuda, self.torch = cuda, torch
        self.connections, self.processes, self.logs = [], [], []
        self.sequences = [0, 0]
        self.lookups = {
            int(layer): torch.tensor(lookup(ids), dtype=torch.int32, device="cuda") for layer, ids in self.hot.items()
        }
        self.host = torch.empty(MAX_CHUNK * INPUT_ROW_BYTES, dtype=torch.uint8, pin_memory=True)
        self.replies = [torch.empty(MAX_CHUNK * OUTPUT_ROW_BYTES, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        self.device_replies = [torch.empty((MAX_CHUNK, WIDTH), dtype=torch.float32, device="cuda") for _ in range(2)]
        self.host_bytes = memoryview(self.host.numpy())
        self.reply_bytes = [memoryview(value.numpy()) for value in self.replies]
        try:
            self.start(config, wait=wait)
        except BaseException:
            self.close()
            raise

    def start(self, config, *, wait=True):
        env = dict(os.environ)
        env.update(json.loads(Path(self.spec["npu_environment"]).read_bytes()))
        env["ASCEND_RT_VISIBLE_DEVICES"] = "2,3"
        source = str(Path(config).resolve().parent / "source")
        env["PYTHONPATH"] = source + ":" + env.get("PYTHONPATH", "")
        for rank in range(2):
            parent, child = socket.socketpair()
            self.connections.append(parent)
            parent.settimeout(1800)
            log = (Path(self.spec["output"]) / f"npu-rank{rank}.log").open("x")
            self.logs.append(log)
            command = [
                self.spec["npu_python"],
                "-m",
                "tools.sidecard.cold_dispatch",
                "--config",
                str(config),
                "--rank",
                str(rank),
                "--fd",
                str(child.fileno()),
                "--parent-pid",
                str(os.getpid()),
            ]
            try:
                self.processes.append(
                    subprocess.Popen(
                        command, env=env, cwd=source, pass_fds=(child.fileno(),), stdout=log, stderr=subprocess.STDOUT
                    )
                )
            finally:
                child.close()
        if wait:
            self.ready()

    def ready(self):
        expected = READY + self.spec["expert_placement_sha256"].encode()
        for connection in self.connections:
            body = bytearray(len(expected))
            receive(connection, memoryview(body))
            if body != expected:
                raise RuntimeError("expert worker ownership readiness differs")
            connection.settimeout(ACTIVE_TIMEOUT)
        (Path(self.spec["output"]) / "npu-processes.json").write_text(
            json.dumps({"pids": [p.pid for p in self.processes], "physical_devices": [2, 3], "hccl": False}, indent=2)
            + "\n"
        )

    def __call__(self, layer, x, ids, weights):
        return self.compute(layer, x, ids, weights)[0]

    def with_shared(self, layer, x, ids, weights):
        return self.compute(layer.index, x, ids, weights, lambda: layer.dense_mlp(x, "mlp.shared_experts"))

    def compute(self, layer, x, ids, weights, shared_call=None):
        if layer not in NPU_LAYERS:
            routed = self.cuda(layer, x, ids, weights)
            return routed, shared_call() if shared_call is not None else None
        t, rows = self.torch, x.shape[0]
        if not 1 <= rows <= MAX_CHUNK:
            raise ValueError("cold dispatch exceeds its private buffers")
        local_ids = self.lookups[layer][ids.long()]
        owned = local_ids >= 0
        cold_weights = t.where(owned, 0, weights).float()
        x_bytes, id_bytes = rows * WIDTH * 2, rows * TOP_K * 4
        self.host[:x_bytes].view(t.float16).reshape(rows, WIDTH).copy_(x, non_blocking=True)
        self.host[x_bytes : x_bytes + id_bytes].view(t.int32).reshape(rows, TOP_K).copy_(ids, non_blocking=True)
        self.host[x_bytes + id_bytes : rows * INPUT_ROW_BYTES].view(t.float32).reshape(rows, TOP_K).copy_(
            cold_weights, non_blocking=True
        )
        t.cuda.synchronize()
        cpu_ids = self.host[x_bytes : x_bytes + id_bytes].view(t.int32).reshape(rows, TOP_K).numpy()
        cpu_weights = (
            self.host[x_bytes + id_bytes : rows * INPUT_ROW_BYTES].view(t.float32).reshape(rows, TOP_K).numpy()
        )
        pending = []
        for rank, connection in enumerate(self.connections):
            if ((cpu_ids >= rank * 144) & (cpu_ids < (rank + 1) * 144) & (cpu_weights > 0)).any():
                tag = HEADER.pack(self.sequences[rank], layer, rows)
                connection.sendall(tag)
                connection.sendall(self.host_bytes[: rows * INPUT_ROW_BYTES])
                pending.append((rank, tag))
        hot = self.cuda(layer, x, t.where(owned, local_ids, 0), t.where(owned, weights, 0))
        shared = shared_call() if shared_call is not None else None
        for rank, tag in pending:
            actual = bytearray(HEADER.size)
            receive(self.connections[rank], memoryview(actual))
            if actual != tag:
                raise ValueError("expert response belongs to another invocation")
            receive(self.connections[rank], self.reply_bytes[rank][: rows * OUTPUT_ROW_BYTES])
            self.device_replies[rank][:rows].copy_(
                self.replies[rank][: rows * OUTPUT_ROW_BYTES].view(t.float32).reshape(rows, WIDTH), non_blocking=True
            )
            self.sequences[rank] += 1
        active = {rank for rank, _ in pending}
        # Fixed owner order; each selected route retains its original weight.
        for rank in range(2):
            if rank in active:
                hot = hot + self.device_replies[rank][:rows]
        return hot, shared

    def close(self):
        for connection in self.connections:
            connection.close()
        for process in self.processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        for log in self.logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    worker(args.config, args.rank, args.fd, args.parent_pid)


if __name__ == "__main__":
    main()
