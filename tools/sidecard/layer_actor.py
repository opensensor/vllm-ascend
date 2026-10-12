# SPDX-License-Identifier: Apache-2.0
"""Serialized Ascend stage and CUDA prefix/suffix coordinator.

Only one request owns caches at a time. A position-zero packet clears every
stage's caches; peer failure makes the coordinator unusable. Two prompt chunks
can overlap across architectures; decode is serial. There is no cross-request
prefix reuse or speculative decoding. Optional KDA-span decode replay retains
eager DSA cache control and the same owner-local tensors and collectives.
"""

import argparse
import ctypes
import hashlib
import json
import os
import signal
import socket
import struct
import subprocess
import time
from datetime import timedelta
from pathlib import Path

from .layer_decode_graph import DecodeGraphs
from .layer_math import TextLayer, rms_norm, validate_config
from .layer_tp_gate import validate_tp
from .layer_weights import GPU_LAYERS, MAX_CHUNK, NPU_LAYERS, CudaExperts, NpuExperts, load_dense
from .stage_wire import MAX_BODY_BYTES, descriptors, metadata, receive_into, send

READY = b"LAYER_READY\n"
STARTUP_TIMEOUT = 1800
REQUEST_TIMEOUT = 300
PREFILL_PROGRESS_TOKENS = 2048
CONTROL = struct.Struct("!iii")
PARENT_DEATH_SIGNAL = 1


def guard_parent(parent_pid):
    """Exit this private Linux actor when its recorded coordinator dies."""
    library = ctypes.CDLL(None, use_errno=True)
    if library.prctl(PARENT_DEATH_SIGNAL, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot register private actor parent-death signal")
    if os.getppid() != parent_pid:
        raise RuntimeError("coordinator exited before its actor installed parent-death guard")


def receive_control(connection):
    body = bytearray()
    while len(body) < CONTROL.size:
        chunk = connection.recv(CONTROL.size - len(body))
        if not chunk:
            raise EOFError("middle-stage control peer closed")
        body.extend(chunk)
    return CONTROL.unpack(body)


def broadcast_stage(tensor, torch):
    """310P HCCL Broadcast returns unchanged receivers; use byte AllGather.

    Match the repository's qualified 310P pipeline transport. The sender is
    rank zero; returned storage remains private to the destination tensor.
    """
    packed = tensor.contiguous().flatten().view(torch.int8)
    gathered = torch.empty(packed.numel() * 2, dtype=torch.int8, device=tensor.device)
    torch.distributed.all_gather_into_tensor(gathered, packed)
    tensor.copy_(gathered[: packed.numel()].view(tensor.dtype).reshape(tensor.shape))


def pack_state(state, positions, torch):
    tensors = (*state, positions)
    return b"".join(value.detach().contiguous().cpu().numpy().tobytes() for value in tensors)


def unpack_state(body, rows, device, torch):
    fields, _ = descriptors(rows)
    dtypes = {"fp16": torch.float16, "fp32": torch.float32, "int64": torch.int64}
    values = []
    for field in fields:
        value = torch.frombuffer(body[field["offset"] : field["offset"] + field["bytes"]], dtype=dtypes[field["dtype"]])
        values.append(value.reshape(field["shape"]).to(device))
    return values


def npu_worker(args):
    """Load and qualify the complete middle stage before serving private packets."""
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3":
        raise ValueError("layer stage requires assigned physical NPU devices 2,3")
    import torch
    import torch_npu

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.npu.set_device(args.rank)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    native_sinkhorn = None
    if args.ops_library:
        torch.ops.load_library(args.ops_library)
        native_sinkhorn = torch.ops._C_ascend.mhc_sinkhorn_310
    torch.distributed.init_process_group(
        "hccl", init_method=args.init_method, rank=args.rank, world_size=2, timeout=timedelta(seconds=STARTUP_TIMEOUT)
    )
    collective_check = torch.full((4096,), args.rank + 1, dtype=torch.float32, device=f"npu:{args.rank}")
    torch.distributed.all_reduce(collective_check)
    if not bool((collective_check.cpu() == 3).all()):
        raise RuntimeError("Ascend FP32 expert reduction failed its startup SUM gate")
    # HCCL AllGather has its own lazy communicator. Initialize it on both
    # ranks before weights load; a receiver starting it while CUDA loads can
    # otherwise exhaust the socket bootstrap timeout before the sender joins.
    warmup = torch.full((32,), args.rank + 11, dtype=torch.int8, device=f"npu:{args.rank}")
    broadcast_stage(warmup, torch)
    if not bool((warmup.cpu() == 11).all()):
        raise RuntimeError("Ascend byte state broadcast failed its startup gate")
    root, device = Path(args.checkpoint), f"npu:{args.rank}"
    config = json.loads((root / "config.json").read_bytes())["text_config"]
    validate_config(config, args.max_context)
    experts = NpuExperts(
        root, args.identity, args.rank, torch, gate_override=args.native_gate_override, reduce_results=False
    )
    experts.validate_repeat(Path(args.output) / f"expert-repeat-rank{args.rank}.json")
    weights, _ = load_dense(root, NPU_LAYERS, device, torch, tp_rank=args.rank)
    layers = [
        TextLayer(
            layer,
            weights[layer],
            config,
            device,
            torch,
            experts,
            args.max_context,
            native_sinkhorn=native_sinkhorn,
            tp_rank=args.rank,
        )
        for layer in NPU_LAYERS
    ]
    validate_tp(layers, root, args.rank, torch, Path(args.output) / f"tp-rank{args.rank}-gate.json")
    decode_graphs = None
    if args.decode_graphs:
        with torch.inference_mode():
            decode_graphs = DecodeGraphs(layers, torch, device)
        (Path(args.output) / f"decode-graphs-rank{args.rank}.json").write_text(
            json.dumps(
                {
                    "spans": [[layer.index for layer in span] for span in decode_graphs.spans],
                    "captured": list(decode_graphs.graphs),
                    "scope": "single-token KDA spans; DSA eager",
                },
                indent=2,
            )
            + "\n"
        )
    control_connection = socket.socket(fileno=args.control_fd)
    receive_buffer = bytearray(MAX_BODY_BYTES)
    connection = socket.socket(fileno=args.fd) if args.rank == 0 else None
    if connection is not None:
        # Idle serving may last indefinitely. The coordinator limits active
        # requests; its private socket closing wakes this blocking receiver.
        connection.settimeout(None)
    torch.distributed.barrier()
    if connection is not None:
        connection.sendall(READY)
    sequence = 0
    expected_position = 0
    active_context_limit = None
    try:
        with torch.inference_mode():
            while True:
                packet = None
                if args.rank == 0:
                    try:
                        packet = receive_into(
                            connection, memoryview(receive_buffer), checkpoint_sha256=args.identity, sequence=sequence
                        )
                        fields, _ = descriptors(packet["rows"])
                        field = fields[-1]
                        positions_cpu = torch.frombuffer(
                            receive_buffer, dtype=torch.int64, count=packet["rows"], offset=field["offset"]
                        )
                        first_position = int(positions_cpu[0])
                        if positions_cpu.tolist() != list(range(first_position, first_position + packet["rows"])):
                            raise ValueError("stage only admits one contiguous request")
                        rows = packet["rows"]
                        context_limit = packet.get("request_context_limit", args.max_context)
                        control_connection.sendall(CONTROL.pack(rows, first_position, context_limit))
                    except (EOFError, OSError, ValueError):
                        control_connection.sendall(CONTROL.pack(-1, -1, -1))
                        raise
                else:
                    rows, first_position, context_limit = receive_control(control_connection)
                if rows < 0:
                    break
                if (
                    not 1 <= rows <= MAX_CHUNK
                    or not 0 <= first_position < args.max_context
                    or not 1 <= context_limit <= args.max_context
                    or first_position + rows > context_limit
                ):
                    raise ValueError("invalid stage cache admission")
                if first_position == 0:
                    active_context_limit = context_limit
                    for layer in layers:
                        layer.reset()
                        if not layer.linear:
                            layer.sparse_enabled = context_limit > layer.config["index_topk"]
                elif first_position != expected_position or context_limit != active_context_limit:
                    raise ValueError("stage positions disagree with its owner-local cache")
                if args.rank == 0:
                    state = unpack_state(memoryview(receive_buffer), rows, device, torch)[:4]
                else:
                    state = [
                        torch.empty(
                            field["shape"],
                            dtype=torch.float16 if field["dtype"] == "fp16" else torch.float32,
                            device=device,
                        )
                        for field in descriptors(rows)[0][:4]
                    ]
                for value in state:
                    broadcast_stage(value, torch)
                if decode_graphs is None:
                    for layer in layers:
                        state = layer.forward(*state)
                else:
                    state = decode_graphs.forward(state)
                if args.rank == 0:
                    if not all(bool(torch.isfinite(value).all().cpu()) for value in state):
                        raise FloatingPointError("Ascend stage produced nonfinite mHC state")
                    positions = torch.arange(first_position, first_position + rows, dtype=torch.int64)
                    body = pack_state(state, positions, torch)
                    send(connection, metadata(rows, sequence, args.identity, request_context_limit=context_limit), body)
                sequence += 1
                expected_position = first_position + rows
    finally:
        control_connection.close()
        if connection is not None:
            connection.close()
        torch.distributed.destroy_process_group()


class Coordinator:
    def __init__(self, config_path):
        import torch

        self.torch = torch
        torch.set_num_threads(2)
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.use_deterministic_algorithms(True)
        self.spec = json.loads(Path(config_path).read_bytes())
        if self.spec.get("routed_swiglu_limit") != 10.0 or not self.spec.get("native_gate_override"):
            raise ValueError("GLM requires authenticated clipped CUDA and Ascend routed kernels")
        override = json.loads(Path(self.spec["native_gate_override"]).read_bytes())
        if self.spec["cuda_backend"]["library_sha256"] != override["cuda_library_sha256"]:
            raise ValueError("CUDA routed kernel differs from the clipped build receipt")
        self.root = Path(self.spec["checkpoint"])
        self.config = json.loads((self.root / "config.json").read_bytes())["text_config"]
        self.max_context = self.spec.get("max_context", 2048)
        validate_config(self.config, self.max_context)
        self.identity = self.spec["checkpoint_sha256"]
        self.output = Path(self.spec["output"])
        self.output.mkdir(exist_ok=True)
        self.processes, self.logs = [], []
        self.connection, self.experts = None, None
        self.sequence, self.position = 0, 0
        self.request_context_limit = self.max_context
        self.metrics = {"prefix_ms": [], "middle_roundtrip_ms": [], "suffix_ms": []}
        self.failed = False
        self.stopping = False
        self.fingerprints = None
        try:
            self.start_npu(config_path)
            self.experts = CudaExperts(self.root, self.identity, self.spec["cuda_backend"], torch)
            weights, self.globals = load_dense(self.root, GPU_LAYERS, "cuda", torch, include_globals=True)
            self.layers = {
                layer: TextLayer(layer, weights[layer], self.config, "cuda", torch, self.experts, self.max_context)
                for layer in GPU_LAYERS
            }
            self.prefix_graphs, self.suffix_graphs = None, None
            if self.spec.get("cuda_decode_graphs", False):
                if not self.spec["cuda_backend"].get("torch_stream_experts", False):
                    raise ValueError("CUDA decode capture requires stream-aware expert kernels")
                with torch.inference_mode():
                    self.prefix_graphs = DecodeGraphs([self.layers[index] for index in range(18)], torch, "cuda")
                    self.suffix_graphs = DecodeGraphs([self.layers[index] for index in range(33, 45)], torch, "cuda")
                (self.output / "cuda-decode-graphs.json").write_text(
                    json.dumps(
                        {
                            "prefix": list(self.prefix_graphs.graphs),
                            "suffix": list(self.suffix_graphs.graphs),
                            "scope": "single-token KDA spans; DSA and vocabulary head eager",
                        },
                        indent=2,
                    )
                    + "\n"
                )
            # Merged QKV preparation temporarily holds both source and final
            # tensors. Release those dead allocator segments before admitting
            # cache/workspace; retaining them wastes several GiB on this model.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            self.connection.settimeout(STARTUP_TIMEOUT)
            received = bytearray()
            while len(received) < len(READY):
                chunk = self.connection.recv(len(READY) - len(received))
                if not chunk:
                    raise RuntimeError("Ascend worker exited during startup")
                received.extend(chunk)
            if bytes(received) != READY:
                raise RuntimeError("Ascend readiness differs from expected stage")
            self.connection.settimeout(REQUEST_TIMEOUT)
            self.receive_buffer = bytearray(MAX_BODY_BYTES)
            torch.cuda.synchronize()
            print(
                json.dumps(
                    dict(
                        kind="layer_runtime_loaded",
                        gpu_free_bytes=torch.cuda.mem_get_info()[0],
                        gpu_torch_bytes=torch.cuda.memory_allocated(),
                        cuda_expert_bytes=self.experts.oracle.allocated_bytes,
                    )
                ),
                flush=True,
            )
        except BaseException:
            self.close()
            raise

    def start_npu(self, config_path):
        environment = dict(os.environ)
        environment.update(json.loads(Path(self.spec["npu_environment"]).read_bytes()))
        environment["ASCEND_RT_VISIBLE_DEVICES"] = "2,3"
        source = str(Path(config_path).resolve().parent / "source")
        environment["PYTHONPATH"] = source + ":" + environment.get("PYTHONPATH", "")
        parent, child = socket.socketpair()
        control_leader, control_follower = socket.socketpair()
        self.connection = parent
        for rank in range(2):
            log = (self.output / f"npu-rank{rank}.log").open("w")
            self.logs.append(log)
            command = [
                self.spec["npu_python"],
                "-m",
                "tools.sidecard.layer_actor",
                "--checkpoint",
                str(self.root),
                "--identity",
                self.identity,
                "--rank",
                str(rank),
                "--init-method",
                self.spec["init_method"],
                "--max-context",
                str(self.max_context),
                "--output",
                str(self.output),
                "--fd",
                str(child.fileno()),
                "--control-fd",
                str(control_leader.fileno() if rank == 0 else control_follower.fileno()),
                "--native-gate-override",
                self.spec["native_gate_override"],
                "--parent-pid",
                str(os.getpid()),
            ]
            if self.spec.get("npu_ops_library"):
                command.extend(("--ops-library", self.spec["npu_ops_library"]))
            if self.spec.get("npu_decode_graphs", False):
                command.append("--decode-graphs")
            process = subprocess.Popen(
                command,
                cwd=source,
                env=environment,
                pass_fds=(child.fileno(), control_leader.fileno()) if rank == 0 else (control_follower.fileno(),),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            self.processes.append(process)
        child.close()
        control_leader.close()
        control_follower.close()
        (self.output / "npu-processes.json").write_text(
            json.dumps({"pids": [process.pid for process in self.processes], "physical_devices": [2, 3]}, indent=2)
            + "\n"
        )

    def reset(self, context_limit=None):
        if self.failed or self.stopping or any(process.poll() is not None for process in self.processes):
            raise RuntimeError("stage peer failed; restart the coordinator")
        for layer in self.layers.values():
            layer.reset()
            if not layer.linear:
                layer.sparse_enabled = (context_limit or self.max_context) > self.config["index_topk"]
        self.position = 0
        self.request_context_limit = context_limit or self.max_context

    def forward(self, tokens):
        t = self.torch
        if self.stopping:
            raise RuntimeError("coordinator is shutting down")
        if not 1 <= len(tokens) <= MAX_CHUNK or self.position + len(tokens) > self.max_context:
            raise ValueError("token chunk exceeds stage or context admission")
        try:
            with t.inference_mode():
                begin = time.perf_counter()
                hidden = t.nn.functional.embedding(t.tensor(tokens, device="cuda"), self.globals["embedding"])
                state = (hidden, None, None, None)
                if self.prefix_graphs is not None:
                    state = self.prefix_graphs.forward(state)
                else:
                    for index in range(18):
                        state = self.layers[index].forward(*state)
                t.cuda.synchronize()
                prefix_end = time.perf_counter()
                positions = t.arange(self.position, self.position + len(tokens), dtype=t.int64)
                prefix_body = pack_state(state, positions, t)
                send(
                    self.connection,
                    metadata(
                        len(tokens), self.sequence, self.identity, request_context_limit=self.request_context_limit
                    ),
                    prefix_body,
                )
                packet = receive_into(
                    self.connection,
                    memoryview(self.receive_buffer),
                    checkpoint_sha256=self.identity,
                    sequence=self.sequence,
                )
                if packet["rows"] != len(tokens):
                    raise ValueError("stage reply row count differs")
                if packet.get("request_context_limit") != self.request_context_limit:
                    raise ValueError("stage reply changed its request cache reservation")
                values = unpack_state(memoryview(self.receive_buffer), len(tokens), "cuda", t)
                if values[-1].cpu().tolist() != positions.tolist():
                    raise ValueError("stage reply positions differ")
                state = values[:4]
                middle_hash = (
                    hashlib.sha256(memoryview(self.receive_buffer)[: packet["payload_bytes"]]).hexdigest()
                    if self.fingerprints is not None
                    else None
                )
                middle_end = time.perf_counter()
                if self.suffix_graphs is not None:
                    state = self.suffix_graphs.forward(state)
                else:
                    for index in range(33, 45):
                        state = self.layers[index].forward(*state)
                hidden = rms_norm(state[0][-1:], self.globals["norm"], self.config["rms_norm_eps"], t)
                logits = t.nn.functional.linear(hidden, self.globals["head"]).float()
                if not bool(t.isfinite(logits).all().cpu()):
                    raise FloatingPointError("full-model logits contain nonfinite values")
                t.cuda.synchronize()
                end = time.perf_counter()
                if self.fingerprints is not None:
                    self.fingerprints.append(
                        {
                            "position": self.position,
                            "prefix_sha256": hashlib.sha256(prefix_body).hexdigest(),
                            "middle_sha256": middle_hash,
                            "logits_sha256": hashlib.sha256(logits.cpu().numpy().tobytes()).hexdigest(),
                        }
                    )
                self.sequence += 1
                self.position += len(tokens)
                for name, milliseconds in (
                    ("prefix_ms", (prefix_end - begin) * 1000),
                    ("middle_roundtrip_ms", (middle_end - prefix_end) * 1000),
                    ("suffix_ms", (end - middle_end) * 1000),
                ):
                    self.metrics[name].append(milliseconds)
                return logits[0]
        except BaseException:
            self.failed = True
            raise

    def prefill(self, tokens, *, overlap=True):
        """Two bounded prefix slots overlap prefill on the shared GPU/NPU.

        GPU prefix and suffix work are serialized on the same RTX. Only the
        next known prompt chunk is prepared while Ascend owns the preceding
        middle chunk. Each suffix completes in sequence; decode never enters
        this path. Packet storage is private until its consumer finishes.
        """
        if not overlap or len(tokens) <= MAX_CHUNK:
            logits = None
            for offset in range(0, len(tokens), MAX_CHUNK):
                logits = self.forward(tokens[offset : offset + MAX_CHUNK])
            return logits
        if self.position != 0 or len(tokens) > self.request_context_limit:
            raise ValueError("prefill pipeline requires a fresh admitted request")
        t = self.torch

        def prefix(chunk):
            start = time.perf_counter()
            hidden = t.nn.functional.embedding(t.tensor(chunk, device="cuda"), self.globals["embedding"])
            state = (hidden, None, None, None)
            for index in range(18):
                state = self.layers[index].forward(*state)
            t.cuda.synchronize()
            self.metrics["prefix_ms"].append((time.perf_counter() - start) * 1000)
            return state

        def submit(state, position, sequence):
            positions = t.arange(position, position + state[0].shape[0], dtype=t.int64)
            started = time.perf_counter()
            send(
                self.connection,
                metadata(state[0].shape[0], sequence, self.identity, request_context_limit=self.request_context_limit),
                pack_state(state, positions, t),
            )
            return (state[0].shape[0], position, sequence, started)

        def receive(pending):
            rows, position, sequence, started = pending
            packet = receive_into(
                self.connection, memoryview(self.receive_buffer), checkpoint_sha256=self.identity, sequence=sequence
            )
            if packet["rows"] != rows or packet.get("request_context_limit") != self.request_context_limit:
                raise ValueError("prefill reply changed its geometry or cache reservation")
            values = unpack_state(memoryview(self.receive_buffer), rows, "cuda", t)
            if values[-1].cpu().tolist() != list(range(position, position + rows)):
                raise ValueError("prefill reply positions differ")
            self.metrics["middle_roundtrip_ms"].append((time.perf_counter() - started) * 1000)
            return values[:4]

        def suffix(state, *, emit_logits):
            start = time.perf_counter()
            for index in range(33, 45):
                state = self.layers[index].forward(*state)
            logits = None
            if emit_logits:
                hidden = rms_norm(state[0][-1:], self.globals["norm"], self.config["rms_norm_eps"], t)
                logits = t.nn.functional.linear(hidden, self.globals["head"]).float()[0]
                if not bool(t.isfinite(logits).all().cpu()):
                    raise FloatingPointError("prefill logits contain nonfinite values")
            t.cuda.synchronize()
            self.metrics["suffix_ms"].append((time.perf_counter() - start) * 1000)
            return logits

        try:
            with t.inference_mode():
                sequence = self.sequence
                pending = submit(prefix(tokens[:MAX_CHUNK]), 0, sequence)
                sequence += 1
                for offset in range(MAX_CHUNK, len(tokens), MAX_CHUNK):
                    if self.stopping:
                        raise RuntimeError("prefill interrupted by coordinator shutdown")
                    next_state = prefix(tokens[offset : offset + MAX_CHUNK])
                    previous = receive(pending)
                    pending = submit(next_state, offset, sequence)
                    sequence += 1
                    suffix(previous, emit_logits=False)
                    processed = offset
                    if processed % PREFILL_PROGRESS_TOKENS == 0:
                        print(
                            json.dumps(
                                dict(
                                    kind="prefill_pipeline_progress",
                                    prompt_tokens=len(tokens),
                                    processed_tokens=processed,
                                )
                            ),
                            flush=True,
                        )
                logits = suffix(receive(pending), emit_logits=True)
                self.sequence = sequence
                self.position = len(tokens)
                return logits
        except BaseException:
            self.failed = True
            raise

    def close(self):
        if self.connection is not None:
            self.connection.close()
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
        for process in self.processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if self.experts is not None:
            self.experts.close()
        for log in self.logs:
            log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--init-method", required=True)
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--control-fd", type=int, required=True)
    parser.add_argument("--ops-library")
    parser.add_argument("--native-gate-override", required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--decode-graphs", action="store_true")
    args = parser.parse_args()
    guard_parent(args.parent_pid)
    npu_worker(args)


if __name__ == "__main__":
    main()
