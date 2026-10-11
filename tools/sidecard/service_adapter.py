# SPDX-License-Identifier: Apache-2.0
"""Explicit cold-load composition for the first CUDA/Ascend GLM service.

Loaded only by the configured vLLM worker extension. The existing native
loader, routing, shared expert, scaling and HCCL reduction retain their seams.
"""

import atexit
import functools
import json
import math
import os
import socket
import struct
import time
import uuid
from pathlib import Path

from .native_expert_probe import digest_file
from .placement import ExpertPlacement
from .readiness import receive_ready
from .reference import ExpertRequest, Route
from .runtime import SocketPeer
from .tensor_wire import TensorPacket, TensorSocketPeer
from .wire import PacketLimits, WireIdentity

CUDA_EXPERTS = 72
CHUNK_ROWS = 32
TOP_K = 8
HIDDEN_WIDTH = 4096
FIRST_LAYER = 3
LAST_LAYER = 45
SHADOW_RTOL = 0.002
SHADOW_ATOL = 0.002


def service_placement():
    layer = ("cuda:0",) * CUDA_EXPERTS
    for rank in range(1, 4):
        layer += (f"npu:{rank}",) * CUDA_EXPERTS
    return ExpertPlacement(("cuda:0", "npu:1", "npu:2", "npu:3"), (layer,) * (LAST_LAYER - FIRST_LAYER))


def local_request(execution_id, layer_id, hidden, ids, weights):
    """Discard masked zero routes without accepting positive foreign routes.

    The native EP seam remaps peer IDs to zero with weight zero. Such repeated
    IDs are padding, not router selections. Original nonzero slots stay intact.
    """
    if not hidden or len(hidden) > CHUNK_ROWS or len(hidden) != len(ids) or len(ids) != len(weights):
        raise ValueError("invalid sidecard service chunk geometry")
    if any(len(row) != HIDDEN_WIDTH or any(not math.isfinite(v) for v in row) for row in hidden):
        raise ValueError("invalid sidecard service hidden rows")
    routes, tokens = local_routes(ids, weights)
    if not routes:
        return None
    return ExpertRequest(
        execution_id, layer_id, "cuda:0", routes, tokens, tuple(tuple(hidden[token]) for token in tokens)
    )


def local_routes(ids, weights):
    if not 1 <= len(ids) <= CHUNK_ROWS or len(ids) != len(weights):
        raise ValueError("invalid sidecard service chunk geometry")
    routes = []
    for token, (experts, row_weights) in enumerate(zip(ids, weights, strict=True)):
        if len(experts) != TOP_K or len(row_weights) != TOP_K:
            raise ValueError("sidecard service requires top eight routing")
        selected = set()
        for slot, (expert, weight) in enumerate(zip(experts, row_weights, strict=True)):
            if not math.isfinite(weight) or weight < 0 or type(expert) is not int or not 0 <= expert < CUDA_EXPERTS:
                raise ValueError("invalid or foreign sidecard service route")
            if weight == 0:
                continue
            if expert in selected:
                raise ValueError("duplicate positive sidecard service expert")
            selected.add(expert)
            routes.append(Route(token, slot, expert))
    if not routes:
        return (), ()
    tokens = tuple(sorted({route.token for route in routes}))
    return tuple(routes), tokens


class SidecardSession:
    def __init__(self, config, model):
        self.report_path = Path(config["report_path"])
        self.layers = {}
        self.peer = None
        self.model = model
        self.tensor_transport = config.get("tensor_transport", False)
        self.report = {
            "schema_version": 1,
            "pid": os.getpid(),
            "profile": "ascend-backbone-cuda-experts",
            "cuda_attention": False,
            "checkpoint_sha256": config["checkpoint_sha256"],
            "cuda_experts_per_layer": CUDA_EXPERTS,
            "routed_layers": LAST_LAYER - FIRST_LAYER,
            "retains_npu_shadow_banks": True,
            "tensor_transport": self.tensor_transport,
        }
        placement = service_placement()
        identity = WireIdentity.for_placement(placement, config["checkpoint_sha256"])
        limits = PacketLimits(HIDDEN_WIDTH, CHUNK_ROWS, CHUNK_ROWS * TOP_K, 288, TOP_K)
        receipt = json.loads(Path(config["worker_ready_path"]).read_bytes())
        if (
            receipt["checkpoint_sha256"] != identity.checkpoint_sha256
            or receipt["placement_sha256"] != identity.placement_sha256
            or receipt["config_sha256"] != config["worker_config_sha256"]
            or receipt["resident_experts"] != CUDA_EXPERTS * (LAST_LAYER - FIRST_LAYER)
            or receipt.get("tensor_transport", False) != self.tensor_transport
        ):
            raise ValueError("CUDA daemon readiness differs from service admission")
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.settimeout(120)
            connection.connect(config["cuda_socket"])
            pid, uid, _ = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            if pid != receipt["pid"] or uid != os.getuid():
                raise ValueError("CUDA daemon socket credentials differ from readiness")
            receive_ready(connection, "cuda:0", identity, time.monotonic() + 120)
            peer_type = TensorSocketPeer if self.tensor_transport else SocketPeer
            self.peer = peer_type(connection, identity, limits, 120)
        except BaseException:
            connection.close()
            raise
        atexit.register(self.close)
        self.report["cuda_worker_pid"] = receipt["pid"]
        self.flush()

    def flush(self):
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({**self.report, "layers": self.layers}, indent=2) + "\n")
        temporary.replace(self.report_path)

    def close(self):
        if self.peer is not None:
            self.peer.close()
            self.peer = None

    def status(self):
        return {
            **self.report,
            "layers": self.layers,
            "cuda_worker_connected": self.peer is not None,
        }


class SidecardMoEMethod:
    def __init__(self, session, layer_id, original):
        self.session, self.layer_id, self.original = session, layer_id, original
        self.sequence = 0
        self.state = {
            "calls": 0,
            "cuda_routes": 0,
            "shadow_checks": 0,
            "shadow_max_absolute_error": 0.0,
            "shadow_max_nrmse": 0.0,
            "real_calls": 0,
            "real_cuda_routes": 0,
            "profile_calls": 0,
            "input_transfer_and_pending_npu_ms": 0.0,
            "exchange_ms": 0.0,
            "combine_ms": 0.0,
            "shadow_ms": 0.0,
        }
        session.layers[str(layer_id)] = self.state

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        import torch
        from vllm.forward_context import get_forward_context

        real_input = get_forward_context().attn_metadata is not None
        if not real_input:
            # vLLM memory profiling can use placeholder attention outputs.
            # Keep that internal, non-request pass on the original method;
            # actual generation always uses CUDA and fails on packet errors.
            self.state["profile_calls"] += 1
            return self.original.apply(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)

        if getattr(layer, "w2_shared_expert", None) is not None or shared_experts is not None:
            raise ValueError("sidecard service method accepts routed experts only")
        bank = layer.w2_experts
        if bank.local_expert_offset != 0 or bank.num_local_experts != CUDA_EXPERTS:
            raise ValueError("sidecard service rank-zero bank differs from CUDA ownership")
        if x.ndim != 2 or x.shape[1] != HIDDEN_WIDTH or topk_ids.shape != (x.shape[0], TOP_K):
            raise ValueError("sidecard service model boundary geometry differs")
        chunks = []
        tensor_transport = self.session.tensor_transport
        transferred = time.perf_counter()
        if tensor_transport:
            import numpy as np

            cpu_hidden = x.detach().to(device="cpu", dtype=torch.float16).contiguous()
            cpu_ids = topk_ids.detach().cpu().tolist()
            cpu_weights = topk_weights.detach().to(device="cpu", dtype=torch.float32).tolist()
            if not np.isfinite(cpu_hidden.numpy()).all():
                raise ValueError("invalid sidecard service hidden rows")
            self.state["input_transfer_and_pending_npu_ms"] += (time.perf_counter() - transferred) * 1000
        for start in range(0, x.shape[0], CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, x.shape[0])
            if tensor_transport:
                ids, weights = cpu_ids[start:stop], cpu_weights[start:stop]
            else:
                hidden = x[start:stop].detach().to(device="cpu", dtype=torch.float16).tolist()
                ids = topk_ids[start:stop].detach().cpu().tolist()
                weights = topk_weights[start:stop].detach().to(device="cpu", dtype=torch.float32).tolist()
            self.sequence += 1
            execution_id = f"{self.layer_id}-{self.sequence}-{uuid.uuid4().hex}"
            if tensor_transport:
                routes, tokens = local_routes(ids, weights)
                request = ExpertRequest(execution_id, self.layer_id, "cuda:0", routes, tokens, ()) if routes else None
            else:
                request = local_request(execution_id, self.layer_id, hidden, ids, weights)
            ordered = torch.zeros((stop - start, TOP_K, HIDDEN_WIDTH), dtype=torch.float32, device="cpu")
            if request is not None:
                started = time.perf_counter()
                if tensor_transport:
                    selected = cpu_hidden[start:stop].index_select(
                        0, torch.tensor(tokens, dtype=torch.long, device="cpu")
                    )
                    raw_reply = self.session.peer.exchange(TensorPacket(request, selected.numpy().tobytes()))
                    reply = raw_reply.control
                    outputs = torch.frombuffer(bytearray(raw_reply.tensor), dtype=torch.float32).reshape(
                        -1, HIDDEN_WIDTH
                    )
                else:
                    reply = self.session.peer.exchange(request)
                    outputs = torch.tensor(reply.outputs, dtype=torch.float32, device="cpu")
                self.state["exchange_ms"] += (time.perf_counter() - started) * 1000
                if reply.routes != request.routes:
                    raise ValueError("sidecard service reply route order differs")
                ordered[[route.token for route in reply.routes], [route.slot for route in reply.routes]] = outputs
                self.state["cuda_routes"] += len(reply.routes)
                if real_input:
                    self.state["real_cuda_routes"] += len(reply.routes)
            # The original native reducer multiplies and adds separately in
            # slot order. Outer GLM composition owns scale, shared and all-reduce.
            started = time.perf_counter()
            result = torch.zeros((stop - start, HIDDEN_WIDTH), dtype=torch.float32, device="cpu")
            route_weights = torch.tensor(weights, dtype=torch.float32, device="cpu")
            for slot in range(TOP_K):
                result.add_(ordered[:, slot] * route_weights[:, slot, None])
            chunks.append(result)
            self.state["combine_ms"] += (time.perf_counter() - started) * 1000
        actual = torch.cat(chunks).to(x.device)
        if real_input and self.state["shadow_checks"] < 3:
            started = time.perf_counter()
            expected = self.original.apply(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
            cpu_actual, cpu_expected = actual.cpu(), expected.float().cpu()
            error = cpu_actual - cpu_expected
            self.state["shadow_max_absolute_error"] = max(
                self.state["shadow_max_absolute_error"], float(error.abs().max())
            )
            self.state["shadow_max_nrmse"] = max(
                self.state["shadow_max_nrmse"],
                float(error.square().mean().sqrt() / cpu_expected.square().mean().sqrt().clamp_min(1e-8)),
            )
            torch.testing.assert_close(cpu_actual, cpu_expected, rtol=SHADOW_RTOL, atol=SHADOW_ATOL)
            self.state["shadow_checks"] += 1
            self.state["shadow_ms"] += (time.perf_counter() - started) * 1000
        self.state["calls"] += 1
        self.state["real_calls"] += int(real_input)
        if self.state["calls"] == 1 or self.state["real_calls"] == 1 or self.layer_id == LAST_LAYER - 1:
            self.session.flush()
        return actual


def install_model(model, config):
    import torch.distributed as dist

    parallel = model.vllm_config.parallel_config
    if (
        not model.vllm_config.model_config.enforce_eager
        or parallel.tensor_parallel_size != 4
        or parallel.pipeline_parallel_size != 1
        or model.vllm_config.speculative_config is not None
    ):
        raise ValueError("sidecard service requires eager target-only TP4/PP1")
    rank = dist.get_rank()
    if rank != 0:
        model._sidecard_rank_report = {
            "rank": rank,
            "pid": os.getpid(),
            "owner": f"npu:{rank}",
            "routed_layers": LAST_LAYER - FIRST_LAYER,
            "experts_per_layer": CUDA_EXPERTS,
        }
        return
    owners = []
    for name, module in model.named_modules():
        if getattr(module, "w2_experts", None) is None:
            continue
        parts = name.split(".")
        layer_id = int(parts[parts.index("layers") + 1])
        owners.append((layer_id, module))
    if sorted(layer for layer, _ in owners) != list(range(FIRST_LAYER, LAST_LAYER)):
        raise ValueError("sidecard service requires complete 42-layer native MoE coverage")
    session = SidecardSession(config, model)
    try:
        for layer_id, module in owners:
            module._method = SidecardMoEMethod(session, layer_id, module._method)
        model._sidecard_session = session
        session.flush()
    except BaseException:
        session.close()
        raise


def install_loader_hook(config_path):
    """Focused opt-in patch at the existing native loader completion boundary."""
    from vllm_ascend.model_loader.glm_native_int4 import GlmNativeInt4Loader

    original = GlmNativeInt4Loader.load_weights
    if getattr(original, "sidecard_hook", False):
        raise ValueError("sidecard loader hook already installed")
    config_path = Path(config_path).resolve()
    config_sha256 = digest_file(config_path)

    @functools.wraps(original)
    def load_weights(loader, model, model_config):
        if digest_file(config_path) != config_sha256:
            raise ValueError("sidecard service configuration changed during startup")
        config = json.loads(config_path.read_bytes())
        if Path(model_config.model).resolve() != Path(config["checkpoint_root"]).resolve():
            raise ValueError("sidecard service and vLLM checkpoint paths disagree")
        original(loader, model, model_config)
        install_model(model, config)

    load_weights.sidecard_hook = True
    GlmNativeInt4Loader.load_weights = load_weights


class SidecardStatusMixin:
    def sidecard_status(self):
        model = self.model_runner.model
        session = getattr(model, "_sidecard_session", None)
        if session is not None:
            return session.status()
        return getattr(model, "_sidecard_rank_report", {"error": "sidecard service was not installed"})
