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
import types
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .native_expert_probe import digest_file
from .ordered_tensor import combine_compact
from .placement import ExpertPlacement
from .readiness import receive_ready
from .reference import ExpertRequest, Route
from .runtime import SocketPeer
from .tensor_wire import TensorPacket, TensorSocketPeer
from .wire import PacketLimits, WireIdentity

CUDA_EXPERTS = 72
ONE_CARD_CUDA_EXPERTS = 144
CHUNK_ROWS = 32
MAX_PACKET_TOKENS = 640
TOP_K = 8
HIDDEN_WIDTH = 4096
FIRST_LAYER = 3
LAST_LAYER = 45
SHADOW_RTOL = 0.002
SHADOW_ATOL = 0.002
OVERLAP_PREFILL_MIN_TOKENS = 32


def service_placement(*, one_card=False):
    if type(one_card) is not bool:
        raise ValueError("one_card must be boolean")
    if one_card:
        layer = ("cuda:0",) * ONE_CARD_CUDA_EXPERTS + ("npu:0",) * CUDA_EXPERTS + ("npu:1",) * CUDA_EXPERTS
        return ExpertPlacement(("cuda:0", "npu:0", "npu:1"), (layer,) * (LAST_LAYER - FIRST_LAYER))
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


def local_routes(ids, weights, *, max_tokens=CHUNK_ROWS, cuda_experts=CUDA_EXPERTS):
    if cuda_experts not in (CUDA_EXPERTS, ONE_CARD_CUDA_EXPERTS):
        raise ValueError("unsupported CUDA service partition")
    if not 1 <= len(ids) <= max_tokens <= MAX_PACKET_TOKENS or len(ids) != len(weights):
        raise ValueError("invalid sidecard service chunk geometry")
    routes = []
    for token, (experts, row_weights) in enumerate(zip(ids, weights, strict=True)):
        if len(experts) != TOP_K or len(row_weights) != TOP_K:
            raise ValueError("sidecard service requires top eight routing")
        selected = set()
        for slot, (expert, weight) in enumerate(zip(experts, row_weights, strict=True)):
            if not math.isfinite(weight) or weight < 0 or type(expert) is not int or not 0 <= expert < cuda_experts:
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
        self.executor = None
        self.model = model
        self.one_card = config.get("one_card", False)
        if type(self.one_card) is not bool:
            raise ValueError("one_card must be boolean")
        self.cuda_experts = ONE_CARD_CUDA_EXPERTS if self.one_card else CUDA_EXPERTS
        self.overlap_shadows = self.one_card
        self.tensor_transport = config.get("tensor_transport", False)
        self.reply_fp16 = config.get("reply_fp16", False)
        self.partial_reduce = config.get("partial_reduce", False)
        if type(self.partial_reduce) is not bool or (
            self.partial_reduce and (self.reply_fp16 or not self.tensor_transport)
        ):
            raise ValueError("partial combination requires tensor transport and FP32 replies")
        if type(self.reply_fp16) is not bool or (self.reply_fp16 and not self.tensor_transport):
            raise ValueError("FP16 replies require tensor transport")
        self.packet_tokens = config.get("packet_tokens", CHUNK_ROWS)
        if type(self.packet_tokens) is not int or not 1 <= self.packet_tokens <= MAX_PACKET_TOKENS:
            raise ValueError("invalid sidecard packet token limit")
        if not self.tensor_transport and self.packet_tokens != CHUNK_ROWS:
            raise ValueError("large service packets require tensor transport")
        self.report = {
            "schema_version": 1,
            "pid": os.getpid(),
            "profile": "ascend-backbone-cuda-experts",
            "cuda_attention": False,
            "checkpoint_sha256": config["checkpoint_sha256"],
            "cuda_experts_per_layer": self.cuda_experts,
            "routed_layers": LAST_LAYER - FIRST_LAYER,
            "retains_npu_shadow_banks": not self.one_card,
            "tensor_transport": self.tensor_transport,
            "packet_tokens": self.packet_tokens,
            "reply_fp16": self.reply_fp16,
            "partial_reduce": self.partial_reduce,
        }
        if self.one_card:
            self.report["one_card"] = True
            self.report["packed_cuda_gate"] = config["packed_cuda_gate"]
            self.report["npu_devices"] = config.get("npu_devices", [0, 1])
            self.report["overlaps_native_cuda"] = True
        placement = service_placement(one_card=self.one_card)
        identity = WireIdentity.for_placement(placement, config["checkpoint_sha256"])
        limits = PacketLimits(HIDDEN_WIDTH, self.packet_tokens, self.packet_tokens * TOP_K, 288, TOP_K)
        receipt = json.loads(Path(config["worker_ready_path"]).read_bytes())
        if (
            receipt["checkpoint_sha256"] != identity.checkpoint_sha256
            or receipt["placement_sha256"] != identity.placement_sha256
            or receipt["config_sha256"] != config["worker_config_sha256"]
            or receipt["resident_experts"] != self.cuda_experts * (LAST_LAYER - FIRST_LAYER)
            or receipt.get("tensor_transport", False) != self.tensor_transport
            or receipt.get("max_tokens", CHUNK_ROWS) != self.packet_tokens
            or receipt.get("max_routes", CHUNK_ROWS * TOP_K) != limits.max_routes
            or receipt.get("reply_fp16", False) != self.reply_fp16
            or receipt.get("partial_reduce", False) != self.partial_reduce
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
            self.peer = (
                TensorSocketPeer(
                    connection, identity, limits, 120, reply_fp16=self.reply_fp16, partial_reduce=self.partial_reduce
                )
                if self.tensor_transport
                else SocketPeer(connection, identity, limits, 120)
            )
        except BaseException:
            connection.close()
            raise
        atexit.register(self.close)
        if self.one_card:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sidecard-cuda")
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
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None

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
        self.overlap_native = getattr(session, "one_card", False)
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
            "native_overlap_calls": 0,
            "native_dispatch_host_ms": 0.0,
            "overlap_decode_shadow_checks": 0,
            "overlap_prefill_shadow_checks": 0,
            "overlap_shadow_max_absolute_error": 0.0,
            "overlap_shadow_max_nrmse": 0.0,
        }
        session.layers[str(layer_id)] = self.state

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input, *, native_dispatch=None):
        import torch
        from vllm.forward_context import get_forward_context

        real_input = get_forward_context().attn_metadata is not None
        if not real_input:
            # vLLM memory profiling can use placeholder attention outputs.
            # Keep that internal, non-request pass on the original method;
            # actual generation always uses CUDA and fails on packet errors.
            self.state["profile_calls"] += 1
            if getattr(self.session, "one_card", False):
                return native_dispatch() if native_dispatch is not None else torch.zeros_like(x, dtype=torch.float32)
            return self.original.apply(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)

        if getattr(layer, "w2_shared_expert", None) is not None or shared_experts is not None:
            raise ValueError("sidecard service method accepts routed experts only")
        bank = layer.w2_experts
        expected_offset = ONE_CARD_CUDA_EXPERTS if getattr(self.session, "one_card", False) else 0
        if bank.local_expert_offset != expected_offset or bank.num_local_experts != CUDA_EXPERTS:
            raise ValueError("sidecard service rank-zero bank differs from CUDA ownership")
        if x.ndim != 2 or x.shape[1] != HIDDEN_WIDTH or topk_ids.shape != (x.shape[0], TOP_K):
            raise ValueError("sidecard service model boundary geometry differs")
        if native_dispatch is not None and (not self.overlap_native or self.session.executor is None):
            raise ValueError("native overlap requires the one-card CUDA session")
        native_result = None
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
        for start in range(0, x.shape[0], self.session.packet_tokens):
            stop = min(start + self.session.packet_tokens, x.shape[0])
            if tensor_transport:
                ids, weights = cpu_ids[start:stop], cpu_weights[start:stop]
            else:
                hidden = x[start:stop].detach().to(device="cpu", dtype=torch.float16).tolist()
                ids = topk_ids[start:stop].detach().cpu().tolist()
                weights = topk_weights[start:stop].detach().to(device="cpu", dtype=torch.float32).tolist()
            self.sequence += 1
            execution_id = f"{self.layer_id}-{self.sequence}-{uuid.uuid4().hex}"
            if tensor_transport:
                routes, tokens = local_routes(
                    ids,
                    weights,
                    max_tokens=self.session.packet_tokens,
                    cuda_experts=getattr(self.session, "cuda_experts", CUDA_EXPERTS),
                )
                request = ExpertRequest(execution_id, self.layer_id, "cuda:0", routes, tokens, ()) if routes else None
            else:
                request = local_request(execution_id, self.layer_id, hidden, ids, weights)
            if not tensor_transport:
                ordered = torch.zeros((stop - start, TOP_K, HIDDEN_WIDTH), dtype=torch.float32, device="cpu")
            raw_outputs = b""
            reply_routes = ()
            if request is not None:
                started = time.perf_counter()
                if tensor_transport:
                    selected = cpu_hidden[start:stop].index_select(
                        0, torch.tensor(tokens, dtype=torch.long, device="cpu")
                    )
                    raw_weights = (
                        np.asarray([weights[r.token][r.slot] for r in request.routes], dtype="<f4").tobytes()
                        if self.session.partial_reduce
                        else b""
                    )
                    packet = TensorPacket(
                        request,
                        selected.numpy().tobytes(),
                        partial_reduce=self.session.partial_reduce,
                        route_weights=raw_weights,
                    )
                    if native_dispatch is not None and native_result is None:
                        # Only immutable host packet I/O runs in this thread.
                        # Input copies and every NPU call remain on the caller's
                        # stream, before and after this independent CUDA RPC.
                        future = self.session.executor.submit(self.session.peer.exchange, packet)
                        try:
                            dispatched = time.perf_counter()
                            native_result = native_dispatch()
                            self.state["native_dispatch_host_ms"] += (time.perf_counter() - dispatched) * 1000
                            self.state["native_overlap_calls"] += 1
                            raw_reply = future.result()
                        except BaseException:
                            self.session.close()
                            raise
                    else:
                        raw_reply = self.session.peer.exchange(packet)
                    reply = raw_reply.control
                    raw_outputs = raw_reply.tensor
                    reply_routes = reply.routes
                else:
                    reply = self.session.peer.exchange(request)
                    outputs = torch.tensor(reply.outputs, dtype=torch.float32, device="cpu")
                self.state["exchange_ms"] += (time.perf_counter() - started) * 1000
                if reply.routes != request.routes:
                    raise ValueError("sidecard service reply route order differs")
                if not tensor_transport:
                    ordered[[route.token for route in reply.routes], [route.slot for route in reply.routes]] = outputs
                self.state["cuda_routes"] += len(reply.routes)
                if real_input:
                    self.state["real_cuda_routes"] += len(reply.routes)
            # The original native reducer multiplies and adds separately in
            # slot order. Outer GLM composition owns scale, shared and all-reduce.
            started = time.perf_counter()
            if tensor_transport:
                if self.session.partial_reduce:
                    combined = np.zeros((stop - start, HIDDEN_WIDTH), dtype=np.float32)
                    if request is not None:
                        combined[list(request.token_ids)] = np.frombuffer(raw_outputs, dtype="<f4").reshape(
                            -1, HIDDEN_WIDTH
                        )
                    result = torch.from_numpy(combined)
                else:
                    result = torch.from_numpy(
                        combine_compact(
                            reply_routes, raw_outputs, weights, HIDDEN_WIDTH, reply_fp16=self.session.reply_fp16
                        )
                    )
            else:
                result = torch.zeros((stop - start, HIDDEN_WIDTH), dtype=torch.float32, device="cpu")
                route_weights = torch.tensor(weights, dtype=torch.float32, device="cpu")
                for slot in range(TOP_K):
                    result.add_(ordered[:, slot] * route_weights[:, slot, None])
            chunks.append(result)
            self.state["combine_ms"] += (time.perf_counter() - started) * 1000
        actual = torch.cat(chunks).to(x.device)
        if native_dispatch is not None:
            if native_result is None:
                native_result = native_dispatch()
            actual = native_result.float() + actual
            # Qualify the overlapped schedule against the same serialized
            # composition on real inputs. Cover three short calls and one
            # bulk prefill per layer; steady requests do not repeat this work.
            prefill = x.shape[0] >= OVERLAP_PREFILL_MIN_TOKENS
            shadow_key = "overlap_prefill_shadow_checks" if prefill else "overlap_decode_shadow_checks"
            shadow_limit = 1 if prefill else 3
            if getattr(self.session, "overlap_shadows", False) and self.state[shadow_key] < shadow_limit:
                started = time.perf_counter()
                serial_native = native_dispatch()
                serial_cuda = self.apply(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
                expected = serial_native.float() + serial_cuda
                cpu_actual, cpu_expected = actual.cpu(), expected.float().cpu()
                error = cpu_actual - cpu_expected
                self.state["overlap_shadow_max_absolute_error"] = max(
                    self.state["overlap_shadow_max_absolute_error"], float(error.abs().max())
                )
                self.state["overlap_shadow_max_nrmse"] = max(
                    self.state["overlap_shadow_max_nrmse"],
                    float(error.square().mean().sqrt() / cpu_expected.square().mean().sqrt().clamp_min(1e-8)),
                )
                torch.testing.assert_close(cpu_actual, cpu_expected, rtol=SHADOW_RTOL, atol=SHADOW_ATOL)
                self.state[shadow_key] += 1
                self.state["shadow_ms"] += (time.perf_counter() - started) * 1000
        if real_input and not getattr(self.session, "one_card", False) and self.state["shadow_checks"] < 3:
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
        or parallel.tensor_parallel_size != (2 if config.get("one_card", False) else 4)
        or parallel.pipeline_parallel_size != 1
        or model.vllm_config.speculative_config is not None
    ):
        raise ValueError("sidecard service requires eager target-only TP4/PP1")
    rank = dist.get_rank()
    if config.get("one_card", False):
        install_one_card_model(model, config, rank)
        return
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


def install_one_card_model(model, config, rank):
    from .one_card import npu_range

    start, stop = npu_range(rank)
    owners = []
    for name, module in model.named_modules():
        bank = getattr(module, "w2_experts", None)
        if bank is None:
            continue
        if (bank.local_expert_offset, bank.num_local_experts) != (start, stop - start):
            raise ValueError("loaded NPU bank differs from one-card admission")
        parts = name.split(".")
        owners.append((int(parts[parts.index("layers") + 1]), module))
    if sorted(layer for layer, _ in owners) != list(range(FIRST_LAYER, LAST_LAYER)):
        raise ValueError("one-card sidecard requires complete native MoE coverage")
    session = SidecardSession(config, model) if rank == 0 else None
    try:
        for layer_id, module in owners:
            native = module._method
            cuda = SidecardMoEMethod(session, layer_id, native) if session is not None else None
            module.routed_experts_forward = types.MethodType(one_card_routed(native, cuda), module)
        if session is not None:
            model._sidecard_session = session
            session.flush()
        model._sidecard_rank_report = {
            "rank": rank,
            "pid": os.getpid(),
            "owner": f"npu:{rank}",
            "expert_range": [start, stop],
            "routed_layers": len(owners),
            "physical_device": config.get("npu_devices", [0, 1])[rank],
        }
    except BaseException:
        if session is not None:
            session.close()
        raise


def one_card_routed(native, cuda):
    """Compose explicitly owned contributions before GLM's existing all-reduce."""

    def routed(module, hidden_states, topk_ids, topk_weights, *, experts=None, reduce_results=True):
        import torch
        from vllm.distributed import tensor_model_parallel_all_reduce

        bank = module.w2_experts
        if experts is not None and experts is not bank:
            raise ValueError("one-card dispatch requires the admitted native bank")
        start, stop = bank.local_expert_offset, bank.local_expert_offset + bank.num_local_experts
        local = (topk_ids >= start) & (topk_ids < stop)
        ids = torch.where(local, topk_ids, torch.full_like(topk_ids, start))
        weights = topk_weights * local.to(topk_weights.dtype)
        layer = types.SimpleNamespace(w2_experts=bank, w2_shared_expert=None)
        native_dispatch = functools.partial(native.apply, layer, hidden_states, weights, ids, None, None)
        if cuda is None:
            actual = native_dispatch()
        else:
            local = (topk_ids >= 0) & (topk_ids < ONE_CARD_CUDA_EXPERTS)
            ids = torch.where(local, topk_ids, torch.zeros_like(topk_ids))
            weights = topk_weights * local.to(topk_weights.dtype)
            if getattr(cuda, "overlap_native", False):
                actual = cuda.apply(layer, hidden_states, weights, ids, None, None, native_dispatch=native_dispatch)
            else:
                actual = native_dispatch().float() + cuda.apply(layer, hidden_states, weights, ids, None, None)
        return tensor_model_parallel_all_reduce(actual) if reduce_results else actual

    return routed


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
        if config.get("one_card", False):
            import torch.distributed as dist

            from .one_card import prepare_banks

            manifest = json.loads((Path(config["checkpoint_root"]) / "native-layout.json").read_bytes())
            if (
                manifest.get("deployment_profile") != "one-card-sidecard-w2"
                or manifest.get("sidecard_cuda_experts") != ONE_CARD_CUDA_EXPERTS
                or digest_file(Path(config["checkpoint_root"]) / "native-layout.json") != config["checkpoint_sha256"]
            ):
                raise ValueError("one-card native checkpoint differs from admitted placement")
            prepare_banks(model, dist.get_rank())
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
