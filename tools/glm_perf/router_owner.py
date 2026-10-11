# SPDX-License-Identifier: Apache-2.0
"""Cold-load GLM router ownership; local expert and shared partials stay local."""

from types import MethodType

import torch
import torch.distributed as dist

from .indexer_owner import MAX_CAPTURE_ROWS, validate_config

ROUTER_OPTIONS = ("router_owner_rank", "router_exchange_budget_bytes", "router_overlap_prefill")
ROUTE_BYTES = torch.int64.itemsize + torch.float32.itemsize


def read_options(options):
    present = [key in options for key in ROUTER_OPTIONS[:2]]
    if not any(present) and ROUTER_OPTIONS[2] not in options:
        return None
    if not all(present):
        raise ValueError("router ownership requires owner rank and exchange byte reservation")
    rank, budget = (options[key] for key in ROUTER_OPTIONS[:2])
    overlap = options.get(ROUTER_OPTIONS[2], False)
    if type(rank) is not int or rank < 0 or type(budget) is not int or budget <= 0 or type(overlap) is not bool:
        raise ValueError("router ownership requires valid integers and boolean overlap")
    return dict(owner_rank=rank, budget_bytes=budget, overlap_prefill=overlap)


class RouterExchange:
    """Prepared compact mixed-dtype packet; never convert IDs through float."""

    def __init__(self, *, group, rank, size, owner_rank, max_tokens, top_k, device, budget_bytes, backend=dist):
        if (
            any(type(v) is not int for v in (rank, size, owner_rank, max_tokens, top_k, budget_bytes))
            or size not in (2, 4)
            or not 0 <= rank < size
            or not 0 <= owner_rank < size
            or min(max_tokens, top_k, budget_bytes) < 1
        ):
            raise ValueError("invalid router exchange geometry")
        self.packet_bytes = max_tokens * top_k * ROUTE_BYTES
        self.required_bytes = (size + 1) * self.packet_bytes
        if self.required_bytes > budget_bytes:
            raise ValueError("router exchange exceeds explicit per-stage reservation")
        self.group, self.rank, self.size, self.owner_rank = group, rank, size, owner_rank
        self.max_tokens, self.top_k, self.backend = max_tokens, top_k, backend
        self.send = torch.empty(self.packet_bytes, dtype=torch.int8, device=device)
        self.receive = torch.empty(size * self.packet_bytes, dtype=torch.int8, device=device)
        self.pending = None

    def post(self, tokens, routes=None, *, overlap=False):
        if self.pending is not None:
            raise RuntimeError("router transfer must finish before scratch reuse")
        if type(tokens) is not int or not 1 <= tokens <= self.max_tokens:
            raise ValueError("router rows exceed prepared capacity")
        elements = tokens * self.top_k
        ids_bytes = elements * torch.int64.itemsize
        count = elements * ROUTE_BYTES
        if self.rank == self.owner_rank:
            if routes is None:
                raise ValueError("router owner must supply IDs and weights")
            ids, weights = routes
            if (
                ids.shape != (tokens, self.top_k)
                or weights.shape != ids.shape
                or ids.dtype != torch.int64
                or weights.dtype != torch.float32
                or ids.device != self.send.device
                or weights.device != self.send.device
            ):
                raise ValueError("router owner requires INT64 IDs and FP32 weights on the prepared device")
            self.send[:ids_bytes].view(torch.int64).view_as(ids).copy_(ids)
            self.send[ids_bytes:count].view(torch.float32).view_as(weights).copy_(weights)
        elif routes is not None:
            raise ValueError("router peer must skip gate and top-k")
        handle = self.backend.all_gather_into_tensor(
            self.receive[: self.size * count], self.send[:count], group=self.group, async_op=overlap
        )
        self.pending = handle, tokens

    def finish(self):
        if self.pending is None:
            raise RuntimeError("router transfer has not been posted")
        handle, tokens = self.pending
        if handle is not None:
            handle.wait()
        self.pending = None
        elements = tokens * self.top_k
        count = elements * ROUTE_BYTES
        start = self.owner_rank * count
        packet = self.receive[start : start + count]
        ids_bytes = elements * torch.int64.itemsize
        return (
            packet[:ids_bytes].view(torch.int64).view(tokens, self.top_k),
            packet[ids_bytes:].view(torch.float32).view(tokens, self.top_k),
        )

    def warmup(self):
        # Prepared storage only. No router or stateful expert is executed.
        self.send[:ROUTE_BYTES].zero_()
        self.backend.all_gather_into_tensor(
            self.receive[: self.size * ROUTE_BYTES], self.send[:ROUTE_BYTES], group=self.group
        )


def bind_moe(module, exchange, *, overlap_prefill=False):
    from vllm_ascend.models.glm5next_w2.moe import _all_reduce_routed, _ep_rank_size

    if hasattr(module, "_glm_router_exchange"):
        raise ValueError("router ownership is cold-load only")
    delegate = module.experts._w2_delegate
    if delegate is None or module.is_sequence_parallel:
        raise ValueError("router ownership requires the composed MoE with replicated tokens")

    def forward(self, hidden_states, already_sequence_parallel=False):
        tokens = hidden_states.shape[0]
        routes = None
        if exchange.rank == exchange.owner_rank:
            logits, _ = self.gate(hidden_states)
            routes = delegate.route(logits)
        overlap = overlap_prefill and tokens > MAX_CAPTURE_ROWS
        exchange.post(tokens, routes, overlap=overlap)
        shared = None
        if overlap and delegate.shared_expert is not None:
            shared = delegate.shared_expert(hidden_states)
        ids, weights = exchange.finish()
        routed = delegate.routed_experts_forward(hidden_states, ids, weights, reduce_results=False)
        if shared is None and delegate.shared_expert is not None:
            shared = delegate.shared_expert(hidden_states)
        combined = delegate.combine(routed, shared)
        if _ep_rank_size()[1] > 1:
            combined = _all_reduce_routed(combined)
        return combined.to(hidden_states.dtype)

    module.forward = MethodType(forward, module)
    module._glm_router_exchange = exchange


def plan_install(model, *, owner_rank, budget_bytes, overlap_prefill=False):
    from vllm_ascend.models.glm5next.model import Glm5NextMoE

    validate_config(model.vllm_config, owner_rank)
    if type(overlap_prefill) is not bool:
        raise ValueError("router overlap requires a boolean")
    owners = [module for module in model.modules() if isinstance(module, Glm5NextMoE)]
    top_k = model._glm_text_config.num_experts_per_token
    rows = model.vllm_config.scheduler_config.max_num_batched_tokens
    size = model.vllm_config.parallel_config.tensor_parallel_size
    required = (size + 1) * rows * top_k * ROUTE_BYTES
    if not owners or any(
        hasattr(module, "_glm_router_exchange")
        or getattr(module.experts, "_w2_delegate", None) is None
        or module.is_sequence_parallel
        for module in owners
    ):
        raise ValueError("router requires unbound composed GLM MoE instances")
    if type(budget_bytes) is not int or budget_bytes < required:
        raise ValueError("router exchange exceeds explicit per-stage reservation")
    return owners, required


def install(model, **options):
    from vllm.distributed import get_tp_group

    owners, required = plan_install(model, **options)
    tp = get_tp_group()
    if tp.world_size != model.vllm_config.parallel_config.tensor_parallel_size:
        raise ValueError("router TP group differs from configured partition")
    device = next(model.parameters()).device
    exchange = RouterExchange(
        group=tp.device_group,
        rank=tp.rank_in_group,
        size=tp.world_size,
        owner_rank=options["owner_rank"],
        max_tokens=model.vllm_config.scheduler_config.max_num_batched_tokens,
        top_k=model._glm_text_config.num_experts_per_token,
        device=device,
        budget_bytes=options["budget_bytes"],
    )
    exchange.warmup()
    if device.type == "npu":
        torch.npu.synchronize()
    for module in owners:
        bind_moe(module, exchange, overlap_prefill=options.get("overlap_prefill", False))
    return dict(moe_layers=len(owners), exchange_bytes=required, hardware_qualified=False)
