# SPDX-License-Identifier: Apache-2.0
"""Cold-load candidate: one complete sparse indexer per tensor-parallel group.

The indexer state stays with its fixed owner for the entire model lifetime.
Every TP rank still participates in the result collective. Attention KV and
head-sharded attention remain local; no cache allocation is removed here.
"""

from types import MethodType

import torch
import torch.distributed as dist

MAX_CAPTURE_ROWS = 8
WARMUP_BYTES = 32
SUPPORTED_TP_SIZES = (2, 4)
OWNER_OPTIONS = ("indexer_owner_rank", "indexer_exchange_budget_bytes", "indexer_overlap_prefill")


def read_options(options):
    present = [key in options for key in OWNER_OPTIONS[:2]]
    if not any(present) and OWNER_OPTIONS[2] not in options:
        return None
    if not all(present):
        raise ValueError("indexer ownership requires both its local rank and separate exchange byte reservation")
    rank, budget = (options[key] for key in OWNER_OPTIONS[:2])
    if type(rank) is not int or rank < 0 or type(budget) is not int or budget <= 0:
        raise ValueError("indexer owner rank and exchange byte reservation require valid integers")
    result = dict(owner_rank=rank, budget_bytes=budget)
    if OWNER_OPTIONS[2] in options:
        overlap = options[OWNER_OPTIONS[2]]
        if type(overlap) is not bool:
            raise ValueError("indexer prefill overlap requires a boolean")
        result["overlap_prefill"] = overlap
    return result


def validate_config(config, owner_rank):
    """Reject incompatible distributed/capture paths before preparing banks."""
    parallel = config.parallel_config
    size = parallel.tensor_parallel_size
    if size not in SUPPORTED_TP_SIZES or type(owner_rank) is not int or not 0 <= owner_rank < size:
        raise ValueError("indexer owner requires a local rank in a TP2 or TP4 group")
    if any(
        getattr(parallel, name, 1) != 1
        for name in ("data_parallel_size", "decode_context_parallel_size", "prefill_context_parallel_size")
    ) or getattr(parallel, "use_sequence_parallel_moe", False):
        raise ValueError("indexer owner initially requires DP1 and replicated token inputs")
    if config.speculative_config is not None or getattr(config, "kv_transfer_config", None) is not None:
        raise ValueError("indexer owner requires target-only execution without KV transfer")
    if config.model_config.enforce_eager:
        return
    compilation = config.compilation_config
    mode = getattr(compilation.cudagraph_mode, "name", compilation.cudagraph_mode)
    sizes = compilation.cudagraph_capture_sizes
    if (
        mode != "FULL_DECODE_ONLY"
        or not sizes
        or any(type(rows) is not int or not 1 <= rows <= MAX_CAPTURE_ROWS for rows in sizes)
    ):
        raise ValueError("indexer owner requires eager or FULL_DECODE_ONLY capture sizes within 1..8")


class IndexerResultExchange:
    """One prepared byte AllGather slab shared by sequential stage indexers.

    Use the existing TP device process group. Its ordered local ranks, rather
    than global rank zero, define the result owner. Only owner bytes are read;
    peer input bytes are irrelevant. A byte view avoids INT32 conversions.
    """

    def __init__(
        self,
        *,
        group,
        rank,
        size,
        owner_rank,
        max_tokens,
        width,
        device,
        budget_bytes,
        backend=dist,
        overlap_prefill=False,
    ):
        if (
            any(type(v) is not int for v in (rank, size, owner_rank, max_tokens, width, budget_bytes))
            or size < 2
            or not 0 <= rank < size
            or not 0 <= owner_rank < size
            or min(max_tokens, width, budget_bytes) < 1
        ):
            raise ValueError("invalid indexer result exchange geometry")
        self.required_bytes = size * max_tokens * width * torch.int32.itemsize
        if self.required_bytes > budget_bytes:
            raise ValueError("indexer result exchange exceeds its explicit per-rank byte reservation")
        if type(overlap_prefill) is not bool:
            raise ValueError("indexer prefill overlap requires a boolean")
        self.rank, self.size, self.owner_rank = rank, size, owner_rank
        self.max_tokens, self.width = max_tokens, width
        self.group, self.backend = group, backend
        self.overlap_prefill = overlap_prefill
        self.pending = None
        self.storage = torch.empty(self.required_bytes, dtype=torch.int8, device=device)

    def warmup(self, synchronize=None):
        """Initialize the collective outside graph capture and cache profiling."""
        count = min(WARMUP_BYTES, self.max_tokens * self.width * torch.int32.itemsize)
        source = torch.zeros(count, dtype=torch.int8, device=self.storage.device)
        self.backend.all_gather_into_tensor(self.storage[: self.size * source.numel()], source, group=self.group)
        if synchronize is not None:
            synchronize()

    def share(self, buffer, tokens):
        if self.pending is not None:
            raise RuntimeError("indexer result must finish before the next layer reuses stage scratch")
        if (
            type(tokens) is not int
            or not 1 <= tokens <= self.max_tokens
            or buffer.ndim != 2
            or buffer.shape[0] < tokens
            or buffer.shape[1] != self.width
            or buffer.dtype != torch.int32
            or buffer.device != self.storage.device
            or not buffer.is_contiguous()
        ):
            raise ValueError("indexer result must fit the prepared contiguous INT32 buffer")
        source = buffer[:tokens].view(torch.int8).reshape(-1)
        receive = self.storage[: self.size * source.numel()]
        if self.overlap_prefill and tokens > MAX_CAPTURE_ROWS:
            handle = self.backend.all_gather_into_tensor(receive, source, group=self.group, async_op=True)
            self.pending = handle, buffer, tokens
        else:
            self.backend.all_gather_into_tensor(receive, source, group=self.group)
            self._publish(buffer, tokens)
        return buffer

    def _publish(self, buffer, tokens):
        count = tokens * self.width * torch.int32.itemsize
        start = self.owner_rank * count
        # Keep the original shared index buffer and its graph-stable address.
        # Returning a slab view would bypass MLA's separately retained buffer.
        if self.rank != self.owner_rank:
            buffer[:tokens].copy_(self.storage[start : start + count].view(torch.int32).view(tokens, self.width))

    def finish(self):
        """Join after useful local preprocessing, before any indices consumer."""
        if self.pending is not None:
            handle, buffer, tokens = self.pending
            handle.wait()
            self._publish(buffer, tokens)
            self.pending = None


def bind_indexer(indexer, exchange):
    """Bind one loaded instance after permanent arithmetic hooks are installed."""
    if hasattr(indexer, "_glm_indexer_result_owner"):
        raise ValueError("indexer ownership is cold-load only; an instance cannot be rebound")
    buffer = indexer.topk_indices_buffer
    if buffer.dtype != torch.int32 or not buffer.is_contiguous() or buffer.shape[1] != exchange.width:
        raise ValueError("indexer lacks the prepared shared INT32 selection buffer")
    if getattr(indexer.indexer_op, "topk_indices_buffer", None) is not buffer:
        raise ValueError("indexer and selection operator must share the MLA index buffer")
    original = indexer.forward

    def forward(self, hidden_states, qr, positions, rotary_emb):
        tokens = hidden_states.shape[0]
        if not 1 <= tokens <= exchange.max_tokens or qr.shape[0] != tokens or positions.shape[0] != tokens:
            raise ValueError("indexer owner requires identical prepared token geometry on every TP rank")
        if exchange.rank == exchange.owner_rank:
            selected = original(hidden_states, qr, positions, rotary_emb)
            if selected is not self.topk_indices_buffer:
                raise RuntimeError("owned indexer must publish the original MLA selection buffer")
        return exchange.share(self.topk_indices_buffer, tokens)

    indexer.forward = MethodType(forward, indexer)
    indexer._glm_indexer_result_owner = exchange


def attention_backends(model, indexers):
    from vllm_ascend._310p.attention.mla_v1_310 import AscendMLAImpl310

    backends = []
    seen = set()
    for module in model.modules():
        backend = getattr(module, "impl", None)
        if isinstance(backend, AscendMLAImpl310) and id(backend) not in seen:
            if any(backend.glm_indexer is indexer for indexer in indexers):
                backends.append(backend)
                seen.add(id(backend))
    consumers = [backend.glm_indexer for backend in backends]
    if len(consumers) != len(indexers) or any(
        sum(owner is indexer for owner in consumers) != 1 for indexer in indexers
    ):
        raise ValueError("prefill overlap requires one 310P MLA frontend consumer per owned indexer")
    if any(hasattr(backend, "_glm_indexer_prefill_exchange") for backend in backends):
        raise ValueError("indexer prefill overlap cannot be rebound")
    return backends


def bind_preprocess(backend, exchange):
    original = backend._mla_preprocess

    def preprocess(self, *args, **kwargs):
        result = original(*args, **kwargs)
        # The 310P frontend first invokes the indexer, then projects local Q/K
        # and writes MLA KV. Its attention cores read indices after this return.
        exchange.finish()
        return result

    backend._mla_preprocess = MethodType(preprocess, backend)
    backend._glm_indexer_prefill_exchange = exchange


def plan_install(model, *, owner_rank, budget_bytes, overlap_prefill=False):
    """Validate geometry and memory without allocation or collective calls."""
    from vllm_ascend.models.glm5next.attention import Indexer

    validate_config(model.vllm_config, owner_rank)
    if type(overlap_prefill) is not bool:
        raise ValueError("indexer prefill overlap requires a boolean")
    if overlap_prefill and model.vllm_config.scheduler_config.max_num_seqs > MAX_CAPTURE_ROWS:
        raise ValueError("prefill-only overlap initially requires at most eight decode sequences")
    indexers = [module for module in model.modules() if isinstance(module, Indexer)]
    if not indexers:
        raise ValueError("indexer ownership found no stage-local GLM sparse indexers")
    buffers = [module.topk_indices_buffer for module in indexers]
    first = buffers[0]
    if any(buffer is not first for buffer in buffers):
        raise ValueError("stage indexers must use one shared MLA index buffer")
    if any(hasattr(module, "_glm_indexer_result_owner") for module in indexers):
        raise ValueError("indexer ownership cannot be hot-swapped or installed twice")
    # Leave arithmetic/selection modes unchanged. The owner must not create
    # an eager capture break absent from peer graphs. Cold model setup must
    # already have selected the existing <=8-row capture-safe implementation.
    if not model.vllm_config.model_config.enforce_eager and any(
        not getattr(module.indexer_op, "capture_safe_selection", False) for module in indexers
    ):
        raise ValueError("indexer owner graphs require an already prepared capture-safe selector")
    if (
        first.ndim != 2
        or first.dtype != torch.int32
        or not first.is_contiguous()
        or min(first.shape) < 1
        or any(getattr(module.indexer_op, "topk_indices_buffer", None) is not first for module in indexers)
    ):
        raise ValueError("indexer owner requires the shared contiguous INT32 MLA selection buffer")
    required = model.vllm_config.parallel_config.tensor_parallel_size * first.numel() * torch.int32.itemsize
    if type(budget_bytes) is not int or budget_bytes < required:
        raise ValueError("indexer result exchange exceeds its explicit per-rank byte reservation")
    if overlap_prefill:
        attention_backends(model, indexers)
    return indexers, required


def install(model, *, owner_rank, budget_bytes, overlap_prefill=False):
    """Compose only this loaded GLM model, once, before any request or capture."""
    from vllm.distributed import get_tp_group

    indexers, _ = plan_install(model, owner_rank=owner_rank, budget_bytes=budget_bytes, overlap_prefill=overlap_prefill)
    first = indexers[0].topk_indices_buffer
    tp = get_tp_group()
    if tp.world_size != model.vllm_config.parallel_config.tensor_parallel_size:
        raise ValueError("indexer owner TP process group differs from configured partition")
    exchange = IndexerResultExchange(
        group=tp.device_group,
        rank=tp.rank_in_group,
        size=tp.world_size,
        owner_rank=owner_rank,
        max_tokens=first.shape[0],
        width=first.shape[1],
        device=first.device,
        budget_bytes=budget_bytes,
        overlap_prefill=overlap_prefill,
    )
    for module in indexers:
        bind_indexer(module, exchange)
    if overlap_prefill:
        for backend in attention_backends(model, indexers):
            bind_preprocess(backend, exchange)
    exchange.warmup(synchronize=torch.npu.synchronize if first.device.type == "npu" else None)
    return dict(
        indexers=len(indexers),
        local_owner_rank=owner_rank,
        tp_rank=tp.rank_in_group,
        exchange_bytes=exchange.required_bytes,
        peer_cache_allocations_retained=True,
        hardware_qualified=False,
        overlap_prefill=overlap_prefill,
    )
