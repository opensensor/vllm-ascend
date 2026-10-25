# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 eager arithmetic and resident signed-W4 expert banks.

This module deliberately imports neither vLLM nor torch_npu: the same routing,
activation and paged-attention arithmetic is exercised by CPU tests. Device
execution requires the existing grouped 310P Cube extension for projections.
"""

from collections.abc import Callable

import torch
from torch import nn

W4_BLOCK = 32
MAX_GROUPED_ROUTES = 128
NZ_OUTPUT_TILE = 16
NZ_INPUT_TILE = 256
# TP2/PP3 real-weight coherence and CPU/NPU comparisons qualify 64-query
# batches. Keep the same per-head top-k selection, masking and FP32 reductions.
ATTENTION_QUERY_TILE = 64
W4_LAYOUT = "minimax-m3-signed-w4-block32-fp16-v1"


def validate_runtime(config):
    """Reject modes whose collective/cache contracts this eager path cannot meet."""
    model, parallel = config.model_config, config.parallel_config
    text = model.hf_text_config
    if getattr(text, "ascend_minimax_m3_w4", None) != W4_LAYOUT:
        raise ValueError("310P M3 requires a completed minimax_m3_w4 export")
    if config.load_config.load_format not in ("minimax_m3_w4", "dummy"):
        raise ValueError("Use --load-format minimax_m3_w4 for the packed checkpoint")
    if model.dtype != torch.float16 or not model.enforce_eager:
        raise ValueError("310P M3 currently requires --dtype half --enforce-eager")
    if config.quant_config is not None or config.speculative_config is not None or config.lora_config is not None:
        raise ValueError("310P M3 W4 does not support extra quantization, MTP, or LoRA")
    if parallel.data_parallel_size != 1 or getattr(parallel, "decode_context_parallel_size", 1) != 1:
        raise ValueError("310P M3 requires DP=1 and DCP=1")
    if getattr(parallel, "prefill_context_parallel_size", 1) != 1:
        raise ValueError("310P M3 requires PCP=1")
    if parallel.enable_eplb or getattr(parallel, "use_sequence_parallel_moe", False):
        raise ValueError("310P M3 does not support EPLB or sequence-parallel MoE")
    tp = parallel.tensor_parallel_size
    if tp not in (1, 2, 4) or text.num_local_experts % tp:
        raise ValueError("310P M3 supports TP=1, 2, or 4 with evenly partitioned experts")
    if tp > 1 and not parallel.enable_expert_parallel:
        raise ValueError("310P M3 requires stage-local --enable-expert-parallel")
    if text.scoring_func != "sigmoid" or text.hidden_act != "swigluoai":
        raise ValueError("Unsupported M3 routing/activation contract")
    sparse = text.sparse_attention_config
    if sparse["sparse_block_size"] != 128 or sparse["sparse_num_index_heads"] != text.num_key_value_heads:
        raise ValueError("310P M3 requires 128-token MSA blocks and one index head per KV head")
    if config.cache_config.block_size != 128 or config.cache_config.cache_dtype not in ("auto", "float16"):
        raise ValueError("310P M3 requires --block-size 128 and FP16 KV cache")


def clipped_swiglu(gate, up, alpha=1.702, beta=1.0, limit=7.0):
    """M3 clips only the positive gate tail; compute the product in FP32."""
    gate = gate.float().clamp(max=limit)
    up = up.float().clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(alpha * gate) * (up + beta)


def route_experts(logits, correction_bias, top_k):
    scores = logits.float().sigmoid()
    selection = scores if correction_bias is None else scores + correction_bias.float()
    ids = selection.topk(top_k, dim=-1).indices
    weights = scores.gather(-1, ids)
    # Correction bias affects selection, never combination weights.
    return weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny), ids


def unpack_w4(codes, scales):
    """Exact CPU oracle for the stored FP16-scale contract (not a device fallback)."""
    if codes.device.type != "cpu":
        raise ValueError("The unpacked expert oracle is CPU-only")
    nibbles = torch.stack((codes & 15, codes >> 4), -1).flatten(-2).to(torch.int8)
    signed = torch.where(nibbles >= 8, nibbles - 16, nibbles)
    return signed.float() * scales.float().repeat_interleave(W4_BLOCK, -2).repeat_interleave(W4_BLOCK, -1)


def pack_w4_nz256(codes):
    """Lossless CPU load-time packing for the current 16x256 grouped kernel.

    INT8 dtype signals NZ layout to the operator; the checkpoint stays UINT8
    canonical. Older 16x128 operator packages are incompatible with this layout.
    """
    if codes.device.type != "cpu" or codes.dtype != torch.uint8 or codes.ndim != 2:
        raise ValueError("NZ packing requires a CPU UINT8 canonical matrix")
    rows, packed_cols = codes.shape
    cols = packed_cols * 2
    if not rows or rows % NZ_OUTPUT_TILE or not cols or cols % NZ_INPUT_TILE:
        raise ValueError("Expert dimensions must contain complete 16x256 NZ tiles")
    row_tiles, col_tiles = rows // NZ_OUTPUT_TILE, cols // NZ_INPUT_TILE
    values = codes.reshape(row_tiles, NZ_OUTPUT_TILE, col_tiles, NZ_INPUT_TILE // 2)
    fields = torch.stack((values & 15, values >> 4), dim=-1)
    fields = fields.reshape(row_tiles, NZ_OUTPUT_TILE, col_tiles, NZ_INPUT_TILE)
    planes = fields.permute(0, 2, 3, 1).reshape(row_tiles, col_tiles, 2, -1)
    return (planes[..., 0, :] | (planes[..., 1, :] << 4)).reshape_as(codes).view(torch.int8).contiguous()


def reference_grouped_mm(x, codes, scales, group_ends):
    """CPU implementation of the native grouped operator, including peer zeros."""
    if x.device.type != "cpu":
        raise ValueError("Reference grouped projection is CPU-only")
    output = torch.zeros((x.shape[0], codes.shape[1]), dtype=x.dtype, device=x.device)
    start = 0
    for expert, stop in enumerate(group_ends.tolist()):
        if stop > start:
            output[start:stop] = (
                x[start:stop].float() @ unpack_w4(codes[expert], scales[expert]).to(torch.float16).float().T
            ).to(x.dtype)
        start = stop
    return output


def grouped_w4_mm(x, codes, scales, group_ends):
    """Stable resident dispatch seam for versioned, validated CANN kernels."""
    op = getattr(torch.ops._C_ascend, "npu_w2_grouped_blocked_dequant_matmul_310", None)
    if op is None:
        raise RuntimeError("M3 W4 requires the grouped 310P Cube extension and OPP")
    return op(x, codes, scales, group_ends)


class PackedM3Experts(nn.Module):
    """Full-width experts partitioned over a stage-local EP group."""

    def __init__(
        self,
        hidden,
        intermediate,
        num_experts,
        ep_size,
        ep_rank,
        top_k,
        *,
        prefix,
        alpha=1.702,
        beta=1.0,
        limit=7.0,
        routed_scale=2.0,
        projector: Callable | None = None,
    ):
        super().__init__()
        if num_experts % ep_size or not 0 <= ep_rank < ep_size:
            raise ValueError("M3 W4 requires evenly partitioned, stage-local experts")
        if hidden % W4_BLOCK or intermediate % W4_BLOCK or not 0 < top_k <= num_experts:
            raise ValueError("Invalid M3 W4 dimensions or top-k")
        self.prefix = prefix
        self.num_local_experts = num_experts // ep_size
        self.local_expert_offset = ep_rank * self.num_local_experts
        self.top_k = top_k
        self.alpha, self.beta, self.limit, self.routed_scale = alpha, beta, limit, routed_scale
        self.projector = projector
        self.max_grouped_routes = MAX_GROUPED_ROUTES
        for projection, rows, cols in (
            ("w1", intermediate, hidden),
            ("w3", intermediate, hidden),
            ("w2", hidden, intermediate),
        ):
            self.register_buffer(
                f"{projection}_codes", torch.empty(self.num_local_experts, rows, cols // 2, dtype=torch.uint8)
            )
            self.register_buffer(
                f"{projection}_scales",
                torch.empty(self.num_local_experts, rows // W4_BLOCK, cols // W4_BLOCK, dtype=torch.float16),
            )

    def forward(self, hidden_states, router_logits, correction_bias=None):
        if hidden_states.numel() == 0:
            return torch.zeros_like(hidden_states)
        op = self.projector
        if op is None:
            if hidden_states.device.type == "cpu":
                if self.w1_codes.dtype != torch.uint8:
                    raise ValueError("NZ expert banks require the matching native NPU operator")
                op = reference_grouped_mm
            else:
                op = grouped_w4_mm
        chunk_tokens = self.max_grouped_routes // self.top_k
        if chunk_tokens == 0:
            raise ValueError("M3 top-k exceeds grouped operator route capacity")
        # INT8 kernels own immutable, precomputed scale metadata. Passing the
        # original FP16 scales avoids recasting every full expert bank per token.
        int8_layouts = getattr(self, "_m3_int8_layouts", ())
        scales = {
            name: getattr(self, f"{name}_scales") if name in int8_layouts else getattr(self, f"{name}_scales").float()
            for name in ("w1", "w2", "w3")
        }
        outputs = []
        for start in range(0, hidden_states.shape[0], chunk_tokens):
            x = hidden_states[start : start + chunk_tokens]
            weights, ids = route_experts(router_logits[start : start + chunk_tokens], correction_bias, self.top_k)
            local_ids = ids.flatten() - self.local_expert_offset
            local_ids = torch.where(
                (local_ids >= 0) & (local_ids < self.num_local_experts), local_ids, self.num_local_experts
            )
            compact = x.shape[0] == 1 and self.top_k <= getattr(op, "max_compact_routes", 0)
            if compact:
                # One token has at most four routes. Pass their local IDs in
                # routing order, avoiding sorting and a full expert histogram.
                sorted_x = x.expand(self.top_k, -1).to(torch.float16).contiguous()
                ends = local_ids.to(torch.int32).contiguous()
                routed_weights = weights.flatten()[:, None]
            else:
                order = local_ids.to(torch.int32).float().argsort(stable=True)
                inverse = order.to(torch.int32).float().argsort()
                token_ids = torch.arange(x.shape[0], device=x.device).repeat_interleave(self.top_k)
                rows = token_ids.index_select(0, order)
                sorted_x = x.index_select(0, rows).to(torch.float16).contiguous()
                expert_ids = torch.arange(self.num_local_experts, device=x.device)
                # 310P INT64 reductions use slow AI-CPU casts/cumsum. Route counts
                # are bounded; retain INT64 only at the native kernel boundary.
                count_dtype = torch.int32 if local_ids.numel() <= torch.iinfo(torch.int32).max else torch.int64
                ends = (
                    (local_ids[:, None] == expert_ids)
                    .sum(0, dtype=count_dtype)
                    .cumsum(0, dtype=count_dtype)
                    .to(torch.int64)
                    .contiguous()
                )
                routed_weights = weights.flatten()[:, None].index_select(0, order)
            gate = op(sorted_x, self.w1_codes, scales["w1"], ends)
            up = op(sorted_x, self.w3_codes, scales["w3"], ends)
            act = clipped_swiglu(gate, up, self.alpha, self.beta, self.limit).to(torch.float16)
            routed = op(act.contiguous(), self.w2_codes, scales["w2"], ends).float()
            # Never depend on uncomputed peer rows being initialized by a kernel.
            local_rows = (
                ends < self.num_local_experts if compact else torch.arange(routed.shape[0], device=x.device) < ends[-1]
            )
            routed = torch.where(local_rows[:, None], routed, 0)
            routed *= routed_weights
            if not compact:
                routed = routed.index_select(0, inverse)
            outputs.append(routed.view(x.shape[0], self.top_k, -1).sum(1))
        return torch.cat(outputs) * self.routed_scale


def paged_sparse_attention(
    query,
    key,
    value,
    index_query,
    index_key,
    key_cache,
    value_cache,
    index_cache,
    *,
    query_starts,
    seq_lens,
    block_table,
    index_block_table,
    slot_mapping,
    index_slot_mapping,
    max_seq_len,
    topk_blocks,
    scale,
    init_blocks=0,
    local_blocks=1,
):
    """Causal MSA with bounded query tiles and ordinary ND paged caches.

    Each index head selects blocks for its corresponding KV head. Host query
    boundaries come from scheduler metadata; no
    device lengths, expert ids, or scores are read back in this path.
    """
    block_size = key_cache.shape[1]
    if (
        block_size != index_cache.shape[1]
        or query.shape[1] % key_cache.shape[2]
        or index_query.shape[1] != key_cache.shape[2]
    ):
        raise ValueError("Incompatible M3 page sizes or GQA heads")
    actual = query_starts[-1]
    if actual > query.shape[0] or topk_blocks <= 0:
        raise ValueError("Invalid M3 sparse attention metadata")
    # Scheduler padding has no cache entries. Valid request tokens must have
    # real slots; slicing by CPU boundaries avoids device nonzero()/sync.
    for cache, update, slots in (
        (key_cache, key, slot_mapping),
        (value_cache, value, slot_mapping),
        (index_cache, index_key, index_slot_mapping),
    ):
        cache.flatten(0, 1).index_copy_(0, slots[:actual].long(), update[:actual].to(cache.dtype))
    output = torch.zeros_like(query)
    if actual == 0:
        return output
    num_blocks = (max_seq_len + block_size - 1) // block_size
    if min(block_table.shape[1], index_block_table.shape[1]) < num_blocks:
        raise ValueError("Block table does not cover max_seq_len")
    positions = torch.arange(num_blocks * block_size, device=query.device)
    block_ids = torch.arange(num_blocks, device=query.device)
    offsets = torch.arange(block_size, device=query.device)
    repeat = query.shape[1] // key_cache.shape[2]
    for request, (start, stop) in enumerate(zip(query_starts[:-1], query_starts[1:])):
        if start == stop:
            continue
        seq_len = seq_lens[request]
        valid_blocks = block_ids * block_size < seq_len
        pages = torch.where(valid_blocks, block_table[request, :num_blocks], 0).long()
        index_pages = torch.where(valid_blocks, index_block_table[request, :num_blocks], 0).long()
        ik = index_cache.index_select(0, index_pages).flatten(0, 1).float()
        # Uninitialized padding may contain NaNs: mask it before dot products.
        ik = torch.where((positions < seq_len)[:, None], ik, 0)
        for begin in range(start, stop, ATTENTION_QUERY_TILE):
            end = min(stop, begin + ATTENTION_QUERY_TILE)
            qpos = seq_len - (stop - start) + torch.arange(begin - start, end - start, device=query.device)
            scores = torch.matmul(index_query[begin:end].float(), ik.T)
            visible = positions[None, :] <= qpos[:, None]
            scores = scores.masked_fill(~visible[:, None, :], -torch.inf)
            scores = scores.view(end - begin, index_query.shape[1], num_blocks, block_size).amax(-1)
            query_blocks = qpos[:, None] // block_size
            causal_blocks = block_ids[None, :] <= query_blocks
            forced_local = block_ids[None, :] > query_blocks - local_blocks
            for head in range(index_query.shape[1]):
                head_scores = torch.where(forced_local & causal_blocks, 1e29, scores[:, head])
                head_scores = torch.where((block_ids[None, :] < init_blocks) & causal_blocks, 1e30, head_scores)
                head_scores = head_scores.masked_fill(~causal_blocks, -torch.inf)
                selected = head_scores.topk(min(topk_blocks, num_blocks), -1).indices
                logical = (selected[..., None] * block_size + offsets).flatten(1)
                selected_pages = pages[selected]
                # Index ND cache rows directly: slicing a head before advanced
                # indexing materializes the entire allocated cache on 310P.
                physical = (selected_pages[..., None] * block_size + offsets).flatten(1)
                head_rows = (physical * key_cache.shape[2] + head).flatten()
                gathered_shape = (end - begin, -1, query.shape[-1])
                k = key_cache.flatten(0, 2).index_select(0, head_rows).view(gathered_shape).float()
                v = value_cache.flatten(0, 2).index_select(0, head_rows).view(gathered_shape).float()
                mask = (logical <= qpos[:, None]) & (logical < seq_len)
                k = torch.where(mask[..., None], k, 0)
                v = torch.where(mask[..., None], v, 0)
                head_start, head_stop = head * repeat, (head + 1) * repeat
                q = query[begin:end, head_start:head_stop].float()
                logits = torch.einsum("qhd,qkd->qhk", q, k) * scale
                probs = logits.masked_fill(~mask[:, None, :], -torch.inf).softmax(-1)
                output[begin:end, head_start:head_stop] = torch.einsum("qhk,qkd->qhd", probs, v).to(query.dtype)
    return output
