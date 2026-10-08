# SPDX-License-Identifier: Apache-2.0
"""Experimental completed-pool writes; decode retains the qualified path.

Build compact row metadata once per scheduler batch from existing CPU query
boundaries. Pool phase and speculative tail validity still come from device
positions. Never read a device scalar or compact a device boolean mask.
"""

import torch

MAX_GRAPH_TOKENS = 8
MAX_SPECULATIVE_TOKENS = 1
POOL_SIZE = 4


class PoolWritePlan:
    # Resident source is executed in a private, unregistered module and may
    # inherit postponed annotations. Avoid dataclass's module-name lookup.
    __slots__ = ("num_tokens", "complete", "tail")

    def __init__(self, num_tokens: int, complete: torch.Tensor, tail: torch.Tensor):
        self.num_tokens = num_tokens
        self.complete = complete  # [candidate pool base, request start, request end]
        self.tail = tail  # [input row, request index]


def plan_rows(boundaries: list[int], num_tokens: int) -> tuple[list[list[int]], list[list[int]]]:
    if (
        len(boundaries) < 2
        or boundaries[0] != 0
        or boundaries[-1] != num_tokens
        or any(a > b for a, b in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError("query boundaries must partition all actual input tokens")
    complete, tail = [], []
    for request, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        # At most ceil(length / 4) completed pools, independent of device phase.
        complete.extend([base, start, end] for base in range(start, end, POOL_SIZE))
        # MTP1 can require the preceding pool as well as the current tail.
        tail.extend([row, request] for row in range(max(start, end - POOL_SIZE - MAX_SPECULATIVE_TOKENS), end))
    return complete, tail


def make_plan(boundaries: torch.Tensor, num_tokens: int, device: torch.device) -> PoolWritePlan:
    if boundaries.device.type != "cpu" or boundaries.ndim != 1 or boundaries.dtype not in (torch.int32, torch.int64):
        raise ValueError("plan requires scheduler-owned CPU integer query boundaries")
    complete, tail = plan_rows(boundaries.tolist(), num_tokens)
    # One small CPU-to-device metadata copy per batch, shared by all layers.
    values = [value for row in complete for value in row] + [value for row in tail for value in row]
    packed = torch.tensor(values, dtype=torch.int64).to(device)
    count = len(complete) * 3
    return PoolWritePlan(num_tokens, packed[:count].view(-1, 3), packed[count:].view(-1, 2))


def write_compact_pools(
    keys,
    gates,
    ape,
    positions,
    index_meta,
    state_meta,
    state_cache,
    key_cache,
    plan,
    num_speculative_tokens,
    *,
    compress,
    storage_write,
    preserve_cache_dtype=False,
):
    if not 0 <= num_speculative_tokens <= MAX_SPECULATIVE_TOKENS:
        raise ValueError("completed-pool candidate is qualified only for no speculation or MTP1")
    if plan.num_tokens != keys.shape[0] or keys.shape != gates.shape or keys.shape[-1] != 128:
        raise ValueError("compact pool plan and key/gate geometry disagree")
    if state_cache.shape[1:] != (POOL_SIZE, 256) or key_cache.shape[-2:] != (1, 128):
        raise ValueError("unsupported kpool cache geometry")

    count = plan.complete.shape[0]
    if count:
        bases, starts, ends = plan.complete.unbind(1)
        first_positions = positions.index_select(0, starts)
        rows = bases + (POOL_SIZE - 1 - first_positions.remainder(POOL_SIZE))
        safe_rows = torch.minimum(rows, ends - 1)
        selected_positions = positions.index_select(0, safe_rows)
        state_slots = state_meta.slot_mapping.index_select(0, safe_rows).long().clamp_min(0)
        blocks = state_slots // POOL_SIZE
        # Gather every required previous-state row before any state write.
        old = state_cache[blocks]
        offsets = torch.arange(POOL_SIZE - 1, -1, -1, device=keys.device)
        local = safe_rows[:, None] - offsets[None, :]
        safe_local = local.clamp_min(0)
        same_pool = (local >= starts[:, None]) & (
            positions[safe_local] == selected_positions[:, None] - offsets[None, :]
        )
        pool_keys = torch.where(same_pool[:, :, None], keys[safe_local].float(), old[:, :, :128])
        pool_gates = torch.where(same_pool[:, :, None], gates[safe_local].float(), old[:, :, 128:])
        slots = index_meta.slot_mapping.index_select(0, safe_rows).long()
        completed = (rows < ends) & (slots >= 0) & ((selected_positions + 1).remainder(POOL_SIZE) == 0)

    if plan.tail.shape[0]:
        tail_rows, requests = plan.tail.unbind(1)
        slots_tail = state_meta.slot_mapping.index_select(0, tail_rows).long()
        safe_slots = slots_tail.clamp_min(0)
        final_positions = index_meta.raw_seq_lens.index_select(0, requests).long() - 1
        first_retained_pool = (final_positions - num_speculative_tokens).clamp_min(0) // POOL_SIZE
        valid = (slots_tail >= 0) & (positions.index_select(0, tail_rows) >= first_retained_pool * POOL_SIZE)
        offsets_tail = (
            state_cache.storage_offset()
            + (safe_slots // POOL_SIZE) * state_cache.stride(0)
            + safe_slots.remainder(POOL_SIZE) * state_cache.stride(1)
        )
        # Match the existing writer's gather dispatch: aclnnIndexSelect rejects
        # BF16 on 310P, while advanced indexing handles this storage format.
        values = torch.cat((keys[tail_rows].float(), gates[tail_rows].float()), dim=-1)
        storage_write(state_cache, offsets_tail, values, valid)

    if count:
        # Permanent native compressors can round directly into the cache's
        # dtype. Preserve their output contract when wrapping bound instances.
        compressed = (
            compress(pool_keys, pool_gates, ape, out_dtype=key_cache.dtype)
            if preserve_cache_dtype
            else compress(pool_keys, pool_gates, ape)
        )
        safe_slots = slots.clamp_min(0)
        offsets_key = (
            key_cache.storage_offset()
            + (safe_slots // key_cache.shape[1]) * key_cache.stride(0)
            + safe_slots.remainder(key_cache.shape[1]) * key_cache.stride(1)
        )
        storage_write(key_cache, offsets_key, compressed.to(key_cache.dtype), completed)


def wrap_builder(original):
    original = getattr(original, "__glm_resident_original__", original)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False, **kwargs):
        metadata = original(self, common_prefix_len, common_attn_metadata, fast_build=fast_build, **kwargs)
        # Some builders reuse metadata; never retain a plan from another batch.
        metadata._glm_completed_pool_plan = None
        if metadata.num_actual_tokens > MAX_GRAPH_TOKENS and metadata.compress_ratio == POOL_SIZE:
            boundaries = metadata.cum_query_lens_cpu
            if boundaries is not None and boundaries.device.type == "cpu":
                metadata._glm_completed_pool_plan = make_plan(
                    boundaries, metadata.num_actual_tokens, metadata.positions.device
                )
        return metadata

    build.__glm_resident_original__ = original
    return build


def wrap_writer(original, cache_tensor, storage_write, compress, *, preserve_cache_dtype=False):
    original = getattr(original, "__glm_resident_original__", original)

    def write(self, keys, gates, ape, positions, pool_size, indexer_metadata, state_metadata):
        plan = getattr(indexer_metadata, "_glm_completed_pool_plan", None)
        speculative = getattr(self.tail_cache, "num_speculative_tokens", 0)
        if (
            plan is None
            or keys.shape[0] <= MAX_GRAPH_TOKENS
            or pool_size != POOL_SIZE
            or not 0 <= speculative <= MAX_SPECULATIVE_TOKENS
        ):
            return original(self, keys, gates, ape, positions, pool_size, indexer_metadata, state_metadata)
        return write_compact_pools(
            keys,
            gates,
            ape,
            positions,
            indexer_metadata,
            state_metadata,
            cache_tensor(self.tail_cache),
            cache_tensor(self.k_cache),
            plan,
            speculative,
            compress=compress,
            storage_write=storage_write,
            preserve_cache_dtype=preserve_cache_dtype,
        )

    write.__glm_resident_original__ = original
    return write


def replacements(native_resources=None):
    from vllm_ascend.attention.indexer_kpool import AscendIndexerKPoolMetadataBuilder
    from vllm_ascend.models.glm5next import sparse_attn_indexer_kpool as module

    return {
        "vllm_ascend.attention.indexer_kpool:AscendIndexerKPoolMetadataBuilder.build": wrap_builder(
            AscendIndexerKPoolMetadataBuilder.build
        ),
        "vllm_ascend.models.glm5next.sparse_attn_indexer_kpool:SparseAttnIndexerKpool._write_pools": wrap_writer(
            module.SparseAttnIndexerKpool._write_pools,
            module._cache_tensor,
            module._masked_storage_write,
            module.compress_kpool,
        ),
    }
