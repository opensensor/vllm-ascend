# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged GLM-5.3-Flash kpool sparse-attention indexer for Ascend."""

import torch
import torch_npu
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp

from vllm_ascend.models.glm5next.kpool_ops import (
    compress_kpool,
    dense_kpool_token_indices,
    expand_kpool_groups,
    score_and_select_kpool_tokens,
    score_kpool,
    select_kpool_groups,
)
from vllm_ascend.models.glm5next.ops.kpool_native import score_kpool_paged, supports_live_kpool_score

MAX_MTP_GRAPH_TOKENS = 8


def _cache_tensor(cache_layer) -> torch.Tensor:
    cache = cache_layer.kv_cache
    if isinstance(cache, (tuple, list)):
        if len(cache) != 1:
            raise RuntimeError("GLM kpool cache must contain exactly one tensor")
        cache = cache[0]
    if not isinstance(cache, torch.Tensor) or cache.numel() == 0:
        raise RuntimeError("GLM kpool cache has not been bound to the model")
    return cache


def _masked_storage_write(
    cache: torch.Tensor,
    row_offsets: torch.Tensor,
    values: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    """Write fixed-width rows with a capture-stable scatter.

    The 310P generic scatter supports FP32 and FP16, but not BF16. Reinterpret
    two-byte cache storage as FP16 so BF16 values are copied bit-for-bit. A
    negative element index skips an invalid row without creating a
    data-dependent index tensor. Valid physical rows must be unique within a
    scheduler step.
    """
    if cache.element_size() == torch.float32.itemsize:
        storage_dtype = torch.float32
    elif cache.element_size() == torch.float16.itemsize:
        storage_dtype = torch.float16
    else:
        raise TypeError(f"GLM kpool cache dtype {cache.dtype} has an unsupported element size")
    storage_elements = cache.untyped_storage().nbytes() // cache.element_size()
    flat_cache = cache.as_strided((storage_elements,), (1,), storage_offset=0).view(storage_dtype)
    dimensions = torch.arange(values.shape[1], device=cache.device)
    offsets = row_offsets[:, None] + dimensions[None, :] * cache.stride(-1)
    invalid_offsets = torch.full_like(offsets, -1)
    flat_offsets = torch.where(valid[:, None], offsets, invalid_offsets).reshape(-1, 1)
    storage_values = values.contiguous().view(storage_dtype).reshape(-1)
    if cache.device.type == "cpu":
        # CPU-only unit tests do not have the torch_npu scatter implementation.
        valid_elements = flat_offsets[:, 0] >= 0
        flat_cache[flat_offsets[valid_elements, 0]] = storage_values[valid_elements]
        return
    torch_npu.npu_scatter_nd_update_(flat_cache, flat_offsets, storage_values)


@CustomOp.register("sparse_attn_indexer_kpool")
class SparseAttnIndexerKpool(CustomOp):
    """Compress complete pools, persist tails, and select causal pool IDs."""

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str | None,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        tail_cache=None,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.tail_cache = tail_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache

    def _write_pools(
        self,
        keys: torch.Tensor,
        gates: torch.Tensor,
        ape: torch.Tensor,
        positions: torch.Tensor,
        pool_size: int,
        indexer_metadata,
        state_metadata,
    ) -> None:
        """Write complete pools, including one continued from an earlier step."""
        num_tokens = keys.shape[0]
        device = keys.device
        state_cache = _cache_tensor(self.tail_cache)
        key_cache = _cache_tensor(self.k_cache)
        state_block_size = state_cache.shape[1] if state_cache.ndim == 3 else 0
        if (
            state_cache.ndim != 3
            or state_block_size < pool_size
            or state_block_size % pool_size
            or state_cache.shape[2] != 2 * self.head_dim
            or getattr(state_metadata, "block_size", state_block_size) != state_block_size
        ):
            raise RuntimeError("GLM kpool state cache has unexpected page geometry")
        if key_cache.shape[-2:] != (1, self.head_dim):
            raise RuntimeError("GLM kpool key cache has unexpected page geometry")

        state_slots = state_metadata.slot_mapping[:num_tokens].long()
        safe_state_slots = state_slots.clamp_min(0)
        state_blocks = torch.div(safe_state_slots, state_block_size, rounding_mode="floor")
        state_offsets = safe_state_slots - state_blocks * state_block_size
        # Gather before scattering this step: a previous step may have left
        # the first three members of a pool in the sliding state page.
        if state_block_size == pool_size:
            old_state = state_cache[state_blocks]
        else:
            pool_starts = torch.div(state_offsets, pool_size, rounding_mode="floor") * pool_size
            pool_rows = pool_starts[:, None] + torch.arange(pool_size, device=device)[None, :]
            old_state = state_cache[state_blocks[:, None], pool_rows]
        offsets = torch.arange(pool_size - 1, -1, -1, device=device)
        local_indices = torch.arange(num_tokens, device=device)[:, None] - offsets[None, :]
        safe_local_indices = local_indices.clamp_min(0)
        boundaries = indexer_metadata.cum_query_lens
        if boundaries is None:
            raise RuntimeError("GLM kpool needs device query boundaries from the scheduler")
        request_ids = torch.searchsorted(
            boundaries,
            torch.arange(num_tokens, device=device, dtype=boundaries.dtype),
            right=True,
        )
        same_pool = (
            (local_indices >= 0)
            & (positions[safe_local_indices] == positions[:, None] - offsets[None, :])
            & (request_ids[safe_local_indices] == request_ids[:, None])
        )
        local_keys = keys[safe_local_indices].float()
        local_gates = gates[safe_local_indices].float()
        old_keys = old_state[:, :, : self.head_dim]
        old_gates = old_state[:, :, self.head_dim :]
        pool_keys = torch.where(same_pool[:, :, None], local_keys, old_keys)
        pool_gates = torch.where(same_pool[:, :, None], local_gates, old_gates)

        # Sliding state pages can recycle the same physical slots within a
        # large prefill. Under speculation retain the pool at the earliest
        # possible rejection as well as all later pools in the window.
        final_positions = indexer_metadata.raw_seq_lens[request_ids].long() - 1
        earliest_positions = final_positions - getattr(self.tail_cache, "num_speculative_tokens", 0)
        final_pool_starts = torch.div(earliest_positions.clamp_min(0), pool_size, rounding_mode="floor") * pool_size
        valid_state = (state_slots >= 0) & (positions >= final_pool_starts)
        state_row_offsets = (
            state_cache.storage_offset() + state_blocks * state_cache.stride(0) + state_offsets * state_cache.stride(1)
        )
        _masked_storage_write(
            state_cache,
            state_row_offsets,
            torch.cat((keys.float(), gates.float()), dim=-1),
            valid_state,
        )

        completed = ((positions + 1) % pool_size == 0) & (indexer_metadata.slot_mapping[:num_tokens] >= 0)
        # Compress fixed-shape rows; only completed pools write cache values.
        # This trades extra compression work for capture-stable tensor shapes.
        compressed = compress_kpool(pool_keys, pool_gates, ape)
        pool_slots = indexer_metadata.slot_mapping[:num_tokens].long().clamp_min(0)
        block_size = key_cache.shape[1]
        # The shared GLM cache is a page-strided view of an int8 backing.
        # 310P has no IndexPutV2 binary for its BF16 advanced-indexed view;
        # write the same physical elements through a flat storage view.
        row_offsets = (
            key_cache.storage_offset()
            + torch.div(pool_slots, block_size, rounding_mode="floor") * key_cache.stride(0)
            + (pool_slots % block_size) * key_cache.stride(1)
        )
        _masked_storage_write(key_cache, row_offsets, compressed.to(key_cache.dtype), completed)

    def forward_oot(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if index_kpool <= 1 or gate_score is None or compress_ape is None or positions is None:
            raise ValueError("GLM kpool requires gate scores, APE, positions, and pool_size > 1")
        if self.skip_k_cache_insert or self.tail_cache is None:
            raise NotImplementedError("GLM kpool requires its paged key and compressor state caches")
        if isinstance(q_quant, tuple):
            raise ValueError("GLM kpool expects unquantized BF16 query vectors")
        forward_metadata = get_forward_context().attn_metadata
        if not isinstance(forward_metadata, dict):
            # The profiling run has no cache pages or valid request positions.
            return self.topk_indices_buffer
        indexer_metadata = forward_metadata[self.k_cache.prefix]
        state_metadata = forward_metadata[self.tail_cache.prefix]
        num_tokens = indexer_metadata.num_actual_tokens
        if num_tokens > min(k.shape[0], q_quant.shape[0], positions.shape[0]):
            raise RuntimeError("GLM kpool metadata exceeds the actual token tensors")
        positions = positions[:num_tokens]
        self._write_pools(
            k[:num_tokens],
            gate_score[:num_tokens],
            compress_ape,
            positions,
            index_kpool,
            indexer_metadata,
            state_metadata,
        )

        if getattr(self, "capture_safe_selection", False) and num_tokens <= MAX_MTP_GRAPH_TOKENS:
            return self._select_tokens_fixed(
                queries=q_quant[:num_tokens],
                weights=weights[:num_tokens],
                positions=positions,
                pool_size=index_kpool,
                indexer_metadata=indexer_metadata,
            )

        capture = BreakableCUDAGraphCapture.current()
        if capture is not None and capture._capturing:
            # QSA's negative tail-count sentinel encodes a dense prefix and
            # ignores group IDs while every pool fits. Clear the shared buffer
            # in the graph so its otherwise-unused values are deterministic;
            # the eager break only fills real indices for long contexts.
            self.topk_indices_buffer[:num_tokens].fill_(-1)
            from vllm_ascend.utils import weak_ref_tensor

            weak_query = weak_ref_tensor(q_quant)
            weak_weights = weak_ref_tensor(weights)
            weak_positions = weak_ref_tensor(positions)

            def select_current_tokens() -> None:
                current_metadata = get_forward_context().attn_metadata
                if not isinstance(current_metadata, dict):
                    raise RuntimeError("GLM kpool graph replay requires attention metadata")
                indexer_metadata = current_metadata[self.k_cache.prefix]
                pool_lengths_cpu = indexer_metadata.seq_lens_cpu
                if pool_lengths_cpu is not None:
                    pool_lengths = pool_lengths_cpu.tolist()
                    cache = _cache_tensor(self.k_cache)
                    if pool_lengths and max(pool_lengths) > indexer_metadata.block_table.shape[1] * cache.shape[1]:
                        raise RuntimeError("GLM kpool block table is shorter than the request's pooled keys")
                    # Pooled host lengths are floor(raw_length / pool_size).
                    # At the budget boundary a raw tail can exceed QSA's
                    # dense-token limit, so fall back to the exact selector.
                    if not pool_lengths or max(pool_lengths) < self.topk_tokens // index_kpool:
                        return
                self._select_tokens(
                    weak_query,
                    weak_weights,
                    weak_positions,
                    index_kpool,
                    indexer_metadata,
                )

            capture.add_eager(select_current_tokens)
            return self.topk_indices_buffer

        return self._select_tokens(q_quant, weights, positions, index_kpool, indexer_metadata)

    def _select_tokens_fixed(self, queries, weights, positions, pool_size, indexer_metadata):
        """Select decode rows with device-only lengths and fixed cache bounds.

        A captured row can cross the dense/sparse threshold or move to another
        request on replay. Read its current request and pages on device. Mask
        unused pages before scoring, including recycled pages containing NaNs.
        Prefill keeps the host-bounded batched path.
        """
        cache = _cache_tensor(self.k_cache)
        block_size = cache.shape[1]
        table = indexer_metadata.block_table
        num_pools = min((self.max_model_len + pool_size - 1) // pool_size, table.shape[1] * block_size)
        if getattr(self, "live_kpool_score", False) and supports_live_kpool_score(queries, cache, pool_size, num_pools):
            logits = score_kpool_paged(
                queries, weights, cache, table, indexer_metadata.cum_query_lens, positions, num_pools
            )
            self.topk_indices_buffer[: queries.shape[0]].fill_(-1)
            # Preserve the existing per-row topk shapes and tie behavior.
            for row in range(queries.shape[0]):
                selected, _, tail_starts, tail_counts = select_kpool_groups(
                    logits[row : row + 1],
                    positions[row : row + 1],
                    self.topk_tokens,
                    pool_size,
                    scores_are_causal=True,
                )
                expanded = expand_kpool_groups(selected, tail_starts, tail_counts, pool_size)
                self.topk_indices_buffer[row : row + 1, : expanded.shape[1]].copy_(expanded)
            return self.topk_indices_buffer
        pool_ids = torch.arange(num_pools, device=cache.device, dtype=torch.long)
        row_ids = torch.arange(queries.shape[0], device=cache.device, dtype=indexer_metadata.cum_query_lens.dtype)
        requests = torch.searchsorted(indexer_metadata.cum_query_lens, row_ids, right=True)
        requests = requests.clamp(max=table.shape[0] - 1).long()
        self.topk_indices_buffer[: queries.shape[0]].fill_(-1)
        for row in range(queries.shape[0]):
            pages = table.index_select(0, requests[row : row + 1])[0]
            complete = (positions[row] + 1) // pool_size
            valid = pool_ids < complete
            physical_pages = pages[pool_ids // block_size].long()
            physical_pages = torch.where(valid, physical_pages, 0)
            keys = cache[physical_pages, pool_ids % block_size, 0]
            keys = torch.where(valid[:, None], keys, 0)
            logits = score_kpool(queries[row : row + 1], weights[row : row + 1], keys)
            selected, _, tail_starts, tail_counts = select_kpool_groups(
                logits, positions[row : row + 1], self.topk_tokens, pool_size
            )
            expanded = expand_kpool_groups(selected, tail_starts, tail_counts, pool_size)
            self.topk_indices_buffer[row : row + 1, : expanded.shape[1]].copy_(expanded)
        return self.topk_indices_buffer

    def _select_tokens(
        self,
        queries: torch.Tensor,
        weights: torch.Tensor,
        positions: torch.Tensor,
        pool_size: int,
        indexer_metadata,
    ) -> torch.Tensor:
        """Select current pooled keys using scheduler-owned host bounds."""
        num_tokens = indexer_metadata.num_actual_tokens
        if num_tokens > min(queries.shape[0], weights.shape[0], positions.shape[0]):
            raise RuntimeError("GLM kpool metadata exceeds the selection tensors")
        queries = queries[:num_tokens]
        weights = weights[:num_tokens]
        positions = positions[:num_tokens]

        cache = _cache_tensor(self.k_cache)
        block_size = cache.shape[1]
        boundaries = indexer_metadata.cum_query_lens_cpu
        if boundaries is None:
            raise RuntimeError("GLM kpool needs host query boundaries from the scheduler")
        pool_budget = self.topk_tokens // pool_size
        pool_lengths = indexer_metadata.seq_lens_cpu.tolist()
        if pool_lengths and max(pool_lengths) > indexer_metadata.block_table.shape[1] * block_size:
            raise RuntimeError("GLM kpool block table is shorter than the request's pooled keys")
        if pool_lengths and max(pool_lengths) <= pool_budget:
            # All requests need every causal token. The existing selection
            # would emit these same indices after scoring zero logits; build
            # every row at once and avoid per-request device launches. Keep
            # The host lengths include this step, including a pool completed
            # by its final token.
            dense = dense_kpool_token_indices(positions, self.topk_tokens, pool_size)
            if self.topk_indices_buffer.shape[1] > dense.shape[1]:
                self.topk_indices_buffer[:num_tokens].fill_(-1)
            self.topk_indices_buffer[:num_tokens, : dense.shape[1]].copy_(dense)
            return self.topk_indices_buffer

        self.topk_indices_buffer[:num_tokens].fill_(-1)
        query_ends = boundaries.tolist()
        for request, (start, end) in enumerate(zip(query_ends[:-1], query_ends[1:])):
            if start == end:
                continue
            num_pools = int(pool_lengths[request])
            pool_ids = torch.arange(num_pools, device=cache.device, dtype=torch.long)
            page_ids = indexer_metadata.block_table[request, pool_ids // block_size].long()
            keys = cache[page_ids, pool_ids % block_size, 0]
            if num_pools <= pool_budget:
                logits = torch.zeros(end - start, num_pools, device=cache.device)
                selected, _, tail_starts, tail_counts = select_kpool_groups(
                    logits, positions[start:end], self.topk_tokens, pool_size
                )
                expanded = expand_kpool_groups(selected, tail_starts, tail_counts, pool_size)
            else:
                expanded = score_and_select_kpool_tokens(
                    queries[start:end],
                    weights[start:end],
                    keys,
                    positions[start:end],
                    self.topk_tokens,
                    pool_size,
                )
            self.topk_indices_buffer[start:end, : expanded.shape[1]] = expanded
        return self.topk_indices_buffer

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_oot(
            hidden_states,
            q_quant,
            k,
            weights,
            gate_score=gate_score,
            compress_ape=compress_ape,
            index_kpool=index_kpool,
            positions=positions,
        )
