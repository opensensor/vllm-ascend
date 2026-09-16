# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""MRV2 model state for Ascend 310P (dense/VL + hybrid/GDN)."""

from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend._310p.ops.rotary_embedding import prepare_mrope_cos_sin_slices_from_runner
from vllm_ascend._310p.worker.v2.rope import Ascend310PRopeState, get_310p_rope_state
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.models.qwen4_exp.dtype_policy import (
    ASCEND_QWEN4EXP_DTYPE_POLICY,
    Qwen4ExpDtypePolicy,
)
from vllm_ascend.models.qwen4_exp.qwen4exp_gdn import (
    GDNBackend,
    GDNConvStateLayout,
    Qwen4ExpGDNParams,
    Qwen4ExpGDNStateLayout,
    Qwen4ExpGDNStatePool,
    gdn_delta_rule,
)
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.default import AscendModelState
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState

from .sampler import Ascend310PSampler


class _Ascend310PModelStateMixin:
    """310P RoPE / FULL-graph seq_lens helpers shared by dense and hybrid.

    Attribute annotations below are provided at runtime by AscendModelState /
    DefaultModelState (or set in the concrete subclass ``__init__``). Declared
    here so mypy can type-check the mixin in isolation.
    """

    model_config: Any
    model: nn.Module
    max_num_reqs: int
    max_num_tokens: int
    max_model_len: int
    device: torch.device
    rope_state: Any
    mm_pruner: Any
    _capture_seq_lens_by_ptr: dict[int, torch.Tensor]

    def _replace_310p_rope_state(self, encoder_cache: EncoderCache | None) -> None:
        del encoder_cache  # EVS / mm_pruner unsupported on 310P (see review notes).
        self.rope_state = get_310p_rope_state(
            self.model_config,
            self.model,
            self.max_num_reqs,
            self.max_num_tokens,
            self.max_model_len,
            self.device,
        )
        # Clear any parent-created EVS pruner: 310P MRv1 has no EVS path, and
        # Ascend310PRopeState lacks the read/update_prefill_positions EVS needs.
        self.mm_pruner = None

    def _record_capture_seq_lens(self, seq_lens: torch.Tensor) -> None:
        """Record the largest captured view for each physical buffer."""
        data_ptr = seq_lens.data_ptr()
        recorded = self._capture_seq_lens_by_ptr.get(data_ptr)
        if recorded is None or seq_lens.numel() > recorded.numel():
            self._capture_seq_lens_by_ptr[data_ptr] = seq_lens

    def _refresh_capture_seq_lens(self, runtime_seq_lens: torch.Tensor) -> None:
        """Copy runtime lengths to buffers read by FULL graph replay."""
        for capture_seq_lens in self._capture_seq_lens_by_ptr.values():
            num_seq_lens = min(capture_seq_lens.numel(), runtime_seq_lens.numel())
            capture_seq_lens[:num_seq_lens].copy_(runtime_seq_lens[:num_seq_lens], non_blocking=True)
            if num_seq_lens < capture_seq_lens.numel():
                capture_seq_lens[num_seq_lens:].zero_()

    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        if for_capture:
            self._record_capture_seq_lens(input_batch.seq_lens)
        elif cudagraph_mode == CUDAGraphMode.FULL:
            # Updating only input_batch.seq_lens is insufficient when replay is
            # bound to a different capture-time address.
            self._refresh_capture_seq_lens(input_batch.seq_lens)

        # Mixin sits before AscendModelState / AscendMambaHybridModelState in MRO.
        return super().prepare_attn(  # type: ignore[misc]
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
        )

    def prepare_inputs(self, input_batch: AscendInputBatch, req_states):
        if self.rope_state is None:
            return super().prepare_inputs(input_batch, req_states)  # type: ignore[misc]

        assert isinstance(self.rope_state, Ascend310PRopeState)
        # Upstream RopeState.prepare_positions uses Triton; 310P builds positions
        # on CPU from staged prefill tables, then H2D copies.
        self.rope_state.prepare_positions_cpu(
            input_batch.idx_mapping_np,
            input_batch.query_start_loc_np,
            req_states.prefill_len.np,
            req_states.num_computed_tokens_np,
            input_batch.num_tokens_after_padding,
        )
        positions = self.rope_state.get_positions(input_batch.num_tokens_after_padding)
        if self.model_config.uses_mrope:
            prepare_mrope_cos_sin_slices_from_runner(self, positions)
        return {"positions": positions}

    def custom_sampler(self, sampler):
        del sampler
        return Ascend310PSampler(), None


class Ascend310PModelState(_Ascend310PModelStateMixin, AscendModelState):
    """Model state with Triton-free 310P sampler / MRoPE and encoder support."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        # Initialize the full Ascend/DefaultModelState contract first so
        # attributes such as ``prompt_embeds_state`` / ``encoder_runner`` exist,
        # then swap RoPE to the Triton-free 310P implementation.
        # follow-imports=skip hides DefaultModelState.__init__; ignore call-arg.
        AscendModelState.__init__(  # type: ignore[call-arg]
            self, vllm_config, model, encoder_cache, device
        )
        # ACLGraph replays the tensor addresses bound during capture. Keep every
        # captured seq_lens buffer so its contents can be refreshed before replay.
        self._capture_seq_lens_by_ptr = {}
        self._replace_310p_rope_state(encoder_cache)


class Ascend310PMambaHybridModelState(_Ascend310PModelStateMixin, AscendMambaHybridModelState):
    """310P hybrid/GDN state: keep Ascend hybrid contract, swap Triton RoPE."""

    def add_request(self, req_index: int, new_req_data) -> None:
        super().add_request(req_index, new_req_data)
        if not self._align_mode:
            return
        # Seed running-state block from resumed prefix length. Use the Mamba
        # page size (``mamba_block_size`` when APC+align is enabled) so the
        # index matches ``mamba_get_block_table_tensor`` / ``MAMBA_BLOCK_SIZE``.
        block_size = self.cache_config.mamba_block_size or self.cache_config.block_size
        self._mamba_state_idx_gpu[req_index].fill_((new_req_data.num_computed_tokens - 1) // block_size)

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        # Initialize the complete upstream/Ascend hybrid contract first (e.g.
        # ``_align_mode`` / mamba metadata), then replace Triton RoPE.
        AscendMambaHybridModelState.__init__(  # type: ignore[call-arg]
            self, vllm_config, model, encoder_cache, device
        )
        self._capture_seq_lens_by_ptr = {}
        self._replace_310p_rope_state(encoder_cache)

    def preprocess_state(
        self,
        input_batch: AscendInputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        """Triton-free align preprocess + precopy for 310P prefix caching.

        Upstream MRv2 uses ``preprocess_mamba_align_fused_kernel[grid]`` which is
        unavailable without Triton. Mirror the same semantics with torch ops and
        tensor-view copies (same approach as MRv1 310P mamba utils fallback).
        """
        if not self._align_mode:
            return
        num_reqs = input_batch.num_reqs
        if num_reqs == 0:
            return

        mamba_group_ids, mamba_spec = self._get_mamba_group_info(kv_cache_config)
        block_size = int(mamba_spec.block_size)
        idx_mapping = input_batch.idx_mapping[:num_reqs].to(dtype=torch.long)
        valid = idx_mapping >= 0
        if not bool(valid.any().item()):
            return
        req_indices = idx_mapping.masked_select(valid)

        state_idx = self._mamba_state_idx_gpu.index_select(0, req_indices)
        num_accepted = self.num_accepted_tokens_gpu.index_select(0, req_indices)
        src_off = torch.maximum(num_accepted - 1, torch.zeros_like(num_accepted))
        self._mamba_src_col_gpu.index_copy_(0, req_indices, state_idx)
        self._mamba_src_off_gpu.index_copy_(0, req_indices, src_off.to(self._mamba_src_off_gpu.dtype))

        num_computed = num_computed_tokens.index_select(0, req_indices)
        query_start_loc = input_batch.query_start_loc[: num_reqs + 1]
        # query_start_loc is in batch order; align with valid batch rows.
        batch_rows = torch.arange(num_reqs, device=idx_mapping.device, dtype=torch.long)
        batch_rows = batch_rows.masked_select(valid)
        query_lens = query_start_loc.index_select(0, batch_rows + 1) - query_start_loc.index_select(0, batch_rows)
        computed_after = num_computed + query_lens.to(num_computed.dtype)
        new_state_idx = (computed_after + block_size - 1) // block_size - 1
        new_state_idx = new_state_idx.to(dtype=self._mamba_state_idx_gpu.dtype)
        self._mamba_state_idx_gpu.index_copy_(0, req_indices, new_state_idx)

        should_reset = (state_idx >= 0) & (state_idx != new_state_idx)
        reset_indices = req_indices.masked_select(should_reset)
        if reset_indices.numel() > 0:
            DeviceOperator.index_fill(self.num_accepted_tokens_gpu, 0, reset_indices, 1)

        self._precopy_mamba_align_torch(
            input_batch=input_batch,
            block_tables=block_tables,
            kv_cache_config=kv_cache_config,
            mamba_group_ids=mamba_group_ids,
            num_reqs=num_reqs,
        )

    def _precopy_mamba_align_torch(
        self,
        input_batch: AscendInputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        mamba_group_ids: list[int],
        num_reqs: int,
    ) -> None:
        """Copy mamba state across block boundaries without Triton."""
        from vllm_ascend.patch.worker.patch_mamba_utils import _tensor_view_from_data_ptr

        forward_context = self.vllm_config.compilation_config.static_forward_context
        copy_funcs = self.model.get_mamba_state_copy_func()
        for batch_i in range(num_reqs):
            req_idx = int(input_batch.idx_mapping[batch_i].item())
            if req_idx < 0:
                continue
            src_col = int(self._mamba_src_col_gpu[req_idx].item())
            dst_col = int(self._mamba_state_idx_gpu[req_idx].item())
            if src_col < 0 or dst_col < 0 or src_col == dst_col:
                continue
            token_bias = int(self._mamba_src_off_gpu[req_idx].item())
            for group_id in mamba_group_ids:
                block_ids = block_tables[group_id][batch_i].detach().to("cpu").tolist()
                # Drop padded / unused slots.
                while block_ids and block_ids[-1] < 0:
                    block_ids.pop()
                if not block_ids or src_col >= len(block_ids) or dst_col >= len(block_ids):
                    continue
                dest_block_id = block_ids[dst_col]
                layer_names = kv_cache_config.kv_cache_groups[group_id].layer_names
                for layer_name in layer_names:
                    attention = forward_context[layer_name]
                    kv_caches = attention.kv_cache
                    for state, state_copy_func in zip(kv_caches, copy_funcs):
                        copy_spec = state_copy_func(state, block_ids, src_col, token_bias + 1)
                        src_state = _tensor_view_from_data_ptr(state, copy_spec.start_addr, copy_spec.num_elements)
                        dst_state = _tensor_view_from_data_ptr(
                            state, state[dest_block_id].data_ptr(), copy_spec.num_elements
                        )
                        dst_state.copy_(src_state.clone())

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        # Upstream uses Triton scatter kernels. On 310P the decorated kernel is
        # unusable; keep the op Triton-free via NPU indexing. Filter padding
        # ``-1`` indices: ``index_fill_`` treats ``-1`` as the last slot.
        del num_computed_tokens

        valid = idx_mapping >= 0
        valid_indices = idx_mapping.masked_select(valid).to(dtype=torch.long)
        if valid_indices.numel() == 0:
            return

        if isinstance(num_sampled, int):
            DeviceOperator.index_fill(self.num_accepted_tokens_gpu, 0, valid_indices, max(num_sampled, 1))
            return

        accepted = torch.clamp(num_sampled.masked_select(valid), min=1).to(self.num_accepted_tokens_gpu.dtype)
        self.num_accepted_tokens_gpu.index_copy_(0, valid_indices, accepted)


class Ascend310PQwen4ExpModelState(Ascend310PMambaHybridModelState):
    """310P Qwen4Exp hybrid state with rollback-safe PLE n-gram context.

    Ports the fork ``Qwen4ExpModelState`` PLE input maintenance onto the 310P
    Triton-free hybrid state (``Ascend310PMambaHybridModelState``):

    * an n-gram CONTEXT BUFFER rebuilt every step from the authoritative
      ``num_computed_tokens`` -- because the context is reconstructed from
      scratch rather than appended to, a rejected-speculative rollback (which
      shrinks ``num_computed_tokens``) needs no special handling and the S=0
      degenerate-first step falls out naturally as all-EOS padding;
    * EOS padding for the leading tokens of a request (positions before index
      0 clamp to EOS);
    * a fixed-shape ``ple_query_start_loc`` with unused capacity represented as
      trailing zero-length requests.

    Buffers keep a FIXED shape and address as the active request count changes
    between replays, so PLE runs inside graph-captured regions even in eager
    mode. PP=1 is enforced: non-first pipeline ranks never receive the raw
    ``input_ids`` that PLE needs.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        # Build the full 310P hybrid contract first (mamba metadata, Triton-free
        # RoPE), then add the rollback-safe PLE n-gram context buffers.
        super().__init__(vllm_config, model, encoder_cache, device)
        self._init_ngram_context(vllm_config)

    def _init_ngram_context(self, vllm_config: VllmConfig) -> None:
        config = self.model_config.hf_text_config
        self.uses_ngram_embedding = bool(getattr(config, "ple_layer_ids", None))
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return

        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise RuntimeError(
                "N-gram PLE embedding currently requires pipeline_parallel_size=1 "
                "because non-first pipeline ranks do not receive the raw input_ids "
                "required by PLE. Please run with PP=1."
            )

        self.ngram_context_len = int(config.ngram_size) - 1
        if self.ngram_context_len <= 0:
            raise ValueError("N-gram embedding requires context length >= 1.")
        self.ngram_eos_token_id = int(config.eos_token_id)
        # PLE runs inside captured regions, so these buffers keep a fixed shape
        # and address as the active request count changes between replays.
        self.ngram_context = torch.full(
            (self.max_num_reqs, self.ngram_context_len),
            self.ngram_eos_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        self.ngram_context_offsets = torch.arange(
            -self.ngram_context_len,
            0,
            dtype=torch.int64,
            device=self.device,
        )
        self.ple_query_start_loc = torch.zeros(
            self.max_num_reqs + 1,
            dtype=torch.int32,
            device=self.device,
        )

    def _prepare_ngram_context(
        self,
        input_batch: AscendInputBatch,
        req_states: Any,
    ) -> torch.Tensor:
        """Rebuild the n-gram context buffer from authoritative token state.

        Row ``i`` holds the ``ngram_context_len`` tokens preceding
        ``num_computed_tokens`` of request ``idx_mapping[i]`` (its next-token
        position), so it stays correct across chunk boundaries. Positions before
        index 0 are EOS-padded. Recomputed each step, so speculative rollbacks
        (a shrunk ``num_computed_tokens``) are handled with no stale carryover.
        """
        num_reqs = input_batch.num_reqs
        context = self.ngram_context
        context.fill_(self.ngram_eos_token_id)
        if num_reqs == 0:
            return context

        request_indices = input_batch.idx_mapping[:num_reqs].long()
        context_end = req_states.num_computed_tokens.gpu[request_indices].long()
        token_indices = context_end.unsqueeze(1) + self.ngram_context_offsets
        valid_tokens = token_indices >= 0
        token_indices.clamp_min_(0)
        context_tokens = req_states.all_token_ids.gpu[request_indices.unsqueeze(1), token_indices]
        context[:num_reqs].copy_(
            torch.where(
                valid_tokens,
                context_tokens,
                context_tokens.new_full((), self.ngram_eos_token_id),
            )
        )
        return context

    def _fill_ple_query_start_loc(self, input_batch: AscendInputBatch) -> torch.Tensor:
        """Fixed-shape query_start_loc; pad with trailing zero-length requests."""
        num_reqs_padded = input_batch.num_reqs_after_padding
        query_start_loc = self.ple_query_start_loc
        query_start_loc[: num_reqs_padded + 1].copy_(input_batch.query_start_loc[: num_reqs_padded + 1])
        # Represent unused capacity as trailing zero-length requests.
        query_start_loc[num_reqs_padded + 1 :].copy_(input_batch.query_start_loc[num_reqs_padded])
        return query_start_loc

    def _fill_ple_query_start_loc_dummy(self, num_reqs: int, num_tokens: int) -> torch.Tensor:
        query_start_loc = self.ple_query_start_loc
        query_start_loc[0] = 0
        tokens_per_req, num_extra_tokens = divmod(num_tokens, num_reqs)
        query_lens = torch.full(
            (num_reqs,),
            tokens_per_req,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )
        if num_extra_tokens > 0:
            query_lens[-num_extra_tokens:] += 1
        torch.cumsum(query_lens, dim=0, out=query_start_loc[1 : num_reqs + 1])
        query_start_loc[num_reqs + 1 :].fill_(num_tokens)
        return query_start_loc

    def prepare_inputs(self, input_batch: AscendInputBatch, req_states: Any) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if not self.uses_ngram_embedding:
            return model_inputs
        model_inputs.update(
            query_start_loc=self._fill_ple_query_start_loc(input_batch),
            ngram_context=self._prepare_ngram_context(input_batch, req_states),
        )
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if not self.uses_ngram_embedding:
            return model_inputs
        ngram_context = self.ngram_context
        ngram_context.fill_(self.ngram_eos_token_id)
        model_inputs.update(
            query_start_loc=self._fill_ple_query_start_loc_dummy(num_reqs, num_tokens),
            ngram_context=ngram_context,
        )
        return model_inputs

    # ------------------------------------------------------------------
    # GDN state lifecycle (plan T5.2)
    #
    # Prefill / decode / preemption / reuse for the Gated DeltaNet conv +
    # recurrent (SSM) state. State is REPLICATED PER TP RANK, so one
    # ``Qwen4ExpGDNStatePool`` is created per rank; the copy / slot-remap
    # semantics (no aliasing, clean reuse, fail-closed preemption) live in the
    # pool and mirror the fork ``MambaAttentionBackendEnum.GDN_ATTN`` copy-func
    # pair. These thin methods fan a lifecycle event across all ranks.
    # ------------------------------------------------------------------
    def _init_gdn_state_pools(
        self,
        params: Qwen4ExpGDNParams,
        *,
        num_blocks: int,
        tp_size: int = 1,
        num_ranks: int = 1,
        num_spec: int = 0,
        layout: GDNConvStateLayout = "DS",
        policy: Qwen4ExpDtypePolicy = ASCEND_QWEN4EXP_DTYPE_POLICY,
        device: torch.device | str | None = None,
    ) -> list[Qwen4ExpGDNStatePool]:
        """Build one per-rank GDN state pool (state replicated per TP rank)."""
        if num_ranks <= 0:
            raise ValueError(f"num_ranks must be >= 1, got {num_ranks}.")
        self.gdn_state_layout = Qwen4ExpGDNStateLayout.from_params(
            params, tp_size=tp_size, num_spec=num_spec, policy=policy, layout=layout
        )
        self.gdn_num_ranks = num_ranks
        pool_device = device if device is not None else self.device
        self._gdn_state_pools = [
            Qwen4ExpGDNStatePool(num_blocks, self.gdn_state_layout, device=pool_device) for _ in range(num_ranks)
        ]
        return self._gdn_state_pools

    def gdn_pool(self, rank: int = 0) -> Qwen4ExpGDNStatePool:
        return self._gdn_state_pools[rank]

    def gdn_begin_request(self, request_id: int) -> list[int]:
        """Allocate a fresh (zeroed, non-aliasing) state block on every rank."""
        return [pool.allocate(request_id) for pool in self._gdn_state_pools]

    def gdn_complete_request(self, request_id: int) -> None:
        """Free a completed request's blocks on every rank (zeroed for reuse)."""
        for pool in self._gdn_state_pools:
            pool.free(request_id)

    def gdn_preempt_request(self, request_id: int) -> None:
        """Fail-closed preemption on every rank; resume must re-seed + recompute."""
        for pool in self._gdn_state_pools:
            pool.preempt(request_id)

    def gdn_is_resident(self, request_id: int) -> bool:
        return all(pool.is_resident(request_id) for pool in self._gdn_state_pools)

    def gdn_active_blocks(self, rank: int = 0) -> dict[int, int]:
        """``request_id -> block_id`` on ``rank`` (for aliasing audits)."""
        return self._gdn_state_pools[rank].active_blocks()

    def gdn_recurrent_state(self, request_id: int, rank: int = 0) -> torch.Tensor:
        return self._gdn_state_pools[rank].recurrent_state(request_id)

    def gdn_write_recurrent(self, request_id: int, state: torch.Tensor, rank: int = 0) -> None:
        self._gdn_state_pools[rank].write_recurrent(request_id, state)

    def gdn_step(
        self,
        request_id: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        rank: int = 0,
        chunked: bool,
        scale: float | None = None,
        use_qk_l2norm: bool = True,
        compute_dtype: torch.dtype | None = None,
        backend: GDNBackend = "eager",
    ) -> torch.Tensor:
        """Advance one request's GDN state by a prefill chunk or a decode step.

        Reads the request's resident recurrent state (fail-closed if the request
        was preempted/completed and not re-seeded), runs the gated delta rule
        continuing from it, then writes the new state back into the same block.
        """
        pool = self._gdn_state_pools[rank]
        initial_state = pool.recurrent_state(request_id)
        out, new_state = gdn_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            chunked=chunked,
            scale=scale,
            use_qk_l2norm=use_qk_l2norm,
            compute_dtype=compute_dtype,
            backend=backend,
        )
        pool.write_recurrent(request_id, new_state)
        return out
