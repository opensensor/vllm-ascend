#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Experimental Multi-head Latent Attention (MLA) backend for Ascend 310P.

Status: BRING-UP / HARDWARE VALIDATION IN PROGRESS. Only reachable when
``VLLM_ASCEND_310P_ENABLE_MLA=1`` (see ``vllm_ascend.envs``); the platform wiring
in ``platform.get_attn_backend_cls`` registers this backend under the 310P
compatibility map key ``(use_mla=True, use_sparse=False)`` only under that flag.

Why a separate 310P MLA backend
-------------------------------
The STANDARD MLA implementation (:class:`vllm_ascend.attention.mla_v1.AscendMLAImpl`)
has two front-ends and one attention core:

* Optimized front-end: ``mla_preprocess`` and ``npu_mla_prolog_v3`` custom ops.
  Both are compiled ONLY in the non-310P (``#else``) branch of
  ``csrc/torch_binding.cpp`` (the ``mla_preprocess`` def sits inside
  ``VLLM_ENABLE_ATB_AND_DIRECT_KERNELS`` within that ``#else``; the
  ``npu_mla_prolog_v3`` comment states the underlying aclnn op is *950-only*).
  => Neither op exists on ascend310p1. This backend MUST NOT take that path.
* Decomposed front-end: ``_mla_preprocess`` -> ``mla_preprocess_decode`` /
  ``mla_preprocess_prefill``. These use only plain projection matmuls, the
  weight-absorption ``bmm`` (``_q_proj_and_k_up_proj`` / ``_v_up_proj``),
  ``rope_single`` and ``exec_kv_*``. This is the path we reuse on 310P, and it
  is naturally selected because ``enabling_mlapo()`` already returns ``False``
  on the 310P hardware profile (no ``UNRESTRICTED_MLAPO`` capability and KV
  transfer is disabled on 310P).
* Attention core: the inherited FIA operators have no ascend310p1 binary.
  Prefill therefore uses the native 310P flash-attention operator on the
  materialized 256-wide heads. Decode uses the dedicated QSA AscendC kernel as
  paged latent MQA over 512-wide NZ cache rows, followed by the normal value
  up-projection. The decomposed NPU implementation remains a numerical oracle.

Target model
------------
GLM-5.3-Flash-W2-310p (``Glm5NextForConditionalGeneration``) is NoPE MLA:
``qk_rope_head_dim == 0``, ``mla_use_nope == True``, ``kv_lora_rank == 512``,
``qk_nope_head_dim == v_head_dim == 256``, 64 heads. Only 11 of 45 layers are
attention (``deepseek_sparse_attention``); the other 34 are ``linear_attention``
(KDA/GDN, already working on 310P). Its kpool indexer scores compressed
four-token keys and selects up to 512 pools. The QSA kernel attends those
2,048 tokens and the current incomplete pool.

NoPE removes the decoupled-rope machinery (``npu_kv_rmsnorm_rope_cache`` rope
part, ``npu_interleave_rope``) from the critical path, which is what makes a
310P port tractable without new rope-cache kernels.
"""

from __future__ import annotations

import torch
import torch_npu

from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310
from vllm_ascend.attention.mla_v1 import (
    AscendMLABackend,
    AscendMLAImpl,
    AscendMLAMetadata,
    AscendMLAMetadataBuilder,
    DecodeMLAPreprocessResult,
)
from vllm_ascend.attention.utils import (
    notify_kv_cache_written,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import (
    record_attention_compute_start,
)
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, nd_to_nz_2d

# 310P kernel tiling constraint (see commit e8f7b2e3f, "support attn_head_size
# larger than 128"): block_size * head_size must not exceed 16384 elements for
# the paged/flash attention operators. The MLA latent cache head size is
# kv_lora_rank + qk_rope_head_dim (512 + 0 = 512 for GLM NoPE), so the block
# size must satisfy block_size <= 16384 / 512 = 32.
_ASCEND_310P_MAX_BLOCK_TIMES_HEAD = 16384
_QSA_COMPRESS_RATIO = 4
_QSA_KERNEL_BLOCK_SIZE = 32
_NPU_TOKEN_ALIGNMENT = 32
_NZ_INNER = 16


def _grouped_absorbed_key_projection(
    q_nope: torch.Tensor,
    weight_uk_t: torch.Tensor,
    group_list: torch.Tensor,
) -> torch.Tensor:
    """Project batch-major MLA queries with graph-safe 310P NZ GMM.

    ``torch.bmm`` selects the legacy, non-capturable ``BatchMatMul`` aclop
    when ``weight_uk_t`` is FRACTAL_NZ on 310P. Grouped matmul uses the Cube
    path, accepts the same NZ weights, and is capturable without converting a
    64-head weight back to ND on every layer.
    """
    num_tokens, num_heads, qk_dim = q_nope.shape
    grouped_query = q_nope.transpose(0, 1).contiguous().view(num_heads * num_tokens, qk_dim)
    projected = torch_npu.npu_grouped_matmul(
        x=[grouped_query],
        weight=[weight_uk_t],
        group_list=group_list,
        split_item=2,
        group_type=0,
    )[0]
    return projected.view(num_heads, num_tokens, -1).transpose(0, 1)


def _qsa_physical_cache_page(cache: torch.Tensor) -> torch.Tensor:
    """Expose a padded NZ page without copying its logical latent slice.

    GLM's MLA view can contain only the first 512 channels of a larger page.
    The native QSA kernel addresses the whole physical page using its channel
    count, while a separate argument supplies the logical KV head count.
    """
    if cache.ndim != 4 or cache.shape[3] != _NZ_INNER:
        raise ValueError("QSA cache must have NZ [blocks, channels/16, block, 16] shape")
    _, logical_channels, block_size, inner = cache.shape
    channel_stride = block_size * inner
    page_stride = cache.stride(0)
    if cache.stride()[1:] != (channel_stride, inner, 1) or page_stride % channel_stride:
        raise ValueError("QSA cache has an unsupported physical page layout")
    physical_channels = page_stride // channel_stride
    if physical_channels < logical_channels:
        raise ValueError("QSA physical page is smaller than its logical latent view")
    if physical_channels == logical_channels:
        return cache
    return cache.as_strided(
        (cache.shape[0], physical_channels, block_size, inner),
        (page_stride, channel_stride, inner, 1),
    )


def _qsa_cache_block_table(block_table: torch.Tensor, cache_block_size: int) -> torch.Tensor:
    """Map the MLA kernel's split block IDs to the shared cache's pages.

    GLM's shared-slot allocator stores one scheduler page per physical tensor
    block, while the attention metadata splits each scheduler page into 32-token
    kernel blocks. QSA addresses the physical tensor page size directly.
    """
    if cache_block_size < _QSA_KERNEL_BLOCK_SIZE or cache_block_size % _QSA_KERNEL_BLOCK_SIZE:
        raise ValueError(f"QSA cache block size is incompatible with 32-token metadata: {cache_block_size}")
    split = cache_block_size // _QSA_KERNEL_BLOCK_SIZE
    if split == 1:
        return block_table
    return torch.div(block_table[:, ::split], split, rounding_mode="floor").contiguous()


def _write_nz_latent_cache(
    cache: torch.Tensor,
    rows: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Write latent rows into the explicit ``[block, D/16, token, 16]`` layout.

    GLM's shared hybrid cache is an ND page-strided view whose dimensions
    already describe the physical NZ row order consumed by the 310P QSA
    kernel. CANN's generic reshape-and-cache operator requires FORMAT_NZ and
    rejects that shared ND view, so write the rows directly on device without
    converting or duplicating the cache.
    """
    if cache.ndim != 4 or cache.shape[-1] != _NZ_INNER:
        raise ValueError("310P MLA cache must be [blocks, D/16, block, 16]")
    if rows.ndim != 3 or rows.shape[1] * rows.shape[2] != cache.shape[1] * _NZ_INNER:
        raise ValueError("310P MLA latent rows do not match the cache head width")
    if slot_mapping.ndim != 1 or slot_mapping.shape[0] != rows.shape[0]:
        raise ValueError("310P MLA slot mapping must contain one entry per latent row")

    if cache.device.type == "npu":
        op_namespace = getattr(torch.ops, "_C_ascend", None)
        op = None if op_namespace is None else getattr(op_namespace, "mla_cache_write_310", None)
        if op is not None:
            # The vLLM scheduler supplies int32 slots on the 310P; the
            # fused cache writer reads int64 slots. Convert before dispatch so
            # its device kernel never interprets pairs of int32 IDs as one.
            op(cache, rows.contiguous(), slot_mapping.to(torch.int64).contiguous())
            return

    # The device indexing path also works for shared, page-strided GLM cache
    # views when the optional fused cache-write extension is unavailable.
    valid = slot_mapping >= 0
    valid_slots = slot_mapping[valid].to(torch.long)
    block_size = cache.shape[2]
    physical_blocks = torch.div(valid_slots, block_size, rounding_mode="floor")
    token_offsets = valid_slots.remainder(block_size)
    cache[physical_blocks, :, token_offsets, :] = rows[valid].to(cache.dtype)


class AscendMLAMetadataBuilder310(AscendMLAMetadataBuilder):
    """310P MLA metadata with one shared native prefill mask per layer group."""

    def __init__(self, *args, **kwargs) -> None:
        self.glm_indexer = kwargs.get("indexer") if getattr(kwargs.get("indexer"), "index_kpool", 1) > 1 else None
        super().__init__(*args, **kwargs)
        self._native_prefill_mask: torch.Tensor | None = None
        self._native_prefill_mask_size = 0

    def _get_native_prefill_mask(self, seq_len: int) -> torch.Tensor:
        mask_size = (seq_len + _NPU_TOKEN_ALIGNMENT - 1) // _NPU_TOKEN_ALIGNMENT * _NPU_TOKEN_ALIGNMENT
        if self._native_prefill_mask is not None and self._native_prefill_mask_size >= mask_size:
            return self._native_prefill_mask
        mask = AttentionMaskBuilder310.gen_causal_additive_mask(
            mask_size,
            self.device,
        )
        self._native_prefill_mask = torch_npu.npu_format_cast(
            nd_to_nz_2d(mask),
            ACL_FORMAT_FRACTAL_NZ,
        )
        self._native_prefill_mask_size = mask_size
        return self._native_prefill_mask

    def build_prefill_metadata(self, *args, **kwargs):
        metadata = super().build_prefill_metadata(*args, **kwargs)
        metadata.attn_mask = self._get_native_prefill_mask(max(1, metadata.max_query_len))
        return metadata

    def build_decode_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata,
    ):
        metadata = super().build_decode_metadata(
            common_prefix_len,
            common_attn_metadata,
        )
        # The native paged kernel consumes the runner's replay-updated device
        # lengths directly. Keep the CPU list separately for scheduling and
        # avoid rebuilding an H2D tensor in the first MLA layer of every step.
        metadata.seq_lens = common_attn_metadata.seq_lens[: self.num_decodes]
        return metadata


class AscendMLAImpl310(AscendMLAImpl):
    """310P MLA attention implementation.

    Reuses the decomposed NoPE front-end from :class:`AscendMLAImpl` and routes
    both attention phases to 310P-native kernels. Unsupported FIA calls are not
    reachable from this implementation.
    """

    uses_nz_cache = True
    glm_indexer = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        indexer = kwargs.get("indexer")
        self.glm_indexer = indexer if getattr(indexer, "index_kpool", 1) > 1 else None

        # Hard invariants for the 310P bring-up path. These must hold for the
        # decomposed front-end to be the one that executes.
        if self.enable_mlapo:
            # Would route to the csrc mla_preprocess op, which is not built for
            # 310P. enabling_mlapo() should already keep this False on 310P.
            raise NotImplementedError(
                "AscendMLAImpl310: enable_mlapo must be False on 310P "
                "(mla_preprocess / npu_mla_prolog_v3 are not compiled for "
                "ascend310p1)."
            )

        # NoPE is the only validated shape for 310P MLA today. A non-zero
        # qk_rope_head_dim would additionally require npu_kv_rmsnorm_rope_cache /
        # npu_interleave_rope to be proven on 310P.
        self._is_nope = self.qk_rope_head_dim == 0
        if not self._is_nope:
            raise NotImplementedError(
                "AscendMLAImpl310 currently supports only NoPE MLA "
                f"(qk_rope_head_dim == 0); got {self.qk_rope_head_dim}. "
                "Decoupled-rope MLA on 310P needs npu_kv_rmsnorm_rope_cache and "
                "npu_interleave_rope validated on ascend310p1 first."
            )

        self._decode_constant_buffers: dict[
            tuple[torch.device, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._absorbed_key_group_lists: dict[int, torch.Tensor] = {}
        self.host_kv_layer = None

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        super().process_weights_after_loading(act_dtype)
        capture_sizes = getattr(self.vllm_config.compilation_config, "cudagraph_capture_sizes", None) or ()
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        decode_sizes = set(range(1, max_num_seqs + 1))
        decode_sizes.update(capture_sizes)
        for num_tokens in decode_sizes:
            if num_tokens not in self._absorbed_key_group_lists:
                self._absorbed_key_group_lists[num_tokens] = torch.arange(
                    num_tokens,
                    self.num_heads * num_tokens + 1,
                    num_tokens,
                    dtype=torch.int64,
                    device=self.W_UK_T.device,
                )

    def _get_absorbed_key_group_list(self, num_tokens: int) -> torch.Tensor:
        group_list = self._absorbed_key_group_lists.get(num_tokens)
        if group_list is None:
            group_list = torch.arange(
                num_tokens,
                self.num_heads * num_tokens + 1,
                num_tokens,
                dtype=torch.int64,
                device=self.W_UK_T.device,
            )
            self._absorbed_key_group_lists[num_tokens] = group_list
        return group_list

    def _q_proj_and_k_up_proj(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_nope, q_pe = (
            self.q_proj(x)[0]
            .view(-1, self.num_heads, self.qk_head_dim)
            .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        )
        ql_nope = _grouped_absorbed_key_projection(
            q_nope,
            self.W_UK_T,
            self._get_absorbed_key_group_list(q_nope.shape[0]),
        )
        return ql_nope, q_pe

    def _mla_preprocess(self, layer_name, hidden_states, kv_cache, attn_metadata):
        if self.glm_indexer is None:
            return super()._mla_preprocess(layer_name, hidden_states, kv_cache, attn_metadata)

        # Share the fused down-projection and normalized q-LoRA activation
        # between GLM's indexer and the regular MLA front-end.
        if self.fused_qkv_a_proj is None or self.q_a_layernorm is None:
            raise RuntimeError("GLM kpool requires fused q-LoRA and its layer norm")
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_no_split = qkv_lora.split([self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)
        q_c = self.q_a_layernorm(q_c)
        kv_no_split = kv_no_split.contiguous()
        input_positions = []
        if attn_metadata.decode is not None:
            input_positions.append(attn_metadata.decode.input_positions)
        if attn_metadata.prefill is not None:
            input_positions.append(attn_metadata.prefill.input_positions)
        positions = input_positions[0] if len(input_positions) == 1 else torch.cat(input_positions)
        num_tokens = attn_metadata.num_actual_tokens
        self.glm_indexer(hidden_states[:num_tokens], q_c[:num_tokens], positions[:num_tokens], None)

        decode_result = None
        prefill_result = None
        if attn_metadata.num_prefills > 0:
            wait_for_kv_layer_from_connector(layer_name)
        if attn_metadata.num_decodes > 0:
            decode_result = self.mla_preprocess_decode(q_c, kv_no_split, kv_cache, attn_metadata)
        if attn_metadata.num_prefills > 0:
            prefill_result = self.mla_preprocess_prefill(q_c, kv_no_split, kv_cache, attn_metadata)
        notify_kv_cache_written(layer_name)
        return decode_result, prefill_result

    def _get_kpool_qsa_plan(
        self, positions: torch.Tensor, token_start: int, num_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indexer = self.glm_indexer
        if indexer is None:
            raise RuntimeError("GLM kpool indexer is missing")
        pool_size = indexer.index_kpool
        pool_budget = indexer.topk_tokens // pool_size
        token_ids = indexer.topk_indices_buffer[
            token_start : token_start + num_tokens, : indexer.topk_tokens : pool_size
        ]
        group_ids = torch.div(token_ids, pool_size, rounding_mode="floor").to(torch.int32).contiguous()
        seq_lens = positions[:num_tokens].to(torch.int32) + 1
        complete_pools = torch.div(seq_lens, pool_size, rounding_mode="floor")
        group_counts = complete_pools.clamp(max=pool_budget)
        tail_starts = complete_pools * pool_size
        tail_counts = seq_lens - tail_starts
        # Every visible token fits in the sparse budget for short sequences.
        # Use the QSA kernel's dense-length encoding there: it avoids reading
        # an indexer selection vector and gives the exact same token set.
        dense = seq_lens <= indexer.topk_tokens
        group_counts = torch.where(dense, seq_lens, group_counts)
        tail_counts = torch.where(dense, -1, tail_counts)
        return group_ids, group_counts, tail_starts, tail_counts

    def _v_up_proj(self, x: torch.Tensor) -> torch.Tensor:
        """Project latent attention heads with the 310P-supported batched GEMM."""
        latent = x.view(self.num_heads, -1, self.kv_lora_rank)
        value = torch.bmm(latent, self.W_UV)
        return value.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)

    # ---------------------------------------------------------------------
    # 310P attention cores. Prefill uses the device flash-attention kernel;
    # decode uses an NPU-resident correctness path while its fused paged-latent
    # kernel is brought up.
    # ---------------------------------------------------------------------
    def _exec_kv_mla_nope(
        self,
        kv_no_split: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        slots: torch.Tensor,
        is_prefill: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize and write the latent once into both native NZ caches."""
        batch, num_kv_heads, sequence, _ = kv_no_split.shape
        if self.kv_a_layernorm is None:
            raise RuntimeError("310P NoPE MLA requires a KV layer normalization")
        latent = self.kv_a_layernorm(kv_no_split.reshape(-1, self.kv_lora_rank)).view(
            batch, num_kv_heads, sequence, self.kv_lora_rank
        )
        cache_rows = latent.reshape(-1, num_kv_heads, self.kv_lora_rank)
        cache_rows = cache_rows.view(-1, self.kv_lora_rank // _NZ_INNER, _NZ_INNER)
        if self.host_kv_layer is None:
            _write_nz_latent_cache(kv_cache[0], cache_rows, slots)
            if kv_cache[1].data_ptr() != kv_cache[0].data_ptr():
                _write_nz_latent_cache(kv_cache[1], cache_rows, slots)
        else:
            self.host_kv_layer.write(cache_rows, slots)
        empty_rope = latent.new_empty(batch, num_kv_heads, sequence, 0)
        if is_prefill:
            return empty_rope, latent
        return kv_cache[1], kv_cache[0]

    def _forward_prefill_naive(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        value: torch.Tensor,
        kv_c_and_k_pe_cache: tuple[torch.Tensor],
        attn_metadata,
    ) -> torch.Tensor:
        """Run materialized NoPE MLA prefill with native 310P flash attention.

        GLM materializes 256-wide K and V for prefill, so this is ordinary MHA
        despite its 512-wide latent decode cache.  The 310P kernel requires
        host-resident per-request lengths and an NZ causal mask.  Both are
        derived from metadata that already exists on the host, avoiding a
        device-to-host synchronization in the prefill hot path.
        """
        del q_pe, k_pe, kv_c_and_k_pe_cache
        prefill_meta = attn_metadata.prefill
        if prefill_meta is None:
            raise RuntimeError("310P MLA prefill metadata is missing")
        if prefill_meta.chunked_context is not None:
            raise NotImplementedError(
                "310P native MLA chunked-prefill context merge is not implemented; "
                "schedule this backend with PrefillNoCache until the paged-latent "
                "prefix kernel is enabled."
            )
        if not (q_nope.dtype == k_nope.dtype == value.dtype == torch.float16):
            raise NotImplementedError(
                "310P native MLA prefill currently requires FP16 Q/K/V tensors; "
                f"got query={q_nope.dtype}, key={k_nope.dtype}, value={value.dtype}."
            )

        cumulative_lengths = prefill_meta.actual_seq_lengths_q
        if not cumulative_lengths:
            raise RuntimeError("310P MLA prefill requires host cumulative query lengths")
        num_tokens = q_nope.shape[0]
        padding = num_tokens - cumulative_lengths[-1]
        if padding < 0:
            raise RuntimeError(
                "310P MLA prefill metadata describes more tokens than the input: "
                f"{cumulative_lengths[-1]} > {num_tokens}."
            )
        sequence_lengths = prefill_meta.query_lens
        if sequence_lengths.device.type != "cpu":
            raise RuntimeError("310P MLA prefill query lengths must remain host-resident.")
        if sequence_lengths.dtype != torch.int32:
            sequence_lengths = sequence_lengths.to(torch.int32)
        if padding:
            sequence_lengths = sequence_lengths.clone()
            sequence_lengths[-1] += padding
        mask = prefill_meta.attn_mask
        if mask is None or mask.dtype != torch.float16:
            raise RuntimeError(
                "310P MLA prefill requires the shared FP16 NZ causal mask from AscendMLAMetadataBuilder310."
            )
        output = torch.empty_like(value)
        record_attention_compute_start()
        torch_npu._npu_flash_attention(
            query=q_nope,
            key=k_nope.contiguous(),
            value=value.contiguous(),
            mask=mask,
            seq_len=sequence_lengths,
            scale_value=self.scale,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            out=output,
        )
        return output.reshape(num_tokens, self.num_heads * self.v_head_dim)

    def _forward_prefill_paged_latent(
        self,
        q_nope: torch.Tensor,
        kv_c_and_k_pe_cache: tuple[torch.Tensor, torch.Tensor],
        attn_metadata,
    ) -> torch.Tensor:
        """Attend a continued prefill to its already-written latent pages.

        The native 310P flash operator only handles the new chunk and does not
        return the softmax statistics needed to merge a cached prefix.  The
        paged-latent operator accepts one visible length per query token, so
        it can include both the prefix and the causal part of this chunk.
        """
        prefill_meta = attn_metadata.prefill
        if prefill_meta is None:
            raise RuntimeError("310P MLA continued prefill metadata is missing")
        if q_nope.dtype != torch.float16:
            raise NotImplementedError("310P paged-latent prefill requires FP16 queries")
        num_tokens = q_nope.shape[0]
        positions = prefill_meta.input_positions
        num_actual_tokens = prefill_meta.actual_seq_lengths_q[-1]
        if num_actual_tokens <= 0 or num_actual_tokens > min(num_tokens, positions.shape[0]):
            raise RuntimeError("310P MLA query tokens or positions are shorter than the actual prefill")

        query = torch.bmm(q_nope[:num_actual_tokens].transpose(0, 1), self.W_UK_T).transpose(0, 1).contiguous()
        # The cache write precedes attention, so position + 1 includes the
        # current token. The runner may pad Q and positions differently; only
        # the host-recorded actual prefill tokens belong in the paged kernel.
        visible_lengths = (positions[:num_actual_tokens].to(device=query.device, dtype=torch.int32) + 1).clamp_min_(1)
        block_table = _qsa_cache_block_table(
            prefill_meta.block_table.to(device=query.device, dtype=torch.int32),
            kv_c_and_k_pe_cache[0].shape[2],
        )
        addressable_tokens = block_table.shape[1] * kv_c_and_k_pe_cache[0].shape[2]
        if prefill_meta.max_seq_lens > addressable_tokens:
            # This is scheduler-owned host metadata, so checking it adds no
            # device synchronization. Reject a truncated MTP table before QSA
            # can read past its final column and turn garbage into a GM address.
            raise ValueError(
                "310P MLA prefill block table is shorter than the visible context: "
                f"{addressable_tokens} addressable tokens < {prefill_meta.max_seq_lens}. "
                "Draft metadata must retain all kernel blocks per scheduler page."
            )
        query_start_loc = prefill_meta.query_start_loc.to(device=query.device, dtype=torch.int32).contiguous()
        if self.glm_indexer is None:
            group_indices, tail_starts, tail_counts, _ = self._get_decode_constant_buffers(
                query.device, num_actual_tokens
            )
            group_counts = visible_lengths
        else:
            group_indices, group_counts, tail_starts, tail_counts = self._get_kpool_qsa_plan(
                positions, attn_metadata.num_decode_tokens, num_actual_tokens
            )
        op = self._get_paged_latent_op()
        if op is None:
            raise RuntimeError("vLLM Ascend was built without the native 310P paged-latent attention operator")
        key_cache, value_cache = kv_c_and_k_pe_cache
        if key_cache.shape != value_cache.shape:
            raise ValueError("QSA key and value caches must have the same logical shape")
        physical_key_cache = _qsa_physical_cache_page(key_cache)
        physical_value_cache = _qsa_physical_cache_page(value_cache)
        if physical_key_cache.shape != physical_value_cache.shape:
            raise ValueError("QSA key and value physical pages must have the same shape")
        # Continued prefill needs the same physical-page contract as decode.
        # The native kernel derives its page stride from the supplied shape,
        # so a narrower logical view cannot describe these padded pages.
        physical_page_args = ()
        if physical_key_cache.shape != key_cache.shape:
            head_dim_blocks = query.shape[2] // _NZ_INNER
            if head_dim_blocks == 0 or query.shape[2] % _NZ_INNER or key_cache.shape[1] % head_dim_blocks:
                raise ValueError("QSA logical cache channels must contain whole KV heads")
            physical_page_args = (key_cache.shape[1] // head_dim_blocks,)
        record_attention_compute_start()
        if self.host_kv_layer is None:
            latent_output = op(
                query,
                physical_key_cache,
                physical_value_cache,
                group_indices,
                group_counts,
                tail_starts,
                tail_counts,
                block_table,
                query_start_loc,
                self.scale,
                _QSA_COMPRESS_RATIO,
                *physical_page_args,
            )
        else:
            if self.glm_indexer is None:
                raise RuntimeError("GLM host MLA requires the kpool indexer")
            # Batch consecutive queries while their selected-page union fits
            # the hot cache; split at an overflow or request boundary.
            segments = self.host_kv_layer.prefill_segments(
                block_table,
                group_indices,
                group_counts,
                tail_starts,
                tail_counts,
                prefill_meta.actual_seq_lengths_q,
                _QSA_COMPRESS_RATIO,
            )
            outputs = []
            for start, end, request in segments:
                hot_table = self.host_kv_layer.stage_prefill(
                    block_table[request : request + 1],
                    group_indices[start:end],
                    group_counts[start:end],
                    tail_starts[start:end],
                    tail_counts[start:end],
                    _QSA_COMPRESS_RATIO,
                )
                segment_start = torch.tensor([0, end - start], dtype=torch.int32, device=query.device)
                outputs.append(
                    op(
                        query[start:end],
                        physical_key_cache,
                        physical_value_cache,
                        group_indices[start:end],
                        group_counts[start:end],
                        tail_starts[start:end],
                        tail_counts[start:end],
                        hot_table,
                        segment_start,
                        self.scale,
                        _QSA_COMPRESS_RATIO,
                        *physical_page_args,
                    )
                )
            latent_output = torch.cat(outputs, dim=1)
        projected = self._v_up_proj(latent_output.transpose(0, 1).contiguous())
        if num_actual_tokens < num_tokens:
            projected = torch.nn.functional.pad(projected, (0, 0, 0, num_tokens - num_actual_tokens))
        return projected

    def _forward_prefill(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        value: torch.Tensor,
        kv_c_and_k_pe_cache: tuple[torch.Tensor],
        attn_metadata,
    ) -> torch.Tensor:
        """Use native flash for fresh chunks and paged latent for cached ones."""
        if attn_metadata.prefill is not None and (
            attn_metadata.prefill.chunked_context is not None
            or (self.glm_indexer is not None and attn_metadata.prefill.max_seq_lens > self.glm_indexer.topk_tokens)
        ):
            return self._forward_prefill_paged_latent(
                q_nope,
                kv_c_and_k_pe_cache,
                attn_metadata,
            )
        return self._forward_prefill_naive(
            q_nope,
            q_pe,
            k_nope,
            k_pe,
            value,
            kv_c_and_k_pe_cache,
            attn_metadata,
        )

    def _forward_decode_naive(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        block_size: int,
        attn_metadata,
        dequant_scale_q_nope=None,
    ) -> torch.Tensor:
        """Weight-absorbed MLA decode over the paged latent cache.

        Contract for the hardware implementation:

        * After ``_q_proj_and_k_up_proj`` the query is absorbed into latent
          space: ``ql_nope`` has head dim ``kv_lora_rank`` (512). The paged KV
          cache stores the latent ``c_kv`` (single latent "head" of dim 512;
          NoPE => no separate rope cache). Decode is therefore MQA with
          head_size == 512, num_kv_heads == 1, num_heads == 64.
        * The attention output is latent (head_size 512) and is projected back
          to ``v_head_dim`` (256) by ``_v_up_proj`` (already inherited) AFTER
          this call returns its raw latent output.
        * Kernel requirement: a paged attention over a 512-wide latent head.
          ``torch_npu._npu_paged_attention`` is the candidate op, but:
            - It assumes query/kv/out share head_size; here that is the latent
              512, which is fine because W_UV is applied afterwards.
            - The 310P 5D NZ KV-cache layout is
              (2, num_blocks, (num_kv_heads*head_size)//16, block_size, 16);
              with head_size 512, num_kv_heads 1 that is
              (2, num_blocks, 32, block_size, 16) and requires
              block_size * 512 <= 16384 => block_size <= 32
              (see _ASCEND_310P_MAX_BLOCK_TIMES_HEAD). The MLA backend must
              therefore request a block size in {32, 16} (see
              AscendMLABackend310.get_supported_kernel_block_sizes).
            - VERIFY on hardware whether _npu_paged_attention accepts head_size
              512 at all on ascend310p1. If it does not, a new AscendC paged
              latent-MQA kernel is required with this exact signature:
                out[t, h, :512] = softmax(scale * q[t, h, :512] @ K_ctx^T) @ K_ctx
              where K_ctx are the gathered latent vectors (dim 512) for the
              block table of request(t), h in [0, 64), causal over context_lens.

        The stock 310P paged-attention operator rejects the 512-wide latent
        head.  Keep the MLA weight absorption, gather the active latent pages
        on device, and execute the two batched matmuls plus FP32 softmax with
        native NPU operators.  This is the correctness implementation and the
        exact math contract for the fused AscendC paged-latent kernel.
        """
        del q_pe, k_pe, dequant_scale_q_nope
        decode_meta = attn_metadata.decode
        if decode_meta is None:
            raise RuntimeError("310P MLA decode metadata is missing")

        num_tokens = q_nope.shape[0]
        seq_lens_list = decode_meta.seq_lens_list[:num_tokens]
        if len(seq_lens_list) != num_tokens:
            raise NotImplementedError(
                "310P decomposed MLA currently requires one decode token per request; "
                f"got {num_tokens} tokens for {len(seq_lens_list)} requests."
            )
        max_context_len = max(seq_lens_list, default=0)
        if max_context_len <= 0:
            raise RuntimeError("310P MLA decode requires a non-empty KV context")

        positions = torch.arange(
            max_context_len,
            dtype=torch.int32,
            device=q_nope.device,
        )
        logical_block_columns = torch.div(positions, block_size, rounding_mode="floor")
        block_offsets = torch.remainder(positions, block_size)
        physical_blocks = decode_meta.block_table[:num_tokens].index_select(
            1,
            logical_block_columns,
        )
        latent_keys = k_nope[
            physical_blocks,
            block_offsets.unsqueeze(0),
            0,
            :,
        ]

        query = q_nope.view(num_tokens, self.num_heads, self.kv_lora_rank)
        scores = torch.bmm(query, latent_keys.transpose(1, 2)) * self.scale
        if decode_meta.seq_lens.device != query.device or decode_meta.seq_lens.dtype != torch.int32:
            decode_meta.seq_lens = decode_meta.seq_lens.to(
                device=query.device,
                dtype=torch.int32,
                non_blocking=True,
            )
        valid_context = positions.unsqueeze(0) < decode_meta.seq_lens[:num_tokens].unsqueeze(1)
        scores.masked_fill_(~valid_context.unsqueeze(1), float("-inf"))
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        latent_output = torch.bmm(probabilities, latent_keys)
        return self._v_up_proj(latent_output.transpose(0, 1).contiguous())

    def _get_decode_constant_buffers(
        self,
        device: torch.device,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return stable dense-selection and token/request mapping buffers."""
        key = (device, num_tokens)
        buffers = self._decode_constant_buffers.get(key)
        if buffers is None:
            group_sentinel = torch.zeros((num_tokens, 1), dtype=torch.int32, device=device)
            tail_starts = torch.zeros(num_tokens, dtype=torch.int32, device=device)
            tail_counts = torch.full(
                (num_tokens,),
                -1,
                dtype=torch.int32,
                device=device,
            )
            query_start_loc = torch.arange(num_tokens + 1, dtype=torch.int32, device=device)
            buffers = (
                group_sentinel,
                tail_starts,
                tail_counts,
                query_start_loc,
            )
            self._decode_constant_buffers[key] = buffers
        return buffers

    def _forward_decode_fused(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        """Run dense latent MQA through the native paged 310P QSA kernel."""
        decode_meta = attn_metadata.decode
        if decode_meta is None:
            raise RuntimeError("310P MLA decode metadata is missing")
        if query.dtype != torch.float16:
            raise NotImplementedError(
                "The native 310P latent-attention kernel currently requires FP16 "
                f"query/cache tensors, got {query.dtype}."
            )

        num_tokens = query.shape[0]
        if num_tokens != attn_metadata.num_decodes and self.glm_indexer is None:
            raise NotImplementedError(
                "310P native MLA currently requires one decode token per request; "
                f"got {num_tokens} tokens for {attn_metadata.num_decodes} requests."
            )
        if decode_meta.seq_lens.device != query.device or decode_meta.seq_lens.dtype != torch.int32:
            decode_meta.seq_lens = decode_meta.seq_lens.to(
                device=query.device,
                dtype=torch.int32,
                non_blocking=True,
            )
        sequence_lengths = decode_meta.seq_lens[:num_tokens]
        # A negative tail-count sentinel tells the fused kernel that
        # ``group_counts`` is the raw dense token length. Passing seq_lens
        # directly avoids three tiny metadata kernels in every attention layer.
        group_sentinel, tail_starts, tail_counts, query_start_loc = self._get_decode_constant_buffers(
            query.device,
            num_tokens,
        )
        if self.glm_indexer is not None:
            group_sentinel, sequence_lengths, tail_starts, tail_counts = self._get_kpool_qsa_plan(
                decode_meta.input_positions, 0, num_tokens
            )

        op = self._get_paged_latent_op()
        if op is None:
            raise RuntimeError("vLLM Ascend was built without the native 310P paged-latent attention operator.")
        block_table = _qsa_cache_block_table(
            decode_meta.block_table[: attn_metadata.num_decodes].to(torch.int32),
            key_cache.shape[2],
        )
        if num_tokens != attn_metadata.num_decodes:
            # QSA already maps each query row to a request using device query
            # boundaries. GLM's per-row pool/tail plan supplies the causal
            # extent, so verification tokens share pages without seeing later
            # draft tokens. Keep boundaries on device for graph replay.
            query_start_loc = (
                attn_metadata.query_start_loc[: attn_metadata.num_decodes + 1].to(torch.int32).contiguous()
            )
        if self.host_kv_layer is not None:
            if self.glm_indexer is None:
                raise RuntimeError("GLM host MLA requires the kpool indexer")
            block_table = self.host_kv_layer.stage(
                block_table,
                group_sentinel,
                sequence_lengths,
                tail_starts,
                tail_counts,
                _QSA_COMPRESS_RATIO,
            )
        if key_cache.shape != value_cache.shape:
            raise ValueError("QSA key and value caches must have the same logical shape")
        logical_channels = key_cache.shape[1]
        head_dim_blocks = query.shape[2] // _NZ_INNER
        if head_dim_blocks == 0 or query.shape[2] % _NZ_INNER or logical_channels % head_dim_blocks:
            raise ValueError("QSA logical cache channels must contain whole KV heads")
        logical_kv_heads = logical_channels // head_dim_blocks
        physical_key_cache = _qsa_physical_cache_page(key_cache)
        physical_value_cache = _qsa_physical_cache_page(value_cache)
        if physical_key_cache.shape != physical_value_cache.shape:
            raise ValueError("QSA key and value physical pages must have the same shape")
        latent_output = op(
            query.contiguous(),
            physical_key_cache,
            physical_value_cache,
            group_sentinel,
            sequence_lengths,
            tail_starts,
            tail_counts,
            block_table,
            query_start_loc,
            self.scale,
            _QSA_COMPRESS_RATIO,
            logical_kv_heads,
        )
        return self._v_up_proj(latent_output.transpose(0, 1).contiguous())

    @staticmethod
    def _get_paged_latent_op():
        op_namespace = getattr(torch.ops, "_C_ascend", None)
        return None if op_namespace is None else getattr(op_namespace, "npu_qsa_sparse_attention_310", None)

    def _forward_decode(
        self,
        decode_preprocess_res: DecodeMLAPreprocessResult,
        block_size: int,
        attn_metadata: AscendMLAMetadata,
    ) -> torch.Tensor:
        """Use fused 310P paged-latent attention with the shared MLA interface."""
        del block_size
        q_nope = decode_preprocess_res.ql_nope
        k_nope = decode_preprocess_res.k_nope
        k_pe = decode_preprocess_res.k_pe
        if q_nope is None or k_nope is None or k_pe is None:
            raise ValueError("310P MLA decode requires query and latent KV cache tensors")
        return self._forward_decode_fused(
            q_nope,
            k_nope,
            k_pe,
            attn_metadata,
        )


class AscendMLABackend310(AscendMLABackend):
    """310P MLA backend selector.

    Uses the standard MLA metadata builder with a 310P-native NZ latent cache,
    flash-attention prefill, and fused paged-latent decode implementation.
    """

    @staticmethod
    def get_name() -> str:
        return "ASCEND_MLA_310P"

    @staticmethod
    def get_impl_cls() -> type[AscendMLAImpl310]:
        return AscendMLAImpl310

    @staticmethod
    def get_builder_cls() -> type[AscendMLAMetadataBuilder310]:
        return AscendMLAMetadataBuilder310

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
    ) -> tuple[int, ...]:
        """Return one latent cache in the 310P NZ page geometry."""
        del cache_type
        if head_size % _NZ_INNER:
            raise ValueError(f"310P native MLA head size must be divisible by {_NZ_INNER}, got {head_size}.")
        return (
            num_blocks,
            num_kv_heads * head_size // _NZ_INNER,
            block_size,
            _NZ_INNER,
        )

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # The 512-wide latent head forces block_size * head_size <= 16384, i.e.
        # block_size <= 32 (see _ASCEND_310P_MAX_BLOCK_TIMES_HEAD). Prefer 32,
        # allow 16 as a fallback if the scheduler needs a smaller page.
        return [32, 16]
