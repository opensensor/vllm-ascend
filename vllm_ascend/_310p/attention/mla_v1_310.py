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

Status: BRING-UP / UNVERIFIED ON HARDWARE. Only reachable when
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
* Attention core: ``_forward_prefill`` / ``_forward_decode`` call
  ``npu_fused_infer_attention_score`` (FIA) / ``..._v2``. FIA is present in the
  deployed torch_npu 2.13.0rc1 Python API, but its aclnn kernel has NOT been
  validated on ascend310p1. This is the single biggest runtime unknown.

Target model
------------
GLM-5.3-Flash-W2-310p (``Glm5NextForConditionalGeneration``) is NoPE MLA:
``qk_rope_head_dim == 0``, ``mla_use_nope == True``, ``kv_lora_rank == 512``,
``qk_nope_head_dim == v_head_dim == 256``, 64 heads. Only 11 of 45 layers are
attention (``deepseek_sparse_attention``); the other 34 are ``linear_attention``
(KDA/GDN, already working on 310P). The sparse indexer runs in ``full`` mode
(``indexer_types`` all ``"full"``) so no top-k sparsification is required and the
attention layers behave as dense MLA.

NoPE removes the decoupled-rope machinery (``npu_kv_rmsnorm_rope_cache`` rope
part, ``npu_interleave_rope``) from the critical path, which is what makes a
310P port tractable without new rope-cache kernels.
"""

from __future__ import annotations

import torch

from vllm_ascend.attention.mla_v1 import (
    AscendMLABackend,
    AscendMLAImpl,
)

# 310P kernel tiling constraint (see commit e8f7b2e3f, "support attn_head_size
# larger than 128"): block_size * head_size must not exceed 16384 elements for
# the paged/flash attention operators. The MLA latent cache head size is
# kv_lora_rank + qk_rope_head_dim (512 + 0 = 512 for GLM NoPE), so the block
# size must satisfy block_size <= 16384 / 512 = 32.
_ASCEND_310P_MAX_BLOCK_TIMES_HEAD = 16384


class AscendMLAImpl310(AscendMLAImpl):
    """310P MLA attention implementation.

    Reuses the decomposed NoPE front-end from :class:`AscendMLAImpl` and hardens
    the entry points so a run can never silently fall into a 310P-unsupported op
    path. The attention core is inherited (FIA) by default; a fully decomposed
    "naive" core that only uses 310P-verified ops (``_npu_flash_attention`` for
    prefill, paged attention over the latent for decode) is specified below and
    guarded so it fails loudly with an exact contract instead of producing
    silently wrong results.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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

    # ---------------------------------------------------------------------
    # Naive attention core (spec only). See module docstring for why this is
    # not enabled by default. These are the exact kernels/contracts the main
    # session must validate or implement on hardware.
    # ---------------------------------------------------------------------
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
        """Materialized (non-absorbed) MLA prefill using 310P dense ops.

        Contract for the hardware implementation:

        * Inputs are already produced by ``mla_preprocess_prefill`` (the
          decomposed front-end): ``q_nope``/``k_nope`` have head dim
          ``qk_nope_head_dim`` (256), ``value`` has head dim ``v_head_dim``
          (256), and for NoPE ``q_pe``/``k_pe`` are empty. Q, K and V therefore
          share head_size == 256, which is a standard (non-latent) MHA shape.
        * This is expressible with ``torch_npu._npu_flash_attention`` exactly as
          the dense 310P backend uses it in
          ``_310p/attention/attention_v1.py::forward_prefill_310`` (num_heads ==
          num_kv_heads == self.num_heads, causal mask, TND/packed layout).
          head_size 256 > 128 is covered by commit e8f7b2e3f as long as the
          prefill flash op accepts it (VERIFY: the e8f7b2e3f relaxation was
          demonstrated for the KV-cache/paged path; confirm the non-paged
          ``_npu_flash_attention`` accepts head_size 256 on ascend310p1).
        * ``_compute_prefill_context`` (chunked-prefill context accumulation in
          the parent) also calls FIA; on 310P chunked prefill must instead reuse
          the paged latent decode kernel below, or be disabled by forcing
          PrefillNoCache-only scheduling for the MLA layers during bring-up.

        Left unimplemented on purpose: wiring the MLA metadata builder to emit
        the packed seq_len/mask layout ``_npu_flash_attention`` expects requires
        NPU iteration and is deferred to the hardware session.
        """
        raise NotImplementedError(
            "AscendMLAImpl310._forward_prefill_naive is a hardware-validation "
            "stub. Use the inherited FIA path first; if FIA is unsupported on "
            "ascend310p1, implement this with torch_npu._npu_flash_attention "
            "(head_size 256, num_kv_heads == num_heads) per the docstring."
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
        """Weight-absorbed MLA decode over the latent, using paged attention.

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

        Left unimplemented on purpose: same reason as prefill.
        """
        raise NotImplementedError(
            "AscendMLAImpl310._forward_decode_naive is a hardware-validation "
            "stub. Use the inherited FIA-v2 path first; if unsupported on "
            "ascend310p1, implement paged latent MQA (head_size 512, "
            "num_kv_heads 1, block_size <= 32) per the docstring."
        )


class AscendMLABackend310(AscendMLABackend):
    """310P MLA backend selector.

    Inherits the ND latent KV-cache shape and metadata builder from the standard
    MLA backend (the decomposed front-end and the FIA core both use that ND
    layout via ``cache_mode="PA_BSND"``), and swaps in the 310P implementation.
    """

    @staticmethod
    def get_name() -> str:
        return "ASCEND_MLA_310P"

    @staticmethod
    def get_impl_cls() -> type[AscendMLAImpl310]:
        return AscendMLAImpl310

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # The 512-wide latent head forces block_size * head_size <= 16384, i.e.
        # block_size <= 32 (see _ASCEND_310P_MAX_BLOCK_TIMES_HEAD). Prefer 32,
        # allow 16 as a fallback if the scheduler needs a smaller page.
        return [32, 16]
