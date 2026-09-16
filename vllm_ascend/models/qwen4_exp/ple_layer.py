# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend Qwen4Exp PLE (parallel layer embedding) injection layer (plan T4.3).

Ports the host-side compute path of the CUDA fork's ``Qwen4ExpPLELayer``
(``vllm/models/qwen4_exp/nvidia/ple_layer.py``) to the Triton-free Ascend 310P
dev path: n-gram row gather (through the T4.1 host PLE table method), the merged
key/value projection (``ple_projection`` dtype), the sigmoid gate, and the
dilated causal short convolution combined with the outer residual.

Compared with the fork the device-only concerns are intentionally left out here:
the stateful decode/spec KV-cache short-conv routing lives in the model runner,
and this layer runs the stateless full-sequence convolution used by the CPU
eager path. Every dtype is read from the authoritative policy
(:data:`ASCEND_QWEN4EXP_DTYPE_POLICY`); no dtype literal is spelled in this file.

Row gather is delegated to the T4.1 :class:`AscendPLEEmbeddingMethod` interface
(``gather_rows`` / ``dequantize``), so both host transports -- (a) pinned-UVA and
(b) ``/dev/shm`` shared-mmap -- are usable without change. The gate + short-conv
math lives in :mod:`vllm_ascend.models.qwen4_exp.ops.ple` and is parity-checked
against the T0.6 ``ple_reference``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from .dtype_policy import ASCEND_QWEN4EXP_DTYPE_POLICY, Qwen4ExpDtypePolicy
from .ops.ple import ple_gate, ple_short_conv

if TYPE_CHECKING:
    from .ngram_embedding import AscendPLEEmbeddingMethod

# The PLE short convolution uses the SiLU activation, matching the fork layer.
_PLE_CONV_ACTIVATION = "silu"


class AscendQwen4ExpPLELayer(nn.Module):
    """PLE injection layer: gather -> project -> gate -> short-conv combine.

    The merged key/value projection runs in ``policy.ple_projection_dtype`` and
    the grouped RMSNorm / gate / convolution reductions accumulate in
    ``policy.ple_norm_accumulation_dtype`` before rounding back, exactly as the
    dtype policy pins for the 310P.

    Args:
        config: the Qwen4Exp text config (needs ``hidden_size``, ``hc_count``,
            ``ple_embed_dim``, ``ple_conv_kernel_size``, ``ngram_size``,
            ``heads_per_ngram``, ``rms_norm_eps``).
        layer_idx: decoder layer index (for identification only).
        ple_method: the T4.1 host PLE table method providing ``gather_rows`` /
            ``dequantize``. Optional at construction; required for ``forward``.
        dtype_policy: authoritative dtype policy.
        params_dtype: override for the projection / norm / conv parameter dtype.
            Defaults to ``policy.ple_projection_dtype``. The CPU parity harness
            passes ``torch.float64`` to isolate the math from fp16 rounding.
        prefix: module prefix (for identification only).
    """

    def __init__(
        self,
        *,
        config: object,
        layer_idx: int,
        ple_method: AscendPLEEmbeddingMethod | None = None,
        dtype_policy: Qwen4ExpDtypePolicy = ASCEND_QWEN4EXP_DTYPE_POLICY,
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.prefix = prefix
        self.dtype_policy = dtype_policy
        self.projection_dtype = dtype_policy.cast_site("ple_projection")
        self.norm_accumulation_dtype = dtype_policy.cast_site("ple_norm_accumulation")
        self.params_dtype = params_dtype if params_dtype is not None else self.projection_dtype
        self.ple_method = ple_method

        self.hidden_size = int(config.hidden_size)
        self.hc_count = int(config.hc_count)
        if self.hc_count <= 1:
            raise ValueError(f"Qwen4Exp requires hc_count > 1, got {self.hc_count}")
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.ple_embed_dim = int(config.ple_embed_dim)
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        # The short-conv dilation is the n-gram size (fork parity).
        self.short_conv_dilation = int(config.ngram_size)
        self.conv_state_len = (self.conv_kernel_size - 1) * self.short_conv_dilation
        self.heads_per_ngram = int(config.heads_per_ngram)
        # Total n-gram heads: (ngram_size - 1) predecessor orders * heads each.
        self.num_ngram_heads = max(self.short_conv_dilation - 1, 0) * self.heads_per_ngram
        if self.num_ngram_heads <= 0:
            raise ValueError("PLE requires at least one n-gram head (ngram_size >= 2)")
        if self.ple_embed_dim % self.num_ngram_heads:
            raise ValueError(
                f"ple_embed_dim ({self.ple_embed_dim}) must be divisible by the "
                f"total n-gram heads ({self.num_ngram_heads})"
            )
        self.per_head_dim = self.ple_embed_dim // self.num_ngram_heads
        self.rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.activation = _PLE_CONV_ACTIVATION

        # Merged key/value projection: ple_embed_dim -> [hc_hidden, hidden].
        # No TP sharding here (the PLE cache is TP-replicated in the fork), and
        # no bias, matching MergedColumnParallelLinear(bias=False, disable_tp).
        self.output_sizes = [self.hc_hidden_size, self.hidden_size]
        self.kv_proj_weight = nn.Parameter(
            torch.zeros(sum(self.output_sizes), self.ple_embed_dim, dtype=self.params_dtype),
            requires_grad=False,
        )
        # Grouped RMSNorm weights are stored as the additive term ``w`` so the
        # norm applies ``(1 + w)`` (fork zeros-init -> identity scale).
        norm_shape = (self.hc_hidden_size,)
        self.norm_key_weight = nn.Parameter(torch.zeros(norm_shape, dtype=self.params_dtype), requires_grad=False)
        self.norm_query_weight = nn.Parameter(torch.zeros(norm_shape, dtype=self.params_dtype), requires_grad=False)
        self.norm_conv_weight = nn.Parameter(torch.zeros(norm_shape, dtype=self.params_dtype), requires_grad=False)
        # Depthwise short-conv filters [C, K] (fork zeros-init).
        self.conv_weight = nn.Parameter(
            torch.zeros(self.hc_hidden_size, self.conv_kernel_size, dtype=self.params_dtype),
            requires_grad=False,
        )

    # -- gather ------------------------------------------------------------- #

    def gather_embeddings(self, ngram_ids: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
        """Batched PLE row gather -> ``[T, ple_embed_dim]`` embeddings.

        ``ngram_ids`` is ``[T, num_ngram_heads]`` of global table row indices.
        Every row is looked up in one batched call through the T4.1 host method
        (no per-row sync), reshaped so head ``h`` of token ``t`` fills columns
        ``[h*per_head_dim, (h+1)*per_head_dim)``, then dequantized to
        ``output_dtype``.
        """
        if self.ple_method is None:
            raise RuntimeError("AscendQwen4ExpPLELayer.forward requires a PLE embedding method (T4.1)")
        if ngram_ids.ndim != 2:
            raise ValueError("ngram_ids must be [T, num_ngram_heads]")
        num_tokens, heads = ngram_ids.shape
        if heads != self.num_ngram_heads:
            raise ValueError(f"ngram_ids has {heads} heads, expected {self.num_ngram_heads}")
        # One batched gather over all (token, head) rows -- batched lookup only.
        rows = self.ple_method.gather_rows(ngram_ids.reshape(-1))
        per_head_dim = rows.shape[-1]
        if per_head_dim != self.per_head_dim:
            raise ValueError(
                f"PLE table row dim ({per_head_dim}) * n-gram heads ({heads}) != ple_embed_dim ({self.ple_embed_dim})"
            )
        embeddings = rows.reshape(num_tokens, heads * per_head_dim)
        return self.ple_method.dequantize(embeddings, output_dtype)

    # -- projection --------------------------------------------------------- #

    def project(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Merged key/value projection, split into ``(key, value)``."""
        weight = self.kv_proj_weight.to(embeddings.dtype)
        kv = torch.matmul(embeddings, weight.t())
        key, value = kv.split(self.output_sizes, dim=-1)
        return key, value

    # -- forward ------------------------------------------------------------ #

    def forward(self, hidden_states: torch.Tensor, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Run the PLE injection for one decoder layer.

        Args:
            hidden_states: ``[T, hc_hidden]`` hc-expanded hidden state (the gate
                query and the outer residual).
            ngram_ids: ``[T, num_ngram_heads]`` global PLE table row indices.

        Returns:
            ``[T, hc_hidden]`` PLE output (gated + short-conv + outer residual).
        """
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must be [T, hc_hidden]")
        if hidden_states.shape[-1] != self.hc_hidden_size:
            raise ValueError(f"hidden_states last dim ({hidden_states.shape[-1]}) != hc_hidden ({self.hc_hidden_size})")
        if ngram_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "PLE expects ngram_ids and hidden_states to share the token "
                f"dimension, got {ngram_ids.shape[0]} and {hidden_states.shape[0]}"
            )
        embeddings = self.gather_embeddings(ngram_ids, hidden_states.dtype)
        key, value = self.project(embeddings)
        gated, conv_input = ple_gate(
            key,
            value,
            hidden_states,
            self.norm_key_weight,
            self.norm_query_weight,
            self.norm_conv_weight,
            self.rms_norm_eps,
            accum_dtype=self.norm_accumulation_dtype,
        )
        output = ple_short_conv(
            conv_input,
            gated,
            hidden_states,
            self.conv_weight,
            self.short_conv_dilation,
            activation=self.activation,
            accum_dtype=self.norm_accumulation_dtype,
        )
        return output.to(hidden_states.dtype)


__all__ = ["AscendQwen4ExpPLELayer"]
