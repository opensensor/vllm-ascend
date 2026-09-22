# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stateful AscendC KDA execution for the GLM W2 adapter on 310P.

The generic W2 KDA oracle is intentionally plain PyTorch, but using it for
serving restarts the recurrence at every forward and executes one Python loop
iteration per token.  The 310P extension already ships the two exact operators
needed by GLM's KDA variant:

* ``chunk_kda_fwd`` for arbitrarily long, variable-length prefills; and
* ``npu_recurrent_gated_delta_rule_310`` for decode/spec steps of up to eight
  tokens per request.  Its ``gk`` input is the per-key-channel log decay used
  by KDA (as distinct from GDN's per-head scalar ``g`` input).

This module owns only execution.  Projection and output-normalization weights
remain on the checkpoint-compatible shipped GLM attention module.
"""

from __future__ import annotations

from typing import Any

import torch

KDA_CHUNK_SIZE = 64


def _safe_gate(
    raw_gate: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
) -> torch.Tensor:
    """Compute GLM's per-channel bounded log-decay in fp32."""
    gate = raw_gate.float()
    num_heads = a_log.numel()
    gate = gate + dt_bias.float().reshape(1, 1, num_heads, gate.shape[-1])
    scale = torch.exp(a_log.float().reshape(1, 1, num_heads, 1))
    return lower_bound * torch.sigmoid(scale * gate)


def _safe_gate_for_layer(self_attn: Any, raw_gate: torch.Tensor) -> torch.Tensor:
    """Safe gate with the weight-only exponential cached after loading."""
    cached = getattr(self_attn, "_kda_safe_gate_cache", None)
    if cached is None or cached[0] is not self_attn.A_log or cached[1] is not self_attn.dt_bias:
        scale = torch.exp(self_attn.A_log.float()).reshape(
            1,
            1,
            self_attn.A_log.numel(),
            1,
        )
        bias = self_attn.dt_bias.float().reshape(
            1,
            1,
            self_attn.A_log.numel(),
            raw_gate.shape[-1],
        )
        cached = (self_attn.A_log, self_attn.dt_bias, scale, bias)
        self_attn._kda_safe_gate_cache = cached
    return float(self_attn.kda_lower_bound) * torch.sigmoid(cached[2] * (raw_gate.float() + cached[3]))


def _l2norm_310p(x: torch.Tensor) -> torch.Tensor:
    # Lazy import keeps the model package importable on CPU development hosts.
    from vllm_ascend._310p.ops.fla.l2norm import l2norm_310p

    return l2norm_310p(x.contiguous())


def _actual_lengths(cu_seqlens: torch.Tensor, num_sequences: int) -> torch.Tensor:
    return (cu_seqlens[1 : num_sequences + 1] - cu_seqlens[:num_sequences]).to(torch.int32).contiguous()


def _flatten_spec_state_indices(
    state_indices: torch.Tensor,
    actual_lengths: torch.Tensor,
    total_tokens: int,
) -> torch.Tensor:
    """Flatten per-request spec slots without a device-to-host synchronization."""
    if state_indices.ndim == 1:
        return state_indices[:total_tokens].to(torch.int32).contiguous()
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX

    # FULL-graph spec decode uses a fixed number of columns per request. Dummy
    # requests have zero actual length, so the kernel ignores their padded slot
    # ids; reshape avoids MaskedSelect, which invalidates 310P stream capture.
    if _EXTRA_CTX.capturing is True:
        return state_indices.reshape(-1)[:total_tokens].to(torch.int32).contiguous()
    max_step = state_indices.shape[1]
    positions = torch.arange(max_step, device=state_indices.device)
    valid = positions.unsqueeze(0) < actual_lengths.unsqueeze(1)
    return state_indices.masked_select(valid).to(torch.int32).contiguous()[:total_tokens]


def _run_recurrent(
    self_attn: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta_raw: torch.Tensor,
    recurrent_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    num_sequences: int,
    num_accepted_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the in-place 310P recurrent kernel against the paged state pool."""
    q = _l2norm_310p(q).squeeze(0).to(torch.float16).contiguous()
    k = _l2norm_310p(k).squeeze(0).to(torch.float16).contiguous()
    v = v.squeeze(0).to(torch.float16).contiguous()
    gk = _safe_gate_for_layer(self_attn, raw_gate).squeeze(0).contiguous()
    beta = beta_raw.float().sigmoid().squeeze(0).to(torch.float16).contiguous()
    actual_lengths = _actual_lengths(cu_seqlens, num_sequences)
    flat_state_indices = _flatten_spec_state_indices(
        state_indices[:num_sequences],
        actual_lengths,
        v.shape[0],
    )
    accepted = None
    if num_accepted_tokens is not None:
        accepted = num_accepted_tokens[:num_sequences].to(torch.int32).contiguous()
        accepted = torch.where(actual_lengths > 0, accepted, torch.zeros_like(accepted)).contiguous()

    return torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310(
        query=q,
        key=k,
        value=v,
        g=None,
        gk=gk,
        beta=beta,
        state=recurrent_state,
        actual_seq_lengths=actual_lengths,
        ssm_state_indices=flat_state_indices,
        num_accepted_tokens=accepted,
        scale_value=self_attn.head_dim**-0.5,
    ).unsqueeze(0)


def _run_prefill(
    self_attn: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta_raw: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    chunk_metadata: Any,
) -> torch.Tensor:
    """Run exact chunked KDA and write its final state back to the cache."""
    cu_seqlens = (
        chunk_metadata.cu_seqlens_host if chunk_metadata.cu_seqlens_kern is None else chunk_metadata.cu_seqlens_kern
    )
    keep = chunk_metadata.keep_meta
    if keep is not None:
        state_indices = state_indices[keep]
        has_initial_state = has_initial_state[keep]

    # chunk_kda_fwd accumulates its carry in fp32.  The persistent 310P decode
    # pool is fp16 because the recurrent kernel requires fp16, so convert only
    # the small set of active request states at a prefill boundary.
    initial_state = recurrent_state[state_indices].float().contiguous()
    initial_state[~has_initial_state] = 0

    result = torch.ops._C_ascend.chunk_kda_fwd(
        _l2norm_310p(q).contiguous(),
        _l2norm_310p(k).contiguous(),
        v.contiguous(),
        raw_gate.float().contiguous(),
        beta_raw.float().sigmoid().contiguous(),
        self_attn.head_dim**-0.5,
        KDA_CHUNK_SIZE,
        layout="BSND",
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_metadata.chunk_indices_chunk64_host,
        safe_gate=True,
        lower_bound=float(self_attn.kda_lower_bound),
        use_gate_in_kernel=True,
        A_log=self_attn.A_log.reshape(-1).float().contiguous(),
        dt_bias=self_attn.dt_bias.float().contiguous(),
        disable_recompute=False,
        return_intermediate_states=False,
        state_v_first=True,
    )
    recurrent_state[state_indices] = result[1].to(recurrent_state.dtype)
    return result[0]


def _split_qkv(self_attn: Any, mixed_qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    parts = mixed_qkv.split(self_attn.local_projection_size, dim=-1)
    return tuple(part.reshape(1, -1, self_attn.local_num_heads, self_attn.head_dim) for part in parts)  # type: ignore[return-value]


def run_stateful_kda_310(
    self_attn: Any,
    mixed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    beta_raw: torch.Tensor,
    conv_weight_t: torch.Tensor,
) -> torch.Tensor:
    """Execute conv + KDA while maintaining vLLM's paged per-request state."""
    from vllm.forward_context import get_forward_context
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    forward_context = get_forward_context()
    metadata_by_layer = forward_context.attn_metadata
    if metadata_by_layer is None:
        return torch.zeros(
            (1, mixed_qkv.shape[0], self_attn.local_num_heads, self_attn.head_dim),
            dtype=mixed_qkv.dtype,
            device=mixed_qkv.device,
        )
    metadata = metadata_by_layer.get(self_attn.prefix)
    if metadata is None:
        return torch.zeros(
            (1, mixed_qkv.shape[0], self_attn.local_num_heads, self_attn.head_dim),
            dtype=mixed_qkv.dtype,
            device=mixed_qkv.device,
        )
    num_actual_tokens = metadata.num_actual_tokens
    mixed_qkv = mixed_qkv[:num_actual_tokens]
    raw_gate = raw_gate[:, :num_actual_tokens]
    beta_raw = beta_raw[:, :num_actual_tokens]

    conv_state, recurrent_state = self_attn.kv_cache
    if recurrent_state.dtype != torch.float16:
        raise RuntimeError("GLM 310P KDA recurrent cache must be float16 for npu_recurrent_gated_delta_rule_310")
    conv_bias = getattr(self_attn.q_conv1d, "bias", None)

    spec_indices = metadata.spec_token_indx
    non_spec_indices = metadata.non_spec_token_indx
    has_spec = metadata.spec_sequence_masks is not None
    if has_spec:
        if metadata.num_prefills == 0 and metadata.num_decodes == 0:
            mixed_spec, gate_spec, beta_spec = mixed_qkv, raw_gate, beta_raw
            mixed_non_spec = gate_non_spec = beta_non_spec = None
        else:
            mixed_spec = mixed_qkv.index_select(0, spec_indices)
            gate_spec = raw_gate.index_select(1, spec_indices)
            beta_spec = beta_raw.index_select(1, spec_indices)
            mixed_non_spec = mixed_qkv.index_select(0, non_spec_indices)
            gate_non_spec = raw_gate.index_select(1, non_spec_indices)
            beta_non_spec = beta_raw.index_select(1, non_spec_indices)
    else:
        mixed_spec = gate_spec = beta_spec = None
        mixed_non_spec, gate_non_spec, beta_non_spec = mixed_qkv, raw_gate, beta_raw

    core_spec = None
    if mixed_spec is not None:
        spec_meta = metadata.spec_decode_metadata.spec_causal_conv1d
        mixed_spec = torch.ops._C_ascend.npu_causal_conv1d_310(
            mixed_spec,
            conv_weight_t,
            bias=conv_bias,
            conv_states=conv_state,
            query_start_loc=spec_meta.query_start_loc,
            cache_indices=spec_meta.cache_indices,
            initial_state_mode=None,
            num_accepted_tokens=spec_meta.num_accepted_tokens,
            activation_mode=1,
            pad_slot_id=PAD_SLOT_ID,
            run_mode=1,
        )
        q_spec, k_spec, v_spec = _split_qkv(self_attn, mixed_spec)
        core_spec = _run_recurrent(
            self_attn,
            q_spec,
            k_spec,
            v_spec,
            gate_spec,
            beta_spec,
            recurrent_state,
            metadata.spec_query_start_loc,
            metadata.spec_state_indices_tensor,
            num_sequences=metadata.num_spec_decodes,
            num_accepted_tokens=spec_meta.num_accepted_tokens,
        )

    core_non_spec = None
    if mixed_non_spec is not None and mixed_non_spec.shape[0] > 0:
        if metadata.num_prefills > 0:
            conv_meta = metadata.non_spec_prefill_metadata.causal_conv1d
            run_mode = 0
        elif metadata.num_decodes > 0:
            conv_meta = metadata.non_spec_decode_metadata.causal_conv1d
            run_mode = 1
        else:
            conv_meta = None
            run_mode = 1
        if conv_meta is not None:
            mixed_non_spec = torch.ops._C_ascend.npu_causal_conv1d_310(
                mixed_non_spec,
                conv_weight_t,
                bias=conv_bias,
                conv_states=conv_state,
                query_start_loc=conv_meta.query_start_loc,
                cache_indices=conv_meta.cache_indices,
                initial_state_mode=conv_meta.initial_state_mode,
                num_accepted_tokens=None,
                activation_mode=1,
                pad_slot_id=PAD_SLOT_ID,
                run_mode=run_mode,
            )
        q_non_spec, k_non_spec, v_non_spec = _split_qkv(self_attn, mixed_non_spec)

        num_decode_tokens = metadata.num_decode_tokens
        core_decode = None
        if metadata.num_decodes > 0:
            core_decode = _run_recurrent(
                self_attn,
                q_non_spec[:, :num_decode_tokens],
                k_non_spec[:, :num_decode_tokens],
                v_non_spec[:, :num_decode_tokens],
                gate_non_spec[:, :num_decode_tokens],
                beta_non_spec[:, :num_decode_tokens],
                recurrent_state,
                metadata.non_spec_query_start_loc[: metadata.num_decodes + 1],
                metadata.non_spec_state_indices_tensor[: metadata.num_decodes],
                num_sequences=metadata.num_decodes,
            )

        core_prefill = None
        if metadata.num_prefills > 0:
            q_prefill = q_non_spec[:, num_decode_tokens:]
            k_prefill = k_non_spec[:, num_decode_tokens:]
            v_prefill = v_non_spec[:, num_decode_tokens:]
            gate_prefill = gate_non_spec[:, num_decode_tokens:]
            beta_prefill = beta_non_spec[:, num_decode_tokens:]
            core_prefill = _run_prefill(
                self_attn,
                q_prefill,
                k_prefill,
                v_prefill,
                gate_prefill,
                beta_prefill,
                recurrent_state,
                metadata.prefill_state_indices,
                metadata.prefill_has_initial_state,
                metadata.non_spec_prefill_metadata.chunk,
            )
        if core_decode is not None and core_prefill is not None:
            core_non_spec = torch.cat((core_decode, core_prefill), dim=1)
        else:
            core_non_spec = core_decode if core_decode is not None else core_prefill

    if core_spec is not None and core_non_spec is not None:
        result = torch.zeros(
            (1, num_actual_tokens, self_attn.local_num_heads, self_attn.head_dim),
            dtype=mixed_qkv.dtype,
            device=mixed_qkv.device,
        )
        result.index_copy_(1, spec_indices, core_spec)
        result.index_copy_(1, non_spec_indices, core_non_spec)
        return result
    if core_spec is not None:
        return core_spec
    if core_non_spec is not None:
        return core_non_spec
    return torch.zeros(
        (1, num_actual_tokens, self_attn.local_num_heads, self_attn.head_dim),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
