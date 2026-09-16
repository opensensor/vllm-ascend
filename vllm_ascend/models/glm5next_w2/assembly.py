# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-runnable eager assembly for the Ascend 310P GLM-5.3-Flash W2 path (G7).

The shipped, config-driven ``glm5next`` base that
``vllm_ascend.models.glm5next_w2.model.AscendGlm5NextW2ForCausalLM`` subclasses is
``torch_npu`` / Triton bound (``FusedMoEFactory`` + the Triton KDA op) and is
**not importable on the CPU host** (no NPU, no Triton on the 310P dev lane). So --
exactly like the DeepSeek V4.1 E4.1 assembly
(``vllm_ascend/models/deepseek_v41/assembly.py``) and the Qwen4Exp T1.5 assembly --
this module builds a *parallel, host-runnable eager assembly* that ties the G4/G5/G6
component modules into GLM's **hybrid** decoder stack (KDA linear-attention layers +
DSA sparse-attention layers) which CONSTRUCTS and FORWARDS on CPU with dummy weights,
so the whole graph is validated host-side without the NPU.

What is REAL (the G-wave components, wired as ``model.py``'s hooks would)
------------------------------------------------------------------------
* **KDA** (G4 ``kda.py``): ``Glm5NextW2KDA`` -- the Triton-free gated-delta linear
  attention core, on the ``kda`` layers. The assembly owns the surrounding
  q|k|v / gate / o projections the shipped ``Glm5NextLinearAttention`` carries.
* **DSA** (G5 ``dsa.py``): ``AscendGlm5NextW2DSA.from_config`` -- the MLA-NoPE
  sparse attention (DeepSeek-indexer top-k selection + restricted attention), on
  the ``full_attn_layers``. Self-contained (owns its projections).
* **W2 MoE** (G6 ``moe.py``): ``Glm5NextW2MoE`` -- GLM sigmoid ``noaux_tc`` top-k
  router -> E1.3 ``AscendW2DynamicFusedMoEMethod310`` (resolved lazily) -> eager
  ``routed * routed_scaling_factor + shared`` combine, on the layers past
  ``first_k_dense_replace``. Dense layers use a plain FP16 SwiGLU MLP.
* **Weight map** (G6 ``weight_mapping.py``): ``map_expert_tensor`` fills the E1.3
  fused param layout in :meth:`load_weights`; FP16 by name; rejection classes
  fire on a malformed artifact.

Pure-eager stand-ins (no dedicated Ascend op exists yet on the CPU lane), tagged
-------------------------------------------------------------------------------
* **multi-head hyper-connection (mhc)** -- the shipped model carries the residual
  as ``mhc_num_residual_streams`` parallel streams mixed by device sinkhorn
  kernels (device-only). Here (as in the DeepSeek assembly) the streams are mixed
  by a plain mean and the block output is injected back into every stream. The
  REAL host mhc ops (G6 ``hc_pre`` / ``hc_post``) are unit-tested in
  ``test_glm5next_w2_moe.py`` and are the device path; this boot exercises the
  attention/MoE wiring, not the sinkhorn mixing.
* **MTP-1** -- GLM ships a single next-token predict layer. Here it is a light
  extra norm + head over the backbone hidden (``mtp_draft``); the MTP MoE layer's
  experts are covered by ``load_weights`` (``moe_block_ids`` includes the MTP
  layer index).

This class is NOT registered as an architecture; it is the CPU twin used only to
validate the assembled graph host-side.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the DeepSeek eager helpers + dummy W2 bank builder verbatim (host-clean).
from vllm_ascend.models.deepseek_v41.assembly import (
    W2Expert,
    _linear,
    _rms_norm,
    _seeded,
    build_dummy_w2_expert_bank,
)

from .dsa import AscendGlm5NextW2DSA
from .dtype_policy import (
    ASCEND_GLM5NEXT_W2_DTYPE_POLICY,
    Glm5NextW2DtypePolicy,
)
from .kda import Glm5NextW2KDA
from .moe import GLM5NEXT_MHC_NUM_RESIDUAL_STREAMS, Glm5NextW2MoE
from .weight_mapping import (
    WeightClass,
    classify_tensor,
    expected_expert_shape,
    map_expert_tensor,
)

__all__ = [
    "AscendGlm5NextW2EagerDecoderLayer",
    "AscendGlm5NextW2EagerModel",
    "AscendGlm5NextW2EagerForCausalLM",
]


def _is_dsa_layer(config: Any, layer_idx: int) -> bool:
    """A layer is DSA (full attention) iff it is in ``full_attn_layers``."""
    full = getattr(config, "full_attn_layers", None)
    if full is not None:
        return layer_idx in set(full)
    # Fallback to the GLM pattern: every 4th layer (index 3, 7, ...) is DSA.
    return (layer_idx % 4) == 3


# ===========================================================================
# KDA attention block: assembly-owned projections around the G4 KDA core
# ===========================================================================
class _KdaAttnBlock(nn.Module):
    """Wraps the G4 :class:`Glm5NextW2KDA` core with its q|k|v / gate / o params.

    The shipped ``Glm5NextLinearAttention`` owns these projections; here they are
    seeded dummy params so the boot exercises the real KDA recurrence end-to-end.
    """

    def __init__(self, *, hidden: int, num_heads: int, head_dim: int, conv_width: int,
                 layer_idx: int, dtype_policy: Glm5NextW2DtypePolicy) -> None:
        super().__init__()
        self.dtype_policy = dtype_policy
        self.main_dtype = dtype_policy.main_dtype
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim
        s = 3000 + 17 * layer_idx
        # q|k|v fused projection + depthwise conv weight (3*inner channels).
        self.qkv_proj = nn.Parameter(_seeded((3 * inner, hidden), s + 1, 0.05, self.main_dtype))
        self.conv_weight = nn.Parameter(_seeded((3 * inner, conv_width), s + 2, 0.05, self.main_dtype))
        # gate (raw_g), beta, output-gate (g_out) projections.
        self.raw_g_proj = nn.Parameter(_seeded((inner, hidden), s + 3, 0.05, self.main_dtype))
        self.beta_proj = nn.Parameter(_seeded((num_heads, hidden), s + 4, 0.05, self.main_dtype))
        self.g_out_proj = nn.Parameter(_seeded((inner, hidden), s + 5, 0.05, self.main_dtype))
        # per-head A_log, dt_bias, o_norm affine, and the output projection.
        self.a_log = nn.Parameter(_seeded((num_heads,), s + 6, 0.05, torch.float32))
        self.dt_bias = nn.Parameter(_seeded((inner,), s + 7, 0.05, torch.float32))
        self.o_norm_weight = nn.Parameter(torch.ones(head_dim, dtype=self.main_dtype))
        self.o_proj = nn.Parameter(_seeded((hidden, inner), s + 8, 0.05, self.main_dtype))
        self.kda = Glm5NextW2KDA(num_heads=num_heads, head_dim=head_dim,
                                 short_conv_kernel_size=conv_width, dtype_policy=dtype_policy)

    def forward(self, normed: torch.Tensor) -> torch.Tensor:
        qkv = _linear(normed, self.qkv_proj, self.main_dtype)
        raw_g = _linear(normed, self.raw_g_proj, self.main_dtype)
        beta_raw = _linear(normed, self.beta_proj, self.main_dtype)
        g_out = _linear(normed, self.g_out_proj, self.main_dtype)

        def _o_proj(x: torch.Tensor) -> torch.Tensor:
            return _linear(x, self.o_proj, self.main_dtype)

        return self.kda(
            qkv,
            conv_weight=self.conv_weight,
            raw_g=raw_g,
            beta_raw=beta_raw,
            a_log=self.a_log,
            dt_bias=self.dt_bias,
            g_out=g_out,
            o_norm_weight=self.o_norm_weight,
            o_proj=_o_proj,
            output_dtype=self.main_dtype,
        )


# ===========================================================================
# Dense FFN (first_k_dense_replace layers): plain FP16 SwiGLU MLP
# ===========================================================================
class _DenseMLP(nn.Module):
    def __init__(self, *, hidden: int, inter: int, layer_idx: int, main_dtype: torch.dtype) -> None:
        super().__init__()
        self.main_dtype = main_dtype
        s = 4000 + 13 * layer_idx
        self.gate = nn.Parameter(_seeded((inter, hidden), s + 1, 0.05, main_dtype))
        self.up = nn.Parameter(_seeded((inter, hidden), s + 2, 0.05, main_dtype))
        self.down = nn.Parameter(_seeded((hidden, inter), s + 3, 0.05, main_dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = _linear(x, self.gate, self.main_dtype)
        u = _linear(x, self.up, self.main_dtype)
        return _linear(F.silu(g) * u, self.down, self.main_dtype)


# ===========================================================================
# Decoder layer: (KDA | DSA) attention -> (dense | W2 MoE) FFN, mhc stand-in
# ===========================================================================
class AscendGlm5NextW2EagerDecoderLayer(nn.Module):
    """One eager GLM-5.3-Flash hybrid decoder layer.

    Attention is KDA (G4) on ``kda`` layers or DSA (G5) on ``full_attn_layers``;
    the FFN is a dense SwiGLU on the first ``first_k_dense_replace`` layers or the
    W2 MoE (G6) afterwards. The residual is carried as ``hc_mult`` parallel streams
    mixed by an eager mean (mhc sinkhorn stand-in; see the module docstring).
    """

    def __init__(self, *, config: Any, layer_idx: int, dtype_policy: Glm5NextW2DtypePolicy,
                 w2_experts: list[W2Expert], shared_expert: W2Expert) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype_policy = dtype_policy
        self.eps = float(getattr(config, "rms_norm_eps", 1e-5))
        self.main_dtype = dtype_policy.main_dtype
        self.accum_dtype = dtype_policy.accumulation_dtype
        self.router_dtype = dtype_policy.router_dtype

        hidden = int(config.hidden_size)
        self.is_dsa = _is_dsa_layer(config, layer_idx)
        self.layer_kind = "dsa" if self.is_dsa else "kda"

        # -- attention: KDA (G4) or DSA (G5) ----------------------------------
        self.input_layernorm = nn.Parameter(torch.ones(hidden, dtype=self.main_dtype))
        if self.is_dsa:
            self.attn = AscendGlm5NextW2DSA.from_config(config, dtype_policy=dtype_policy)
            self.attn.reset_parameters(seed=2000 + layer_idx)
        else:
            self.attn = _KdaAttnBlock(
                hidden=hidden,
                num_heads=int(getattr(config, "kda_num_heads", 64)),
                head_dim=int(getattr(config, "kda_head_dim", 128)),
                conv_width=int(getattr(config, "kda_short_conv_kernel_size", 4)),
                layer_idx=layer_idx,
                dtype_policy=dtype_policy,
            )

        # -- FFN: dense SwiGLU (first_k_dense_replace) or W2 MoE --------------
        self.post_attention_layernorm = nn.Parameter(torch.ones(hidden, dtype=self.main_dtype))
        first_dense = int(getattr(config, "first_k_dense_replace", 0))
        self.is_moe = layer_idx >= first_dense
        if self.is_moe:
            num_experts = int(config.n_routed_experts)
            self.gate = nn.Parameter(_seeded((num_experts, hidden), 9000 + layer_idx, 0.05, self.main_dtype))
            self.w2_experts = w2_experts
            self.moe = Glm5NextW2MoE.from_config(
                config,
                w2_experts=w2_experts,
                shared_expert=shared_expert.forward,
                dtype_policy=dtype_policy,
            )
        else:
            self.mlp = _DenseMLP(
                hidden=hidden,
                inter=int(getattr(config, "intermediate_size", config.moe_intermediate_size)),
                layer_idx=layer_idx,
                main_dtype=self.main_dtype,
            )

    def _collapse(self, streams: torch.Tensor) -> torch.Tensor:
        """mhc stand-in: mean-collapse the residual streams to the block input."""
        return streams.to(self.accum_dtype).mean(dim=1).to(self.main_dtype)

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """``hidden_states`` is the ``[T, hc_mult, hidden]`` multi-stream residual."""
        # --- attention block -------------------------------------------------
        block_input = self._collapse(hidden_states)
        normed = _rms_norm(block_input, self.input_layernorm, self.eps, self.accum_dtype)
        attn_out = self.attn(normed, positions) if self.is_dsa else self.attn(normed)
        hidden_states = hidden_states + attn_out.to(hidden_states.dtype).unsqueeze(1)

        # --- FFN block -------------------------------------------------------
        block_input = self._collapse(hidden_states)
        normed = _rms_norm(block_input, self.post_attention_layernorm, self.eps, self.accum_dtype)
        if self.is_moe:
            router_logits = _linear(normed, self.gate, self.router_dtype)
            ffn_out = self.moe(normed, router_logits, experts=self.w2_experts)
        else:
            ffn_out = self.mlp(normed)
        hidden_states = hidden_states + ffn_out.to(hidden_states.dtype).unsqueeze(1)
        return hidden_states


# ===========================================================================
# Backbone
# ===========================================================================
class AscendGlm5NextW2EagerModel(nn.Module):
    """Eager GLM-5.3-Flash backbone: embed -> hybrid decoder stack -> collapse."""

    def __init__(self, *, config: Any, dtype_policy: Glm5NextW2DtypePolicy) -> None:
        super().__init__()
        self.config = config
        self.dtype_policy = dtype_policy
        self.eps = float(getattr(config, "rms_norm_eps", 1e-5))
        self.main_dtype = dtype_policy.main_dtype
        self.accum_dtype = dtype_policy.accumulation_dtype

        self.vocab_size = int(config.vocab_size)
        self.hidden_size = int(config.hidden_size)
        self.hc_mult = int(getattr(config, "mhc_num_residual_streams", GLM5NEXT_MHC_NUM_RESIDUAL_STREAMS))
        self.num_hidden_layers = int(config.num_hidden_layers)

        self.embed_tokens = nn.Parameter(
            _seeded((self.vocab_size, self.hidden_size), 1234, 0.05, dtype_policy.embedding_dtype)
        )

        # One shared dummy W2 expert bank + shared expert for every MoE layer.
        num_experts = int(config.n_routed_experts)
        moe_inter = int(config.moe_intermediate_size)
        self.w2_experts = build_dummy_w2_expert_bank(num_experts, self.hidden_size, moe_inter, seed=100)
        self.shared_expert = build_dummy_w2_expert_bank(1, self.hidden_size, moe_inter, seed=555)[0]

        layers = [
            AscendGlm5NextW2EagerDecoderLayer(
                config=config,
                layer_idx=layer_idx,
                dtype_policy=dtype_policy,
                w2_experts=self.w2_experts,
                shared_expert=self.shared_expert,
            )
            for layer_idx in range(self.num_hidden_layers)
        ]
        self.layers = nn.ModuleList(layers)
        self.norm = nn.Parameter(torch.ones(self.hidden_size, dtype=self.main_dtype))
        # Layer-kind bookkeeping (for validation/reporting).
        self.dsa_layers = tuple(i for i, layer in enumerate(self.layers) if layer.is_dsa)
        self.kda_layers = tuple(i for i, layer in enumerate(self.layers) if not layer.is_dsa)
        self.moe_layers = tuple(i for i, layer in enumerate(self.layers) if layer.is_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids.long(), self.embed_tokens.to(self.main_dtype))

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_input_ids(input_ids)  # [T, hidden]
        hidden_states = hidden.unsqueeze(1).repeat(1, self.hc_mult, 1)  # [T, hc, hidden]
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)
        sample_hidden = hidden_states.to(self.accum_dtype).mean(dim=1).to(self.main_dtype)
        return _rms_norm(sample_hidden, self.norm, self.eps, self.accum_dtype)


# ===========================================================================
# Causal LM (+ MTP-1)
# ===========================================================================
class AscendGlm5NextW2EagerForCausalLM(nn.Module):
    """Host-runnable eager GLM-5.3-Flash causal LM for the dummy-weight CPU boot.

    Owns the backbone, the LM head, a lean hybrid KV report, the streamed W2 weight
    loader (G6), and the single MTP draft head (MTP-1). The CPU twin of the device
    ``AscendGlm5NextW2ForCausalLM``; not registered as an architecture.
    """

    def __init__(self, *, config: Any, dtype_policy: Glm5NextW2DtypePolicy | None = None) -> None:
        super().__init__()
        self.config = config
        self.dtype_policy = dtype_policy or ASCEND_GLM5NEXT_W2_DTYPE_POLICY
        self.model = AscendGlm5NextW2EagerModel(config=config, dtype_policy=self.dtype_policy)
        self.lm_head = nn.Parameter(
            _seeded((int(config.vocab_size), int(config.hidden_size)), 4321, 0.05, self.dtype_policy.lm_head_dtype)
        )
        # MTP-1: single next-token predict head (extra norm + head over hidden).
        self.num_nextn_predict_layers = int(getattr(config, "num_nextn_predict_layers", 1))
        self.mtp_norm = nn.Parameter(torch.ones(int(config.hidden_size), dtype=self.dtype_policy.main_dtype))
        # E1.3 fused param banks, allocated on demand by load_weights.
        self._expert_param_banks: dict[str, dict[str, torch.Tensor]] = {}
        self.load_report: dict[str, Any] = {}

    # -- forward / sampling -------------------------------------------------

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        if positions is None:
            positions = torch.arange(input_ids.shape[0], device=input_ids.device)
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _linear(hidden_states, self.lm_head, self.dtype_policy.logits_dtype)

    def sample_token(self, input_ids: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """Forward + greedy (argmax) sample of the last position's next token."""
        logits = self.compute_logits(self.forward(input_ids, positions))
        return logits[-1].argmax()

    def mtp_draft(self, input_ids: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """MTP-1 draft: backbone hidden -> MTP norm -> LM head -> next-token argmax.

        A light single-layer next-token predictor (GLM ships one MTP layer). The
        MTP MoE layer's experts are covered by :meth:`load_weights`.
        """
        hidden = self.forward(input_ids, positions)
        drafted = _rms_norm(hidden, self.mtp_norm, self.model.eps, self.dtype_policy.accumulation_dtype)
        logits = self.compute_logits(drafted)
        return logits[-1].argmax()

    # -- KV-cache report (lean, hybrid) -------------------------------------

    def kv_group_report(self) -> dict[str, Any]:
        """Semantic hybrid KV report: KDA linear-attn layers vs DSA full-attn.

        KDA layers carry a short-conv state + recurrent linear-attention state
        (no growing KV); DSA layers carry an MLA latent cache + a sparse-indexer
        compressed-history cache. Returns the per-class layer counts + spec labels.
        """
        return {
            "kda_layers": list(self.model.kda_layers),
            "dsa_layers": list(self.model.dsa_layers),
            "num_kda_layers": len(self.model.kda_layers),
            "num_dsa_layers": len(self.model.dsa_layers),
            "kda_spec": "linear_attention(conv_state + recurrent_state)",
            "dsa_spec": "mla_latent + sparse_indexer_cache",
            "num_hidden_layers": self.model.num_hidden_layers,
        }

    # -- streamed W2 weight loader (G6) -------------------------------------

    def _expert_geometry(self) -> dict[str, int]:
        return {
            "hidden_size": int(self.config.hidden_size),
            "moe_intermediate_size": int(self.config.moe_intermediate_size),
            "n_routed_experts": int(self.config.n_routed_experts),
            "num_hidden_layers": int(self.config.num_hidden_layers),
            "first_k_dense_replace": int(getattr(self.config, "first_k_dense_replace", 0)),
            "num_nextn_predict_layers": int(getattr(self.config, "num_nextn_predict_layers", 0)),
            "mtp_layer_index": int(getattr(self.config, "mtp_layer_index", int(self.config.num_hidden_layers))),
        }

    def _ensure_expert_bank(self, block: str, geometry: dict[str, int]) -> dict[str, torch.Tensor]:
        """Allocate the fused E1.3 param bank (w13/w2 codes+scale) for a block."""
        if block not in self._expert_param_banks:
            num_experts = geometry["n_routed_experts"]
            hidden = geometry["hidden_size"]
            inter = geometry["moe_intermediate_size"]
            self._expert_param_banks[block] = {
                # w13 fuses gate(w1)+up(w3): 2*inter output rows.
                "w13_codes": torch.zeros((num_experts, 2 * inter, hidden // 4), dtype=torch.uint8),
                "w13_scale": torch.zeros((num_experts, 2 * inter // 32, hidden // 32), dtype=torch.float32),
                "w2_codes": torch.zeros((num_experts, hidden, inter // 4), dtype=torch.uint8),
                "w2_scale": torch.zeros((num_experts, hidden // 32, inter // 32), dtype=torch.float32),
            }
        return self._expert_param_banks[block]

    def _place_expert_tensor(self, name: str, weight: torch.Tensor, geometry: dict[str, int]) -> str | None:
        """Validate + place one routed-expert tensor into its fused bank slot."""
        mapping = map_expert_tensor(name, geometry)
        expected = expected_expert_shape(mapping.slot, mapping.kind, geometry)
        if tuple(weight.shape) != expected:
            raise ValueError(f"{name}: shape {tuple(weight.shape)} != expected {expected}")
        bank = self._ensure_expert_bank(mapping.block, geometry)
        target = bank[mapping.target_param]
        rows = weight.shape[0]
        target[mapping.expert_id, mapping.row_offset:mapping.row_offset + rows] = weight
        return mapping.target_param

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> dict[str, int]:
        """Stream ``(name, tensor)`` -> fused W2 banks / FP16 accounting.

        Routed-expert tensors are validated (shape) and placed into the fused E1.3
        param banks via the G6 weight mapping; FP16 tensors are counted by name;
        vision tensors are excluded. Returns a load report (per-lane counts). This
        is the CPU twin of the device streamed loader -- it fills the W2 seam that
        matters and never materialises the full expert bank in one place.
        """
        geometry = self._expert_geometry()
        report = {"w2_expert": 0, "fp16": 0, "exclude": 0, "expert_blocks": 0}
        for name, weight in weights:
            cls = classify_tensor(name)
            if cls is WeightClass.EXCLUDE:
                report["exclude"] += 1
            elif cls is WeightClass.W2_EXPERT:
                self._place_expert_tensor(name, weight, geometry)
                report["w2_expert"] += 1
            else:
                report["fp16"] += 1
        report["expert_blocks"] = len(self._expert_param_banks)
        self.load_report = report
        return report
