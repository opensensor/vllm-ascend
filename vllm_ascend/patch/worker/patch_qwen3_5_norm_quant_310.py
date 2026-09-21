#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Fold Qwen3.5's RMSNorms into the activation quant of the linear below them.

Each W8A8 projection quantizes its input per token, and on the two sites where
that input is an RMSNorm output the 310P kernels can produce the INT8 tensor and
its scale in one pass -- ``mlp.gate_up_proj`` on every layer and
``self_attn.qkv_proj`` on the full-attention ones.

Both forwards below mirror upstream ``Qwen3NextAttention.forward`` and
``Qwen3NextDecoderLayer.forward`` step for step and fall back to the plain path
wherever the fusion does not apply. Nothing is installed unless
``VLLM_ASCEND_ENABLE_FUSED_NORM_QUANT`` is set, so the default build is
untouched -- see ``vllm_ascend._310p.ops.fused_norm_quant`` for why it is off.
"""

import torch
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

import vllm_ascend.envs as envs_ascend
from vllm_ascend._310p.ops.fused_norm_quant import (
    add_rms_norm_quant,
    apply_quantized,
    can_fuse,
    rms_norm_quant,
)


def _attention_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    quantized_hidden_states: torch.Tensor = None,
    pertoken_scale: torch.Tensor = None,
) -> torch.Tensor:
    if quantized_hidden_states is not None:
        # input_layernorm already produced qkv_proj's INT8 input and its scale;
        # hidden_states is then only carrying the output dtype.
        qkv = apply_quantized(self.qkv_proj, quantized_hidden_states, pertoken_scale, hidden_states.dtype)
    else:
        qkv, _ = self.qkv_proj(hidden_states)
    q, k, v, gate = self._project_qkv_gate(qkv, positions)
    attn_output = self.attn(q, k, v)
    if gate is not None:
        attn_output = attn_output * torch.sigmoid(gate)
    output, _ = self.o_proj(attn_output)
    return output


def _mlp_forward_quantized(self, quantized_x, pertoken_scale, output_dtype):
    """``forward`` for an activation the caller already quantized for gate_up.

    Valid only when ``expert_gate`` is None -- the gate reads the unquantized
    activation, which the fused kernel never materialises. The caller checks.
    """
    gate_up = apply_quantized(self.gate_up_proj, quantized_x, pertoken_scale, output_dtype)
    out = self.act_fn(gate_up)
    out, _ = self.down_proj(out)
    return out


def _decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor = None,
    **kwargs: object,
):
    full_num_tokens = positions.shape[-1]
    # Sequence parallelism moves tokens between a norm and the projection that
    # consumes its output, so nothing fuses on that path.
    sequence_parallel = self.use_attn_reduce_scatter_for_moe

    if (
        not sequence_parallel
        and self.layer_type == "full_attention"
        and can_fuse(getattr(self.self_attn, "qkv_proj", None), hidden_states)
    ):
        if residual is None:
            residual = hidden_states
            quantized, scale = rms_norm_quant(hidden_states, self.input_layernorm)
        else:
            quantized, scale, residual = add_rms_norm_quant(hidden_states, residual, self.input_layernorm)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            quantized_hidden_states=quantized,
            pertoken_scale=scale,
        )
    else:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if sequence_parallel:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[:full_num_tokens]

        if self.layer_type == "linear_attention":
            # in_proj is FLOAT in the W8A8 checkpoints, so there is no
            # activation quant here to fold the norm into.
            hidden_states = self.linear_attn(hidden_states=hidden_states)
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        else:
            raise ValueError("Invalid layer_type")

    if self.layer_scale:
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states * (self.attn_layer_scale.to(hidden_states.dtype)[0] + 1)
        else:
            hidden_states = hidden_states * (self.attn_layer_scale.to(hidden_states.dtype) + 1)

    if sequence_parallel:
        tp_world_size = get_tensor_model_parallel_world_size()
        # small trick using minus, eg. -17 % 8 = 7
        sp_pad = (-hidden_states.shape[0]) % tp_world_size
        # pad if not divisible by world size
        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, sp_pad))
        hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)

    # Fully Connected. A sparse MoE block has no gate_up_proj, and an expert
    # gate needs the unquantized activation, so both keep the plain norm.
    if (
        not sequence_parallel
        and hasattr(self.mlp, "forward_quantized")
        and getattr(self.mlp, "expert_gate", None) is None
        and can_fuse(getattr(self.mlp, "gate_up_proj", None), hidden_states)
    ):
        quantized, scale, residual = add_rms_norm_quant(hidden_states, residual, self.post_attention_layernorm)
        hidden_states = self.mlp.forward_quantized(quantized, scale, hidden_states.dtype)
    else:
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if sequence_parallel:
            hidden_states = self.mlp(hidden_states, already_sequence_parallel=True)
        else:
            hidden_states = self.mlp(hidden_states)

    if self.layer_scale:
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states * (self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1)
        else:
            assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                f"shape must be the same {len(hidden_states.shape)}, {len(self.ffn_layer_scale.shape)}"
            )
            hidden_states = hidden_states * (self.ffn_layer_scale.to(hidden_states.dtype) + 1)

    return hidden_states, residual


if envs_ascend.VLLM_ASCEND_ENABLE_FUSED_NORM_QUANT:
    Qwen2MoeMLP.forward_quantized = _mlp_forward_quantized
    Qwen3NextAttention.forward = _attention_forward
    # Set on the Qwen3.5 subclass only: Qwen3Next keeps upstream's forward.
    Qwen3_5DecoderLayer.forward = _decoder_layer_forward
