# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp multi-token predictor for the Ascend 310P model path.

The MTP checkpoint contains FP16 fused experts. The target model's INT8 expert
loader cannot load them, so this module owns a separate FP16 draft layer.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import copy

import torch
from torch import nn
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsPP
from vllm.model_executor.models.utils import make_empty_intermediate_tensors_factory, maybe_prefix
from vllm.sequence import IntermediateTensors

from .dtype_policy import Qwen4ExpDtypePolicy
from .model import (
    AscendQwen4ExpDecoderLayer,
    AscendQwen4ExpForCausalLM,
    _GatedResidual,
    _grouped_rms_norm,
    _linear,
    _remap_non_expert,
    _resolve_expert_sharding,
)
from .moe import _w8a16_linear_npu, route_topk, swiglu_gate_up


class _MTPFP16MoE(nn.Module):
    """Local slice of the FP16 MTP checkpoint, optionally stored as W8A16."""

    def __init__(
        self,
        config: object,
        policy: Qwen4ExpDtypePolicy,
        sharding: tuple[int, int],
        quantize_experts: bool = False,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.num_experts = int(config.num_experts)
        self.top_k = min(int(config.num_experts_per_tok), self.num_experts)
        self.expert_rank, self.expert_tp_size = sharding
        if self.num_experts % self.expert_tp_size:
            raise ValueError("MTP expert count must be divisible by tensor parallel size")
        self.num_local_experts = self.num_experts // self.expert_tp_size
        self.expert_offset = self.expert_rank * self.num_local_experts
        hidden = int(config.hidden_size)
        intermediate = int(config.moe_intermediate_size)
        self.hidden_size = hidden
        self.intermediate_size = intermediate
        self.quantized_experts = quantize_experts
        self.gate = nn.Parameter(torch.zeros(self.num_experts, hidden, dtype=policy.main_dtype))
        if quantize_experts:
            # The checkpoint remains FP16; each local expert is quantized once
            # during load. The runtime 310P W8A16 matmul consumes [K, N] INT8
            # weights and one FP16 symmetric scale per output channel.
            self.gate_up_proj = nn.ParameterList(
                nn.Parameter(torch.zeros(hidden, 2 * intermediate, dtype=torch.int8), requires_grad=False)
                for _ in range(self.num_local_experts)
            )
            self.down_proj = nn.ParameterList(
                nn.Parameter(torch.zeros(intermediate, hidden, dtype=torch.int8), requires_grad=False)
                for _ in range(self.num_local_experts)
            )
            self.gate_up_proj_scale = nn.ParameterList(
                nn.Parameter(torch.ones(2 * intermediate, dtype=policy.main_dtype), requires_grad=False)
                for _ in range(self.num_local_experts)
            )
            self.down_proj_scale = nn.ParameterList(
                nn.Parameter(torch.ones(hidden, dtype=policy.main_dtype), requires_grad=False)
                for _ in range(self.num_local_experts)
            )
        else:
            self.gate_up_proj = nn.ParameterList(
                nn.Parameter(torch.zeros(2 * intermediate, hidden, dtype=policy.main_dtype))
                for _ in range(self.num_local_experts)
            )
            self.down_proj = nn.ParameterList(
                nn.Parameter(torch.zeros(hidden, intermediate, dtype=policy.main_dtype))
                for _ in range(self.num_local_experts)
            )
        shared_intermediate = int(getattr(config, "shared_expert_intermediate_size", 0) or 0)
        if shared_intermediate % self.expert_tp_size:
            raise ValueError("MTP shared expert size must be divisible by tensor parallel size")
        self.local_shared_intermediate = shared_intermediate // self.expert_tp_size
        if self.local_shared_intermediate:
            self.shared_gate_up = nn.Parameter(
                torch.zeros(2 * self.local_shared_intermediate, hidden, dtype=policy.main_dtype)
            )
            self.shared_down = nn.Parameter(
                torch.zeros(hidden, self.local_shared_intermediate, dtype=policy.main_dtype)
            )
            self.shared_expert_gate = nn.Parameter(torch.zeros(1, hidden, dtype=policy.main_dtype))
        self.renormalize = bool(getattr(config, "norm_topk_prob", True))
        self.routed_scaling_factor = float(getattr(config, "routed_scaling_factor", 1.0) or 1.0)
        self._tp_reduce = None
        if self.expert_tp_size > 1:
            from vllm.distributed import tensor_model_parallel_all_reduce

            self._tp_reduce = tensor_model_parallel_all_reduce

    def load_expert_weight(self, projection: str, local_id: int, source: torch.Tensor) -> None:
        weights = getattr(self, projection)
        with torch.no_grad():
            if not self.quantized_experts:
                weights[local_id].copy_(source)
                return
            # Quantize one expert at a time so an entire 512-expert checkpoint
            # tensor is never expanded to FP32 on the NPU or host.
            source_fp32 = source.to(torch.float32)
            scales = (source_fp32.abs().amax(dim=1) / 127).clamp_min(1e-8)
            quantized = torch.round(source_fp32 / scales[:, None]).clamp(-127, 127).to(torch.int8)
            weights[local_id].copy_(quantized.t().contiguous())
            getattr(self, projection + "_scale")[local_id].copy_(scales.to(self.policy.main_dtype))

    def _expert_linear(self, x: torch.Tensor, projection: str, local_id: int) -> torch.Tensor:
        weights = getattr(self, projection)
        if not self.quantized_experts:
            return _linear(x, weights[local_id], self.policy.accumulation_dtype)
        scales = getattr(self, projection + "_scale")[local_id]
        if x.device.type == "npu":
            return _w8a16_linear_npu(x, weights[local_id], scales)
        return x.to(torch.float32) @ (weights[local_id].to(torch.float32) * scales.to(torch.float32)).to(torch.float32)

    def _forward_eager(self, x: torch.Tensor) -> torch.Tensor:
        logits = _linear(x, self.gate, self.policy.router_dtype)
        weights, ids = route_topk(
            logits,
            self.top_k,
            renormalize=self.renormalize,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        flat_ids = ids.flatten()
        flat_weights = weights.flatten()
        slot_ids = torch.arange(flat_ids.numel(), device=x.device)
        token_ids = slot_ids // self.top_k
        local_ids = flat_ids - self.expert_offset
        local_mask = (local_ids >= 0) & (local_ids < self.num_local_experts)
        local_ids = local_ids[local_mask]
        slot_ids = slot_ids[local_mask]
        token_ids = token_ids[local_mask]
        flat_weights = flat_weights[local_mask]
        route_outputs = torch.zeros(
            (flat_ids.numel(), x.shape[-1]), dtype=self.policy.accumulation_dtype, device=x.device
        )
        if local_ids.numel():
            order = torch.argsort(local_ids.to(self.policy.accumulation_dtype), stable=True)
            local_ids = local_ids[order]
            slot_ids = slot_ids[order]
            token_ids = token_ids[order]
            flat_weights = flat_weights[order]
            counts = torch.bincount(local_ids, minlength=self.num_local_experts)
            start = 0
            # One device-to-host boundary, matching the target MoE path.
            for expert_id, count in enumerate(counts.tolist()):
                if count == 0:
                    continue
                stop = start + count
                selected = x[token_ids[start:stop]]
                gate_up = self._expert_linear(selected, "gate_up_proj", expert_id)
                expert_output = self._expert_linear(swiglu_gate_up(gate_up), "down_proj", expert_id)
                route_outputs.index_copy_(
                    0, slot_ids[start:stop], expert_output.to(route_outputs.dtype) * flat_weights[start:stop, None]
                )
                start = stop
        output = route_outputs.view(x.shape[0], self.top_k, x.shape[-1]).sum(dim=1)
        if self.local_shared_intermediate:
            gate_up = _linear(x, self.shared_gate_up, self.policy.accumulation_dtype)
            shared = _linear(swiglu_gate_up(gate_up), self.shared_down, self.policy.accumulation_dtype)
            shared_gate = torch.sigmoid(_linear(x, self.shared_expert_gate, self.policy.accumulation_dtype))
            output = output + shared.to(output.dtype) * shared_gate
        if self.expert_tp_size > 1:
            assert self._tp_reduce is not None
            output = self._tp_reduce(output)
        return output.to(self.policy.main_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        capture = BreakableCUDAGraphCapture.current()
        if capture is None or not capture._capturing:
            return self._forward_eager(x)

        # Expert counts are read by the host to dispatch the MTP's small
        # per-expert matmuls. Keep that read outside graph capture, and copy
        # the result into a stable graph-pool tensor for following layers.
        from vllm_ascend.utils import weak_ref_tensor

        output = torch.empty_like(x)
        weak_output = weak_ref_tensor(output)
        weak_x = weak_ref_tensor(x)

        def run_experts_eager() -> None:
            weak_output.copy_(self._forward_eager(weak_x))

        capture.add_eager(run_experts_eager)
        return output


class _MTPPredictor(nn.Module):
    def __init__(self, vllm_config: object, prefix: str, policy: Qwen4ExpDtypePolicy) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.policy = policy
        self.hidden_size = int(config.hidden_size)
        self.hc_count = int(config.hc_count)
        self.num_mtp_layers = int(getattr(config, "mtp_num_hidden_layers", 1))
        if self.num_mtp_layers < 1:
            raise ValueError("Qwen4Exp MTP requires at least one draft layer")
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            params_dtype=policy.embedding_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.fc_embedding = nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size, dtype=policy.main_dtype))
        self.fc_hidden = nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size, dtype=policy.main_dtype))
        self.pre_fc_norm_embedding = nn.Parameter(torch.zeros(self.hidden_size, dtype=policy.main_dtype))
        self.pre_fc_norm_hidden = nn.Parameter(torch.zeros(self.hc_count * self.hidden_size, dtype=policy.main_dtype))
        self.expert_sharding = _resolve_expert_sharding(vllm_config)
        runtime_device = getattr(getattr(vllm_config, "device_config", None), "device", None)
        quantize_experts = getattr(runtime_device, "type", None) == "npu"
        self.layers = nn.ModuleList()
        # The target's INT8 bank is several GiB. Do not construct it only to
        # replace it with the MTP checkpoint's FP16 bank.
        attention_config = copy(config)
        attention_config.num_experts = 0
        for idx in range(self.num_mtp_layers):
            # Absolute ids keep QSA draft cache entries distinct from target layers.
            layer = AscendQwen4ExpDecoderLayer(
                config=attention_config,
                layer_type="full_attention",
                layer_idx=int(config.num_hidden_layers) + idx,
                dtype_policy=policy,
                prefix=maybe_prefix(prefix, f"layers.{idx}"),
                expert_sharding=self.expert_sharding,
            )
            if layer.ple is not None:
                raise ValueError("Qwen4Exp MTP layers must not use PLE")
            if getattr(config, "num_experts", 0):
                layer.mlp = _MTPFP16MoE(
                    config,
                    policy,
                    self.expert_sharding,
                    quantize_experts=quantize_experts,
                )
            self.layers.append(layer)
        self.hyper_connection_mixer = _GatedResidual(
            hc_count=self.hc_count,
            hidden_size=self.hidden_size,
            lowrank=int(config.hc_lowrank),
            eps=float(config.rms_norm_eps),
            params_dtype=policy.main_dtype,
            compute_dtype=policy.accumulation_dtype,
            use_combine=False,
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.hc_count * self.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if intermediate_tensors is not None:
            hidden_states = intermediate_tensors["hidden_states"]
        else:
            if (
                hidden_states is None
                or hidden_states.ndim != 2
                or hidden_states.shape[-1] != self.hc_count * self.hidden_size
            ):
                raise ValueError("MTP requires target multi-stream hidden states")
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("MTP requires input_ids or inputs_embeds")
                inputs_embeds = self.embed_input_ids(input_ids)
            embedding = _grouped_rms_norm(
                inputs_embeds,
                self.pre_fc_norm_embedding,
                float(self.config.rms_norm_eps),
                self.hidden_size,
                self.policy.accumulation_dtype,
            )
            embedding = _linear(embedding, self.fc_embedding, self.policy.accumulation_dtype)
            hidden_states = _grouped_rms_norm(
                hidden_states,
                self.pre_fc_norm_hidden,
                float(self.config.rms_norm_eps),
                self.hidden_size,
                self.policy.accumulation_dtype,
            )
            hidden_states = hidden_states.reshape(-1, self.hc_count, self.hidden_size)
            projected = _linear(hidden_states, self.fc_hidden, self.policy.accumulation_dtype)
            # Missing HC injection weights mean unit residual into every stream.
            hidden_states = (projected + embedding[:, None, :]).flatten(-2).to(self.policy.main_dtype)
        multi_hidden = self.layers[spec_step_idx % self.num_mtp_layers](hidden_states, positions, input_ids)
        sample_hidden, _ = self.hyper_connection_mixer.mix(multi_hidden)
        return sample_hidden.to(self.policy.main_dtype), multi_hidden


class AscendQwen4ExpMTP(nn.Module, SupportsPP, MixtureOfExperts):
    """Text-only Qwen4Exp MTP drafter with a separate FP16 expert bank."""

    packed_modules_mapping = AscendQwen4ExpForCausalLM.packed_modules_mapping
    requires_raw_input_tokens = True
    uses_model_owned_mrope = True

    def __init__(self, *, vllm_config: object, prefix: str = "") -> None:
        super().__init__()
        if getattr(vllm_config.cache_config, "mamba_cache_mode", None) == "all":
            raise ValueError("Qwen4Exp MTP requires --mamba-cache-mode=align")
        if getattr(vllm_config.parallel_config, "pipeline_parallel_size", 1) != 1:
            raise ValueError("Qwen4Exp MTP currently requires pipeline parallel size 1")
        self.vllm_config = vllm_config
        self.dtype_policy = Qwen4ExpDtypePolicy.from_vllm_config(vllm_config)
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.model = _MTPPredictor(vllm_config, maybe_prefix(prefix, "mtp"), self.dtype_policy)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            params_dtype=self.dtype_policy.lm_head_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def share_target_lm_head_if_identical(self, target_model: nn.Module) -> bool:
        """Release the duplicate draft head only when its loaded weights match."""
        target_head = getattr(target_model, "lm_head", None)
        if target_head is None:
            return False
        draft_weight = self.lm_head.weight
        target_weight = target_head.weight
        if draft_weight.shape != target_weight.shape or not torch.equal(draft_weight, target_weight):
            return False
        self.lm_head = target_head
        return True

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds, spec_step_idx)

    def compute_logits(self, hidden_states: torch.Tensor, spec_step_idx: int = 0) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stream MTP tensors and the target's shared embedding/head by name."""
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        tp_rank, tp_size = self.model.expert_sharding
        for raw_name, tensor in weights:
            name = raw_name.removeprefix("model.language_model.").removeprefix("language_model.")
            if name.startswith("model.mtp."):
                name = name.removeprefix("model.")
            if name.startswith("mtp."):
                name = name.replace("mtp.", "model.", 1)
            elif name.startswith("embed_tokens."):
                name = "model." + name
            elif name.startswith("model.embed_tokens.") or name.startswith("lm_head."):
                pass
            else:
                continue
            if name == "model.embed_tokens.weight":
                self.model.embed_tokens.weight_loader(params[name], tensor)
                loaded.add(name)
                continue
            if name == "lm_head.weight":
                self.lm_head.weight_loader(params[name], tensor)
                loaded.add(name)
                continue
            if name.endswith(
                (
                    ".fc_embedding.weight",
                    ".fc_hidden.weight",
                    ".pre_fc_norm_embedding.weight",
                    ".pre_fc_norm_hidden.weight",
                )
            ):
                name = name.removesuffix(".weight")
            if name.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")):
                target_base = name.replace(".mlp.experts.", ".mlp.")
                bank = self.model.layers[int(name.split(".")[2])].mlp
                local = bank.num_local_experts
                if name.endswith(".gate_up_proj"):
                    expected = (bank.num_experts, 2 * bank.intermediate_size, bank.hidden_size)
                else:
                    expected = (bank.num_experts, bank.hidden_size, bank.intermediate_size)
                if tensor.dtype != self.dtype_policy.main_dtype or tuple(tensor.shape) != expected:
                    raise ValueError(
                        f"{raw_name}: expected {self.dtype_policy.main_dtype} {expected}, "
                        f"got {tensor.dtype} {tuple(tensor.shape)}"
                    )
                for index in range(local):
                    target_name = f"{target_base}.{index}"
                    projection = "gate_up_proj" if name.endswith(".gate_up_proj") else "down_proj"
                    bank.load_expert_weight(projection, index, tensor[tp_rank * local + index])
                    loaded.add(target_name)
                    if bank.quantized_experts:
                        loaded.add(f"{target_base}_scale.{index}")
                continue
            target = params.get(name)
            if target is not None and tuple(target.shape) == tuple(tensor.shape):
                with torch.no_grad():
                    target.copy_(tensor.to(target.dtype))
                loaded.add(name)
                continue
            if name.endswith((".mlp.shared_expert.gate_proj.weight", ".mlp.shared_expert.up_proj.weight")):
                base = name.rsplit(".mlp.shared_expert.", 1)[0] + ".mlp.shared_gate_up"
                target = params.get(base)
                if target is not None:
                    local = target.shape[0] // 2
                    part = 0 if ".gate_proj." in name else 1
                    with torch.no_grad():
                        target[part * local : (part + 1) * local].copy_(
                            tensor[tp_rank * local : (tp_rank + 1) * local].to(target.dtype)
                        )
                    loaded.add(base)
                continue
            if name.endswith(".mlp.shared_expert.down_proj.weight"):
                base = name.rsplit(".mlp.shared_expert.", 1)[0] + ".mlp.shared_down"
                target = params.get(base)
                if target is not None:
                    local = target.shape[1]
                    with torch.no_grad():
                        target.copy_(tensor[:, tp_rank * local : (tp_rank + 1) * local].to(target.dtype))
                    loaded.add(base)
                continue
            qsa_targets = AscendQwen4ExpForCausalLM._place_qsa_q_gate_tensor(self, params, name, tensor)
            if qsa_targets is not None:
                loaded.update(qsa_targets)
                continue
            qsa_target = AscendQwen4ExpForCausalLM._place_qsa_head_tensor(self, params, name, tensor, tp_rank, tp_size)
            if qsa_target is not None:
                loaded.add(qsa_target)
                continue
            placements = _remap_non_expert(name, self.config)
            if placements is None:
                continue
            for target_name, source_slice, offset in placements:
                target = params.get(target_name)
                if target is None:
                    continue
                source = tensor if source_slice is None else tensor[source_slice]
                if tuple(source.shape) != tuple(target.shape) and offset is None:
                    raise ValueError(f"{raw_name}: expected {tuple(target.shape)}, got {tuple(source.shape)}")
                with torch.no_grad():
                    if offset is None:
                        target.copy_(source.to(target.dtype))
                    else:
                        target[offset : offset + source.shape[0]].copy_(source.to(target.dtype))
                loaded.add(target_name)
        return loaded


__all__ = ["AscendQwen4ExpMTP"]
