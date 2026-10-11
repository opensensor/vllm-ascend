# SPDX-License-Identifier: Apache-2.0
"""Explicit experimental INT8 worker profile, separate from the W4 service.

Selecting this extension changes the target expert quantization. The checkpoint
remains W4 on disk. Conversion runs once after the original loader validates its
complete local expert inventory, and before KV profiling or graph capture.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import torch
from torch import nn
from vllm.logger import init_logger

from tools.qwen4exp.channel_weight_recipe import channel_bank
from tools.qwen4exp.native_channel_int8 import NativeChannelInt8, NativeDirectChannelActivation
from tools.qwen4exp.native_prefill import _capturing
from tools.qwen4exp.prefix_npu_validation import QwenPrefixValidationExtension

CACHED_ROWS = (3, 6, 30, 60, 128, 640, 1280, 2560, 6400, 12800, 25600)
RECIPE = "w4_g128_to_channel_int8_v1"


class ServingConfiguration:
    """Retain graph addresses; bound eager descriptors across arbitrary tails."""

    def config(self, values, device):
        if values[0] in CACHED_ROWS:
            return super().config(values, device)
        if _capturing(device):
            raise RuntimeError("experimental INT8 graph shape was not admitted")
        # Eager tensors allocate and launch on the current stream. Stream-ordered
        # allocator reuse keeps the descriptor alive until its queued use ends.
        return torch.tensor(values, dtype=torch.int64, device=device)


class ServingProjection(ServingConfiguration, NativeChannelInt8):
    pass


class ServingActivation(ServingConfiguration, NativeDirectChannelActivation):
    pass


def resources(build):
    manifest = json.loads((Path(build) / "provenance.json").read_text())
    entries = (manifest["bridge"], manifest["binaries"]["native_hybrid_prefill"])
    for entry in entries:
        if hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("experimental INT8 binary digest mismatch")
    torch.ops.load_library(manifest["bridge"]["path"])
    kernel = getattr(torch.classes, manifest["namespace"]).Kernel
    launch = getattr(torch.ops, manifest["namespace"]).launch
    binary = manifest["binaries"]["native_hybrid_prefill"]["path"]
    return (
        ServingProjection(kernel(binary, "qwen_channel_int8_projection_v1"), launch),
        ServingActivation(
            kernel(binary, "qwen_direct_channel_activation_v1"),
            launch,
            local_kernel=kernel(binary, "qwen_direct_local_channel_activation_v1"),
        ),
    )


def stage_bank_on_host(bank):
    """Allocate unloaded W4 storage on CPU; upload only the converted INT8 bank.

    The original loader must subsequently fill and validate every operand.
    This deliberately does not copy the constructor's zero-initialized device
    bank to the host. It is opt-in and awaits a fresh-engine hardware gate.
    """
    destination = bank.weight.device
    for name in ("weight", "weight_scale", "weight_offset"):
        tensor = getattr(bank, name)
        setattr(bank, name, nn.Parameter(torch.empty_like(tensor, device="cpu"), requires_grad=False))
    bank.weight_sum = torch.empty_like(bank.weight_sum, device="cpu")
    return destination


def replace_bank(bank, *, progress=None, cache_path=None, target_device=None):
    """Replace, rather than retain, the original device bank after conversion."""
    old_bytes = sum(t.numel() * t.element_size() for t in bank.parameters())
    old_bytes += bank.weight_sum.numel() * bank.weight_sum.element_size()
    weight, scale, relative_error = channel_bank(
        bank, progress=progress, cache_path=cache_path, target_device=target_device
    )
    new_bytes = weight.numel() * weight.element_size() + scale.numel() * scale.element_size()
    bank.weight = nn.Parameter(weight, requires_grad=False)
    bank.weight_scale = nn.Parameter(scale, requires_grad=False)
    del bank.weight_offset
    del bank.weight_sum
    return {"w4_bytes": old_bytes, "int8_bytes": new_bytes, "weight_relative_l2": relative_error}


def forward_grouped(self, inputs, weights, ids):
    """Same router ownership and ordered route reduction; whole-K INT8 dots."""
    import torch_npu

    from vllm_ascend.models.qwen4_exp import w4_moe

    projection, activation = self._channel_int8_resources
    dispatch = w4_moe.build_grouped_expert_dispatch(
        weights,
        ids,
        num_local_experts=self.num_local_experts,
        expert_offset=self.expert_offset,
        weight_dtype=self.compute_dtype,
        count_mode=self.grouped_route_count_mode,
    )
    # All route counts and ownership remain on device. INT32 gather avoids the
    # 310P's slower INT64 indexing implementation for bounded route indices.
    order = dispatch.token_indices.index_select(0, dispatch.order.to(torch.int32)).to(torch.int32).contiguous()
    ends = dispatch.group_list.contiguous()
    token_codes, token_scales = activation(inputs)
    gate = self.projections["gate_up_proj"]
    projected = projection(
        token_codes.index_select(0, order).contiguous(),
        gate.weight,
        gate.weight_scale,
        token_scales.index_select(0, order).contiguous(),
        ends,
    )
    hidden = torch_npu.npu_swiglu(projected, dim=-1)
    hidden_codes, hidden_scales = activation(hidden, ends)
    down = self.projections["down_proj"]
    output = projection(hidden_codes, down.weight, down.weight_scale, hidden_scales, ends)
    return w4_moe.finalize_grouped_routes(output, dispatch, weights, self.compute_dtype, self.grouped_finalize)


def install():
    """Called only by selecting ChannelInt8Extension on a fresh test engine."""
    from vllm_ascend.models.qwen4_exp.model import AscendQwen4ExpForCausalLM
    from vllm_ascend.models.qwen4_exp.w4_moe import W4SparseMoE

    original = AscendQwen4ExpForCausalLM.load_weights
    logger = init_logger("vllm.qwen_channel_int8")

    def load_weights(self, weights):
        policy = getattr(self.model.config, "ascend_channel_int8", None)
        if not isinstance(policy, dict) or policy.get("recipe") != RECIPE:
            raise ValueError("experimental INT8 extension requires an explicit matching checkpoint override")
        staged_devices = {}
        if type(policy.get("stage_w4_on_host", False)) is not bool:
            raise ValueError("stage_w4_on_host must be an explicit boolean")
        if policy.get("stage_w4_on_host", False):
            # The existing streaming loader retains its complete inventory,
            # shape, dtype, scale, and offset checks; only destination changes.
            for index, layer in enumerate(self.model.layers):
                moe = layer.mlp
                if not isinstance(moe, W4SparseMoE) or not moe.native_int4 or not moe.fused_gate_up:
                    raise ValueError("host staging requires the qualified fused W4 expert geometry")
                for name, bank in moe.projections.items():
                    staged_devices[index, name] = stage_bank_on_host(bank)
            previous_threads = torch.get_num_threads()
            torch.set_num_threads(min(6, len(os.sched_getaffinity(0))))
            try:
                loaded = original(self, weights)
            finally:
                torch.set_num_threads(previous_threads)
        else:
            loaded = original(self, weights)
        native_resources = resources(policy["build"])
        logger.info("INT8_EXPERT_CONVERSION start: synchronizing loaded W4 operands")
        torch.npu.synchronize()
        logger.info("INT8_EXPERT_CONVERSION operands ready; converting expert-sized CPU batches")
        started = time.monotonic()
        records = []
        original_threads = torch.get_num_threads()
        torch.set_num_threads(min(6, len(os.sched_getaffinity(0))))
        try:
            for index, layer in enumerate(self.model.layers):
                moe = layer.mlp
                if not isinstance(moe, W4SparseMoE) or not moe.native_int4 or not moe.fused_gate_up:
                    raise ValueError("experimental INT8 target requires the qualified fused W4 expert geometry")
                record = {}
                for name, bank in moe.projections.items():
                    bank_start = time.monotonic()
                    cache_path = (
                        (Path(policy["cache"]) / f"rank-{moe.expert_tp_rank}" / f"layer-{index:02d}-{name}.safetensors")
                        if policy.get("cache")
                        else None
                    )
                    logger.info("INT8_EXPERT_CONVERSION bank_start layer=%d bank=%s cache=%s", index, name, cache_path)

                    def progress(done, total, index=index, name=name):
                        logger.info("INT8_EXPERT_CONVERSION layer=%d bank=%s experts=%d/%d", index, name, done, total)

                    record[name] = replace_bank(
                        bank,
                        progress=progress,
                        cache_path=cache_path,
                        target_device=staged_devices.get((index, name)),
                    )
                    logger.info(
                        "INT8_EXPERT_CONVERSION bank_done layer=%d bank=%s elapsed_s=%.2f",
                        index,
                        name,
                        time.monotonic() - bank_start,
                    )
                moe._channel_int8_resources = native_resources
                records.append(record)
                logger.info("INT8_EXPERT_CONVERSION layer=%d banks=%s", index, record)
        finally:
            torch.set_num_threads(original_threads)
        torch.npu.synchronize()
        torch.npu.empty_cache()
        logger.info(
            "INT8_EXPERT_CONVERSION complete layers=%d elapsed_s=%.2f w4_bytes=%d int8_bytes=%d "
            "allocated_bytes=%d startup_only_host_transfer=true",
            len(records),
            time.monotonic() - started,
            sum(bank["w4_bytes"] for layer in records for bank in layer.values()),
            sum(bank["int8_bytes"] for layer in records for bank in layer.values()),
            torch.npu.memory_allocated(),
        )
        return loaded

    AscendQwen4ExpForCausalLM.load_weights = load_weights
    W4SparseMoE._forward_grouped_chunk = forward_grouped
    W4SparseMoE._forward_routed = forward_grouped


class ChannelInt8Extension(QwenPrefixValidationExtension):
    """Retain the baseline's thermal, resident, and prefix diagnostic controls."""


install()
