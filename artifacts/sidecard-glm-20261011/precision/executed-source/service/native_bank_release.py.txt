# SPDX-License-Identifier: Apache-2.0
"""Release CUDA-only native expert storage after real placement qualification."""

import gc

from .native_expert_probe import NATIVE_LAYOUT, SCALE_LAYOUT

BANK_FIELDS = (
    "gate_up_packed_bank",
    "gate_packed_bank",
    "up_packed_bank",
    "down_packed_bank",
    "gate_up_scale_bank",
    "gate_scale_bank",
    "up_scale_bank",
    "down_scale_bank",
)
EXPERT_FIELDS = ("gate_packed", "up_packed", "down_packed", "gate_scale", "up_scale", "down_scale")
DECODE_CHECKS = 3
PREFILL_CHECKS = 1


def release_cuda_only_bank(bank, partition, layer, qualification):
    if partition.native_enabled(layer):
        raise ValueError("cannot release an actively routed native bank")
    if qualification["decode"] < DECODE_CHECKS or qualification["prefill"] < PREFILL_CHECKS:
        raise ValueError("native release requires completed real-input placement checks")
    if getattr(bank, "sidecard_native_released", False):
        return bank.sidecard_native_released_bytes
    if (
        bank.offload_to_cpu
        or not bank.grouped_ready
        or bank.num_local_experts != 72
        or bank.local_expert_offset not in (144, 216)
        or len(bank) != 288
        or getattr(bank, "native_weight_layout", None) != NATIVE_LAYOUT
        or getattr(bank, "native_weight_scale_layout", None) != SCALE_LAYOUT
    ):
        raise ValueError("native release requires the authenticated grouped one-card bank")
    import torch

    tensors = [getattr(bank, field, None) for field in BANK_FIELDS]
    if any(not isinstance(tensor, torch.Tensor) or tensor.ndim != 3 or tensor.shape[0] != 72 for tensor in tensors):
        raise ValueError("native bank references are incomplete")
    device = tensors[0].device
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("native bank devices differ")
    # Count allocations once; projection and per-expert views alias these banks.
    storages = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes() for tensor in tensors}
    released = sum(storages.values())
    if device.type == "npu":
        torch.npu.synchronize()
    for expert in bank[bank.local_expert_offset : bank.local_expert_offset + bank.num_local_experts]:
        for field in EXPERT_FIELDS:
            setattr(expert, field, None)
    for field in BANK_FIELDS:
        setattr(bank, field, None)
    bank.grouped_ready = False
    bank.sidecard_native_released = True
    bank.sidecard_native_released_bytes = released
    del tensors
    gc.collect()
    if device.type == "npu":
        torch.npu.empty_cache()
    return released
