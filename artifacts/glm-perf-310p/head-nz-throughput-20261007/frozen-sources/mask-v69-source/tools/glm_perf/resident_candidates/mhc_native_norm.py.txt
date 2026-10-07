# SPDX-License-Identifier: Apache-2.0
"""Opt-in mHC input norm experiment, preserving FP32 input and weight math."""


def native_norm(x, weight, eps, operation):
    # Casting x to FP16 before this operation changes the residual contract.
    # Keep FP32 through normalization, then use the existing final dtype.
    normalized, _ = operation(x.float(), weight.float(), eps)
    return normalized.to(weight.dtype)


def replacements():
    # Worker-only dependency; preparation does not initialize a device or
    # allocate persistent weight copies.
    import torch_npu

    def normalize(x, weight, eps):
        return native_norm(x, weight, eps, torch_npu.npu_rms_norm)

    return {"vllm_ascend.patch.worker.patch_mhc_norm:_mhc_rms_norm": normalize}
