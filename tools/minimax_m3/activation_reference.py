# SPDX-License-Identifier: Apache-2.0
"""Portable activation round trips for controlled M3.1 QAT experiments.

These deliberately slow tensor references are not production quantized GEMMs.
Native NVIDIA quantizers provide independent expected outputs for device gates.
"""

import torch


def round_e4m3(value, power=torch.exp2):
    absolute = value.float().abs()
    step = power((torch.floor(torch.log2(absolute.clamp_min(2**-24))) - 3).clamp_min(-9))
    return (torch.round(absolute / step) * step).clamp_max(448) * value.sign()


def nvfp4_roundtrip(value, power=torch.exp2):
    """E2M1 with E4M3 scales per 16 values; requires unit input global scale."""
    groups = value.float().unflatten(-1, (-1, 16))
    scale = round_e4m3(groups.abs().amax(-1) / 6, power)
    # Reciprocal then multiply is intentional: division changes boundary ties
    # compared with the native NVIDIA quantizer, even on the same GPU.
    normalized = groups * torch.reciprocal(torch.where(scale > 0, scale, 1.0))[..., None]
    absolute = normalized.abs()
    step = power(torch.floor(torch.log2(absolute.clamp_min(1e-30))).clamp_min(0) - 1)
    quantized = (torch.round(absolute / step) * step).clamp_max(6) * normalized.sign()
    return (quantized * scale[..., None]).flatten(-2).to(value.dtype)


def mxfp8_roundtrip(value, power=torch.exp2):
    """E4M3 with ceiling-rounded power-of-two scales per 32 values."""
    groups = value.float().unflatten(-1, (-1, 32))
    scale = power(torch.ceil(torch.log2(torch.where(groups.abs().amax(-1) > 0, groups.abs().amax(-1), 1.0) / 448)))
    quantized = round_e4m3(groups / scale[..., None], power)
    return (quantized * scale[..., None]).flatten(-2).to(value.dtype)


def quantized_dense_projection(prefix):
    return ".layers." in prefix and prefix.endswith(
        (
            ".qkv_proj",
            ".q_proj",
            ".k_proj",
            ".v_proj",
            ".index_q_proj",
            ".index_k_proj",
            ".indexer_proj",
            ".o_proj",
            ".gate_up_proj",
            ".down_proj",
        )
    )


def warmup_activation_reference(device):
    """Reserve CANN operator resources before source banks occupy device memory."""
    host = torch.linspace(-6, 6, 512, dtype=torch.float16).reshape(8, 64)
    value = host.to(device)
    for function in (nvfp4_roundtrip, mxfp8_roundtrip):
        actual = function(value).cpu()
        expected = function(host)
        if not torch.equal(actual, expected):
            raise RuntimeError("Activation warmup disagrees with CPU; refusing to load the model")
    if not torch.equal(value.float().cpu(), host.float()):
        raise RuntimeError("Device cast corrupted during activation warmup")
