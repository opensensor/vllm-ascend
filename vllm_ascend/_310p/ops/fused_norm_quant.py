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
"""RMSNorm and per-token activation quant fused into one 310P kernel.

``npu_dynamic_quant`` does no arithmetic but still reads and writes the whole
activation before every W8A8 matmul. Wherever that activation comes straight out
of an RMSNorm, ``_C_ascend.npu_rms_norm_dynamic_quant`` and its add-residual
sibling produce the INT8 tensor and its per-token scale directly, and that pass
disappears.

The kernels exist only on 310P -- ``csrc/build_aclnn.sh`` builds them in the
ascend310p branch -- and are declared fp16-only there, so every entry point
here is guarded.

Off unless ``VLLM_ASCEND_ENABLE_FUSED_NORM_QUANT=1``. The fusion is correct --
it reproduces the three-op path to within an int8 step, and greedy output is
unchanged -- but it is free rather than a win: the fused kernel only just beats
``npu_add_rms_norm`` plus ``npu_dynamic_quant`` at hidden 5120 and loses badly
below that, so end to end it measures as nothing. The numbers, and what the
kernel would need to actually pay, are in tools/310p/README.md.
"""

import torch
from vllm.model_executor.layers.layernorm import GemmaRMSNorm

import vllm_ascend.envs as envs_ascend
from vllm_ascend.utils import is_310p

_FUSED_AVAILABLE: bool | None = None

# Attribute the derived gamma is cached under on the norm module.
_GAMMA_ATTR = "_ascend_fused_gamma"


def _fused_available() -> bool:
    """Whether the fused kernels are built and registered on this device.

    Resolved on first use, not at import: the extension registers its ops when
    ``enable_custom_op()`` runs, which is later than the worker patches load.
    """
    global _FUSED_AVAILABLE
    if _FUSED_AVAILABLE is None:
        _FUSED_AVAILABLE = (
            envs_ascend.VLLM_ASCEND_ENABLE_FUSED_NORM_QUANT
            and is_310p()
            and hasattr(torch.ops._C_ascend, "npu_add_rms_norm_dynamic_quant")
        )
    return _FUSED_AVAILABLE


def can_fuse(linear: torch.nn.Module | None, hidden_states: torch.Tensor) -> bool:
    """Whether ``linear`` can consume a pre-quantized activation from here.

    False for an unquantized linear, for a quant scheme with no pre-quantized
    entry point, for a projection that does not exist (a sparse MoE block has no
    ``gate_up_proj``), for a dtype the kernel does not accept, and off 310P.
    """
    if not _fused_available() or hidden_states.dtype is not torch.float16:
        return False
    return hasattr(getattr(linear, "quant_method", None), "apply_quantized")


def apply_quantized(
    linear: torch.nn.Module,
    quantized_x: torch.Tensor,
    pertoken_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Run ``linear`` on an activation the caller already quantized.

    Both call sites are column-parallel, so this skips only the activation quant
    ``linear.forward`` would have done -- no collective is bypassed.
    """
    bias = None if linear.skip_bias_add else getattr(linear, "bias", None)
    return linear.quant_method.apply_quantized(
        linear, quantized_x, pertoken_scale, bias=bias, output_dtype=output_dtype
    )


def norm_gamma(norm: torch.nn.Module) -> torch.Tensor:
    """The gamma ``norm`` would apply, in the form the kernel wants.

    A ``GemmaRMSNorm`` scales by ``1 + weight`` rather than ``weight``, so that
    add has to happen somewhere. Doing it per call would add a kernel launch to
    every fused site, and 310P decode is launch-bound, so the derived tensor is
    cached on the module -- the weight is frozen by the time any forward runs.
    """
    if not isinstance(norm, GemmaRMSNorm):
        return norm.weight

    gamma = getattr(norm, _GAMMA_ATTR, None)
    if gamma is None:
        gamma = (1.0 + norm.weight).detach()
        setattr(norm, _GAMMA_ATTR, gamma)
    return gamma


def rms_norm_quant(x: torch.Tensor, norm: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm then per-token quant, as ``(quantized_x, pertoken_scale)``."""
    return torch.ops._C_ascend.npu_rms_norm_dynamic_quant(x, norm_gamma(norm), None, None, norm.variance_epsilon)


def add_rms_norm_quant(
    x: torch.Tensor, residual: torch.Tensor, norm: torch.nn.Module
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Residual add, RMSNorm, then per-token quant.

    Returns ``(quantized_x, pertoken_scale, residual_out)``. ``residual_out`` is
    the fp16 sum, which is what ``AscendGemmaRMSNorm310.forward_oot`` returns as
    its residual, so the two paths stay interchangeable layer to layer.
    """
    return torch.ops._C_ascend.npu_add_rms_norm_dynamic_quant(
        x, residual, norm_gamma(norm), None, None, norm.variance_epsilon
    )
