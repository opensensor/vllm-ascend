# SPDX-License-Identifier: Apache-2.0
"""Admission and composition gates; native numerical comparisons run on 310P."""

import types
from unittest.mock import Mock

import pytest
import torch

from tools.minimax_m3 import activation_native
from tools.minimax_m3.activation_native import ClippedSwiGLUKernel
from tools.minimax_m3.resident import candidate_source
from vllm_ascend._310p.ops.minimax_m3 import clipped_swiglu


@pytest.mark.parametrize("alpha,beta,limit", [(1.702, 1.0, 7.0), (1.0, 0.0, 3.0)])
def test_cpu_fallback_preserves_parameters(alpha, beta, limit):
    kernel = ClippedSwiGLUKernel.__new__(ClippedSwiGLUKernel)
    kernel.launch = Mock(side_effect=AssertionError("CPU input reached native launch"))
    gate = torch.linspace(-20, 20, 512).half()
    up = gate.flip(0)
    torch.testing.assert_close(kernel(gate, up, alpha, beta, limit), clipped_swiglu(gate, up, alpha, beta, limit))
    kernel.launch.assert_not_called()


@pytest.mark.parametrize("mutation", ["grad", "dtype", "device", "shape", "strided", "empty", "tail", "parameters"])
def test_unqualified_inputs_are_not_admitted(mutation):
    gate = Mock(device=types.SimpleNamespace(type="npu"), dtype=torch.float16, shape=(4, 3072))
    gate.is_contiguous.return_value = True
    gate.numel.return_value = 12288
    up = Mock(device=gate.device, dtype=gate.dtype, shape=gate.shape)
    up.is_contiguous.return_value = True
    alpha = 1.702
    if mutation == "dtype":
        up.dtype = torch.float32
    if mutation == "device":
        up.device = types.SimpleNamespace(type="cpu")
    if mutation == "shape":
        up.shape = (1, 3072)
    if mutation == "strided":
        up.is_contiguous.return_value = False
    if mutation == "empty":
        gate.numel.return_value = 0
    if mutation == "tail":
        gate.numel.return_value = 12287
    if mutation == "parameters":
        alpha = 1.7
    with torch.set_grad_enabled(mutation == "grad"):
        assert not ClippedSwiGLUKernel.accepts(gate, up, alpha, 1.0, 7.0)


def test_descriptor_reuse_keeps_original_input_storage(monkeypatch):
    kernel = ClippedSwiGLUKernel.__new__(ClippedSwiGLUKernel)
    kernel.kernel, kernel.launch, kernel.configs = object(), Mock(), {}
    monkeypatch.setattr(kernel, "accepts", lambda *args: True)
    gate, up = torch.zeros(4, 3072).half(), torch.ones(4, 3072).half()
    first, second = kernel(gate, up), kernel(gate, up)
    assert first.dtype == second.dtype == torch.float32
    assert first.data_ptr() != second.data_ptr()
    calls = kernel.launch.call_args_list
    assert calls[0].args[1][0] is gate and calls[0].args[1][1] is up
    assert calls[0].args[1][3] is calls[1].args[1][3]
    assert len(kernel.configs) == 1


@pytest.mark.parametrize("dense", [True, False])
def test_admission_checks_device_storage_format(monkeypatch, dense):
    gate = Mock(device=types.SimpleNamespace(type="npu"), dtype=torch.float16, shape=(4, 3072))
    gate.is_contiguous.return_value = True
    gate.numel.return_value = 12288
    monkeypatch.setattr(activation_native, "_dense_format", lambda value: dense)
    with torch.inference_mode():
        assert ClippedSwiGLUKernel.accepts(gate, gate, 1.702, 1.0, 7.0) is dense


def test_activation_composes_with_hot_module_and_projection():
    source = """
def route_experts(): pass
def clipped_swiglu(): return "old"
def paged_sparse_attention(): pass
def grouped_w4_mm(*args): return "old"
class PackedM3Experts:
    def forward(self): return clipped_swiglu(1, 2), grouped_w4_mm(1, 2, 3, 4)
"""
    module = types.ModuleType("composition")
    exec(candidate_source(source, "projection", "activation"), module.__dict__)
    activation = Mock(return_value="fused")
    changes = module.replacements({"projection": lambda *args: "int8", "activation": activation})
    assert changes["vllm_ascend._310p.ops.minimax_m3:PackedM3Experts.forward"](None) == ("fused", "int8")
    for target in ("vllm_ascend._310p.ops.minimax_m3", "vllm_ascend.models.minimax_m3.minimax_m3"):
        assert isinstance(changes[target + ":clipped_swiglu"], types.FunctionType)
    activation.assert_called_once_with(1, 2, 1.702, 1.0, 7.0)
