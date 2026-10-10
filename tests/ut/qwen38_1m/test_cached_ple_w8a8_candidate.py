# SPDX-License-Identifier: Apache-2.0
"""Match the existing W8 policy and keep graph cache/rollback lifetimes safe."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tools.qwen4exp.resident_candidates import cached_ple_w8a8 as candidate
from vllm_ascend.models.qwen4_exp.lm_head_w8a8 import dynamic_w8a8_linear, quantize_linear_weight


def module():
    generator = torch.Generator().manual_seed(6812)
    return SimpleNamespace(
        kv_proj_weight=torch.nn.Parameter(torch.randn(32, 16, generator=generator).half(), requires_grad=False),
        projection_execution="float16",
    )


@torch.no_grad()
def test_projection_matches_existing_w8_policy_and_quantizes_only_once(monkeypatch):
    monkeypatch.setattr(candidate, "_supports_device", lambda device: True)
    value = module()
    weight = value.kv_proj_weight.clone()
    pointer = value.kv_proj_weight.data_ptr()
    quantizer = Mock(wraps=quantize_linear_weight)
    monkeypatch.setattr(candidate, "quantize_linear_weight", quantizer)
    inputs = torch.randn(3, 16).half()
    packed, scales = quantize_linear_weight(weight)
    expected = dynamic_w8a8_linear(inputs, packed.transpose(0, 1).contiguous(), scales)
    fallback = Mock(side_effect=AssertionError("eligible candidate fell back"))
    for _ in range(3):
        actual = candidate.project(value, inputs, fallback=fallback)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    quantizer.assert_called_once()
    assert value.kv_proj_weight.data_ptr() == pointer
    assert torch.equal(value.kv_proj_weight, weight)
    assert value.projection_execution == "float16"


@torch.no_grad()
def test_capture_hits_are_allowed_but_new_or_changed_weights_require_warmup(monkeypatch):
    monkeypatch.setattr(candidate, "_supports_device", lambda device: True)
    value = module()
    monkeypatch.setattr(candidate, "_is_capturing", lambda device: True)
    with pytest.raises(RuntimeError, match="before graph capture"):
        candidate.prepare_weight(value)
    monkeypatch.setattr(candidate, "_is_capturing", lambda device: False)
    first = candidate.prepare_weight(value)
    monkeypatch.setattr(candidate, "_is_capturing", lambda device: True)
    second = candidate.prepare_weight(value)
    assert all(a is b for a, b in zip(first, second))
    value.kv_proj_weight.add_(0.25)
    with pytest.raises(RuntimeError, match="after weight changes"):
        candidate.prepare_weight(value)


@pytest.mark.parametrize("reason", ["cpu", "grad", "existing_w8", "dtype", "inference_parameter", "alignment"])
def test_unsupported_paths_use_original_projection(monkeypatch, reason):
    value = module()
    monkeypatch.setattr(candidate, "_supports_device", lambda device: reason != "cpu")
    if reason == "existing_w8":
        value.projection_execution = "w8a8_dynamic"
    elif reason == "dtype":
        value.kv_proj_weight = torch.nn.Parameter(value.kv_proj_weight.float(), requires_grad=False)
    elif reason == "alignment":
        value.kv_proj_weight = torch.nn.Parameter(value.kv_proj_weight[:-1].clone(), requires_grad=False)
    elif reason == "inference_parameter":
        with torch.inference_mode():
            value.kv_proj_weight = torch.nn.Parameter(value.kv_proj_weight.clone(), requires_grad=False)
    inputs = torch.randn(3, 16).half()
    expected = object()
    fallback = Mock(return_value=expected)
    with torch.set_grad_enabled(reason == "grad"):
        assert candidate.project(value, inputs, fallback=fallback) is expected
    fallback.assert_called_once_with(value, inputs)
    assert not hasattr(value, candidate.CACHE_ATTRIBUTE)


@torch.no_grad()
def test_release_requires_cleared_graphs_and_allows_rebuild(monkeypatch):
    monkeypatch.setattr(candidate, "_supports_device", lambda device: True)
    value = module()
    first = candidate.prepare_weight(value)
    monkeypatch.setattr(candidate, "_is_capturing", lambda device: True)
    with pytest.raises(RuntimeError, match="clearing old graphs"):
        candidate.release_weight(value)
    assert hasattr(value, candidate.CACHE_ATTRIBUTE)
    monkeypatch.setattr(candidate, "_is_capturing", lambda device: False)
    candidate.release_weight(value)
    candidate.release_weight(value)
    assert candidate.prepare_weight(value)[0] is not first[0]
