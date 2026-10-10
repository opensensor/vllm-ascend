# SPDX-License-Identifier: Apache-2.0
"""GDN native norm respects decode limits, TP reduction and existing casts."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tools.qwen4exp.resident_candidates import gdn_output_rms as candidate
from vllm_ascend.models.qwen4_exp.model import _GDNAttention


@pytest.mark.parametrize(
    "tokens,grad,supported,tp",
    [
        (3, False, True, 1),
        (6, False, True, 2),
        (18, False, True, 2),
        (19, False, True, 1),
        (3, True, True, 1),
        (3, False, False, 1),
    ],
)
def test_decode_dispatch_preserves_complete_output_and_tp(monkeypatch, tokens, grad, supported, tp):
    norm = Mock(
        side_effect=lambda x, gamma, eps: (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * gamma, None)
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_rms_norm=norm))
    monkeypatch.setattr(candidate, "_supports_device", lambda device: supported)
    generator = torch.Generator().manual_seed(1103)
    module = SimpleNamespace(
        params=SimpleNamespace(head_v_dim=16),
        num_v_heads=2,
        value_dim=32,
        norm_weight=torch.ones(16),
        in_proj_z=torch.randn(32, 16, generator=generator),
        out_proj=torch.randn(16, 32, generator=generator),
        compute_dtype=torch.float32,
        params_dtype=torch.float16,
        rms_norm_eps=1e-6,
        tp_size=tp,
        _tp_reduce=Mock(side_effect=lambda x: x * 2),
    )
    inputs = torch.randn(tokens, 16, generator=generator)
    out = torch.randn(tokens, 2, 16, generator=generator)
    saved = out.clone()
    with torch.set_grad_enabled(grad):
        expected = _GDNAttention._project_output(module, inputs, out)
        module._tp_reduce.reset_mock()
        actual = candidate.project_output(module, inputs, out)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(out, saved)
    assert norm.call_count == (not grad and supported and tokens <= candidate.MAX_DECODE_TOKENS)
    assert module._tp_reduce.call_count == (tp > 1)


def test_tp_without_reducer_fails_explicitly(monkeypatch):
    monkeypatch.setattr(candidate, "_supports_device", lambda device: False)
    module = SimpleNamespace(
        params=SimpleNamespace(head_v_dim=16),
        num_v_heads=1,
        value_dim=16,
        norm_weight=torch.ones(16),
        in_proj_z=torch.eye(16),
        out_proj=torch.eye(16),
        compute_dtype=torch.float32,
        params_dtype=torch.float16,
        rms_norm_eps=1e-6,
        tp_size=2,
        _tp_reduce=None,
    )
    with pytest.raises(RuntimeError, match="all-reduce"):
        candidate.project_output(module, torch.ones(3, 16), torch.ones(3, 1, 16))


def test_gamma_cache_refreshes_after_weight_update_and_refuses_capture_allocation(monkeypatch):
    module = SimpleNamespace(norm_weight=torch.ones(128, dtype=torch.float16), compute_dtype=torch.float32)
    first = candidate.prepare_norm_weight(module)
    assert first.dtype == torch.float32 and candidate.prepare_norm_weight(module) is first
    module.norm_weight.add_(1)
    monkeypatch.setattr(candidate, "_is_capturing", lambda device: True)
    with pytest.raises(RuntimeError, match="warm up"):
        candidate.prepare_norm_weight(module)
    monkeypatch.setattr(candidate, "_is_capturing", lambda device: False)
    refreshed = candidate.prepare_norm_weight(module)
    assert refreshed is not first and torch.equal(refreshed, torch.full((128,), 2.0))
    with torch.inference_mode():
        module.norm_weight = torch.ones(128, dtype=torch.float16)
    candidate.prepare_norm_weight(module)
    assert getattr(module, candidate.CACHE_ATTRIBUTE)[1] is refreshed
