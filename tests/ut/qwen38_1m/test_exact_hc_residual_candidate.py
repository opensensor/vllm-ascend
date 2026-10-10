# SPDX-License-Identifier: Apache-2.0
"""Native HC residual must preserve the original four-row projection."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tools.qwen4exp.resident_candidates import exact_hc_residual as candidate
from tools.qwen4exp.resident_candidates import fused_hc_projection, shared_hc_operand


@pytest.mark.parametrize("rows", [128, 640])
def test_original_projection_and_rounding_are_preserved(monkeypatch, rows):
    generator = torch.Generator().manual_seed(310 + rows)
    hyper = torch.randn(rows, 2048, generator=generator).half()
    normalized = torch.randn(rows, 2048, generator=generator).half()
    block = torch.randn(rows, 512, generator=generator).half()
    weight = torch.randn(4, 2048, generator=generator).half()
    module = SimpleNamespace(
        use_combine=True,
        hc_count=4,
        hidden_size=512,
        params_dtype=torch.float16,
        compute_dtype=torch.float32,
        block_inject_weight=weight,
    )
    monkeypatch.setattr(candidate, "_supports_device", lambda device: True)
    linear = Mock(side_effect=lambda x, w, dtype: torch.nn.functional.linear(x, w))
    monkeypatch.setattr(candidate, "_linear", linear)
    calls = []

    def native(h, b, injection):
        calls.append((h, b, injection))
        return (h.float().view(rows, 4, 512) + b.float()[:, None] * injection.float()[:, :, None]).flatten(1).half()

    original = [v.clone() for v in (hyper, normalized, block, weight)]
    with torch.no_grad():
        result = candidate.make_combine(native)(module, block, (hyper, normalized))
    linear.assert_called_once_with(normalized, weight, torch.float32)
    assert calls[0][0] is hyper and calls[0][1] is block
    injection = 2.0 * torch.sigmoid(torch.nn.functional.linear(normalized, weight) / 4)
    assert calls[0][2].dtype == torch.float16
    torch.testing.assert_close(calls[0][2], injection, rtol=0, atol=0)
    torch.testing.assert_close(result, native(hyper, block, injection), rtol=0, atol=0)
    assert all(torch.equal(left, right) for left, right in zip(original, (hyper, normalized, block, weight)))


@pytest.mark.parametrize("reason", ["grad", "device", "width", "tokens", "streams", "dtype", "noncontiguous", "decode"])
def test_unsupported_cases_use_reference_without_native_launch(monkeypatch, reason):
    module = SimpleNamespace(
        use_combine=True, hc_count=4, hidden_size=512, params_dtype=torch.float16, compute_dtype=torch.float32
    )
    hyper = torch.zeros(128, 2048).half()
    block = torch.zeros(128, 512).half()
    monkeypatch.setattr(candidate, "_supports_device", lambda device: reason != "device")
    if reason == "width":
        module.hidden_size = 513
    if reason == "tokens":
        hyper, block = hyper[:0], block[:0]
    if reason == "streams":
        module.hc_count = 2
    if reason == "dtype":
        block = block.float()
    if reason == "noncontiguous":
        block = torch.zeros(128, 1024).half()[:, ::2]
    if reason == "decode":
        hyper, block = hyper[:18], block[:18]
    sentinel = object()
    fallback = Mock(return_value=sentinel)
    monkeypatch.setattr(fused_hc_projection, "combine", fallback)
    native = Mock(side_effect=AssertionError("unsupported native launch"))
    residual = (hyper, hyper)
    with torch.set_grad_enabled(reason == "grad"):
        assert candidate.make_combine(native)(module, block, residual) is sentinel
    fallback.assert_called_once_with(module, block, residual)
    native.assert_not_called()


def test_mix_releases_redundant_joined_cache_and_reuses_shared_operand(monkeypatch):
    module = SimpleNamespace(_qwen_hc_fused_projection_cache=object())
    values, sentinel = torch.zeros(128), object()
    mix = Mock(return_value=sentinel)
    monkeypatch.setattr(shared_hc_operand, "mix", mix)
    assert candidate.mix(module, values) is sentinel
    assert not hasattr(module, "_qwen_hc_fused_projection_cache")
    mix.assert_called_once_with(module, values)


def test_mix_cannot_release_joined_storage_during_capture(monkeypatch):
    module = SimpleNamespace(_qwen_hc_fused_projection_cache=object())
    monkeypatch.setattr(fused_hc_projection, "_is_capturing", lambda device: True)
    with pytest.raises(RuntimeError, match="after clearing old graphs"):
        candidate.mix(module, torch.zeros(128))
    assert hasattr(module, "_qwen_hc_fused_projection_cache")


@pytest.mark.parametrize("rows", [1, 3, 18, 127])
def test_small_batches_keep_original_mix(monkeypatch, rows):
    module, values, sentinel = object(), torch.zeros(rows, 1), object()
    original = Mock(return_value=sentinel)
    monkeypatch.setattr(candidate.release_hc_projection, "mix", original)
    monkeypatch.setattr(shared_hc_operand, "mix", Mock(side_effect=AssertionError("small batch must retain baseline")))
    assert candidate.mix(module, values) is sentinel
    original.assert_called_once_with(module, values)


def test_replacements_require_the_versioned_residual_resource():
    with pytest.raises(KeyError, match="hc_residual_v3"):
        candidate.replacements({"hc_residual_v1": object()})
    replacements = candidate.replacements({"hc_residual_v3": object()})
    assert replacements["vllm_ascend.models.qwen4_exp.model:_GatedResidual.mix"] is candidate.mix
    assert callable(replacements["vllm_ascend.models.qwen4_exp.model:_GatedResidual.combine"])
