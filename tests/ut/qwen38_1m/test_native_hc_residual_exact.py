# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp.resident_candidates import native_hc_residual_exact as candidate


@pytest.fixture
def fixture(monkeypatch):
    monkeypatch.setattr(candidate, "_supports_device", lambda device: device.type == "cpu")
    torch.manual_seed(7)
    model = SimpleNamespace(
        use_combine=True,
        hc_count=4,
        hidden_size=512,
        params_dtype=torch.float16,
        compute_dtype=torch.float32,
        block_inject_weight=torch.randn(4, 2048, dtype=torch.float16),
    )
    inputs = (torch.randn(3, 2048, dtype=torch.float16), torch.randn(3, 2048))
    block = torch.randn(3, 512, dtype=torch.float16)
    calls = []

    def linear(x, weight, compute_dtype):
        calls.append((x.shape, weight.shape, compute_dtype))
        return torch.nn.functional.linear(x.half(), weight)

    def reference(hyper, output, injection):
        return (
            (hyper.float().view(3, 4, 512) + output.float()[:, None] * injection.float()[:, :, None]).flatten(1).half()
        )

    def fallback(self, output, residuals):
        return reference(
            residuals[0],
            output,
            2.0 * torch.sigmoid(linear(residuals[1], self.block_inject_weight, self.compute_dtype) / 4),
        )

    return model, inputs, block, calls, linear, reference, fallback


def test_native_fusion_preserves_original_projection_shape_and_half_rounding(fixture):
    model, residuals, block, calls, linear, reference, fallback = fixture
    native_calls = []

    def op(hyper, output, injection):
        native_calls.append(injection)
        return reference(hyper, output, injection)

    combine = candidate.make_combine(op, linear=linear, fallback=fallback)
    with torch.inference_mode():
        expected = fallback(model, block, residuals)
        actual = combine(model, block, residuals)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert calls == [(torch.Size([3, 2048]), torch.Size([4, 2048]), torch.float32)] * 2
    assert len(native_calls) == 1 and native_calls[0].dtype == torch.float16


@pytest.mark.parametrize("change", [{"compute_dtype": torch.float64}, {"hc_count": 2}, {"hidden_size": 513}])
def test_unsupported_geometry_or_precision_keeps_original_method(fixture, change):
    model, residuals, block, calls, linear, reference, fallback = fixture
    for name, value in change.items():
        setattr(model, name, value)
    sentinel = object()
    combine = candidate.make_combine(
        lambda *args: pytest.fail("native path must not run"), linear=linear, fallback=lambda *args: sentinel
    )
    with torch.inference_mode():
        assert combine(model, block, residuals) is sentinel


def test_training_uses_original_method(fixture):
    model, residuals, block, calls, linear, reference, fallback = fixture
    combine = candidate.make_combine(
        lambda *args: pytest.fail("native path must not run"), linear=linear, fallback=fallback
    )
    assert torch.equal(
        combine(model, block, residuals).view(torch.int16), fallback(model, block, residuals).view(torch.int16)
    )
