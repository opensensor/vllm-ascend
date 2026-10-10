# SPDX-License-Identifier: Apache-2.0
"""The joined projection residual must survive native/fallback dispatch."""

from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp.resident_candidates import fused_hc_projection as projection
from tools.qwen4exp.resident_candidates import fused_hc_residual as candidate


@pytest.mark.parametrize("rows", [3, 18, 2560])
@pytest.mark.parametrize("injection_dtype", [torch.float16, torch.float32])
def test_joined_residual_native_math_preserves_inputs(monkeypatch, rows, injection_dtype):
    monkeypatch.setattr(candidate, "_supports_device", lambda device: True)
    module = SimpleNamespace(
        hc_count=4, hidden_size=512, use_combine=True, params_dtype=torch.float16, compute_dtype=torch.float32
    )
    generator = torch.Generator().manual_seed(rows)
    hyper = torch.randn(rows, 2048, generator=generator).half()
    block = torch.randn(rows, 512, generator=generator).half()
    injection = (2 * torch.rand(rows, 4, generator=generator)).to(injection_dtype)
    residual = projection.ProjectionResidual(hyper, injection)
    before = [value.clone() for value in (hyper, block, injection)]
    calls = []

    def op(h, b, i):
        calls.append((h, b, i))
        return (h.float().view(rows, 4, 512) + b.float()[:, None] * i[:, :, None]).flatten(1).half()

    with torch.inference_mode():
        expected = projection.combine(module, block, residual)
        actual = candidate.make_combine(op)(module, block, residual)
    assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    assert len(calls) == 1 and all(a is b for a, b in zip(calls[0], (hyper, block, injection)))
    assert all(torch.equal(a, b) for a, b in zip((hyper, block, injection), before))


@pytest.mark.parametrize("reason", ["cpu", "grad", "width", "strided"])
def test_ineligible_joined_residual_uses_reference_without_native_launch(monkeypatch, reason):
    monkeypatch.setattr(candidate, "_supports_device", lambda device: reason != "cpu")
    width = 32 if reason == "width" else 512
    module = SimpleNamespace(
        hc_count=4, hidden_size=width, use_combine=True, params_dtype=torch.float16, compute_dtype=torch.float32
    )
    hyper = torch.full((3, 4 * width), 0.25, dtype=torch.float16)
    block = torch.full((3, width), 0.5, dtype=torch.float16)
    injection = torch.ones(3, 4)
    if reason == "strided":
        injection = torch.ones(3, 8)[:, ::2]
    residual = projection.ProjectionResidual(hyper, injection)

    def forbidden(*args):
        pytest.fail("ineligible residual dispatched to the native kernel")

    with torch.set_grad_enabled(reason == "grad"):
        expected = projection.combine(module, block, residual)
        actual = candidate.make_combine(forbidden)(module, block, residual)
    assert torch.equal(expected, actual)


def test_factory_requires_admitted_native_resource():
    with pytest.raises(KeyError, match="hc_residual_v1"):
        candidate.replacements({})
    values = candidate.replacements({"hc_residual_v1": object()})
    assert values["vllm_ascend.models.qwen4_exp.model:_GatedResidual.mix"] is projection.mix
    assert callable(values["vllm_ascend.models.qwen4_exp.model:_GatedResidual.combine"])
