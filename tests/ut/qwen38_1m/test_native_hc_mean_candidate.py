# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tools.qwen4exp.resident_candidates import native_hc_mean as candidate
from vllm_ascend.models.qwen4_exp import model


@pytest.mark.parametrize(
    "combine,grad,prefill_only", [(True, False, False), (False, False, False), (True, True, False), (True, False, True)]
)
def test_native_mix_dispatch_and_fallback_preserve_residuals(monkeypatch, combine, grad, prefill_only):
    monkeypatch.setattr(candidate, "_supports_device", lambda device: True)
    monkeypatch.setattr(model, "_linear_operand_dtype", lambda device, weight, compute: weight)
    module = model._GatedResidual(
        hc_count=4,
        hidden_size=512,
        lowrank=16,
        eps=1e-6,
        params_dtype=torch.float16,
        compute_dtype=torch.float32,
        use_combine=combine,
    )
    hyper = torch.full((3, 2048), 0.25).half()
    before = hyper.clone()
    calls = []

    def op(normalized, gate):
        calls.append(normalized)
        return (normalized.view(3, 4, 512) * gate.view(3, 4, 512)).mean(1).half()

    with torch.set_grad_enabled(grad):
        expected, saved = module.mix(hyper)
        actual, residuals = candidate.make_mix(op, prefill_only=prefill_only)(module, hyper)
    assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    assert bool(calls) == (combine and not grad and not prefill_only)
    assert residuals[0] is hyper and torch.equal(residuals[1], saved[1])
    assert torch.equal(hyper, before)


def test_resident_candidate_requires_both_resources():
    with pytest.raises(KeyError, match=candidate.RESOURCE_NAME):
        candidate.replacements({})
    with pytest.raises(KeyError, match=candidate.RESIDUAL_RESOURCE):
        candidate.replacements({candidate.RESOURCE_NAME: object()})
