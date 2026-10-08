# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tools.qwen4exp.resident_candidates import native_hc_residual as candidate
from tools.qwen4exp.resident_candidates import padded_hc_injection as padded
from vllm_ascend.models.qwen4_exp import model


@pytest.mark.parametrize(
    "tokens,decode_only,grad",
    [(3, False, False), (6, False, False), (2560, False, False), (2560, True, False), (3, False, True)],
)
def test_native_dispatch_and_fallback_preserve_inputs(monkeypatch, tokens, decode_only, grad):
    monkeypatch.setattr(candidate, "_supports_device", lambda device: True)
    monkeypatch.setattr(model, "_linear_operand_dtype", lambda device, weight, compute: weight)
    module = model._GatedResidual(
        hc_count=4, hidden_size=512, lowrank=16, eps=1e-6, params_dtype=torch.float16, compute_dtype=torch.float32
    )
    hyper = torch.full((tokens, 2048), 0.25).half()
    normalized = hyper.float()
    block = torch.full((tokens, 512), 0.5).half()
    calls = []

    def op(hyper, block, injection):
        calls.append(injection)
        return (hyper.float().view(tokens, 4, 512) + block.float()[:, None] * injection[:, :, None]).flatten(1).half()

    before = [t.clone() for t in (hyper, normalized, block, module.block_inject_weight)]
    with torch.set_grad_enabled(grad):
        expected = padded.combine_with_prefill_addcmul(module, block, (hyper, normalized))
        actual = candidate.make_combine(op, decode_only=decode_only)(module, block, (hyper, normalized))
    assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    assert bool(calls) == (not grad and (not decode_only or tokens <= 6))
    assert all(torch.equal(a, b) for a, b in zip((hyper, normalized, block, module.block_inject_weight), before))


def test_resource_required_and_mix_only_rejected():
    with pytest.raises(KeyError, match=candidate.RESOURCE_NAME):
        candidate.replacements({})
    module = model._GatedResidual(
        hc_count=4,
        hidden_size=512,
        lowrank=16,
        eps=1e-6,
        params_dtype=torch.float16,
        compute_dtype=torch.float32,
        use_combine=False,
    )
    with pytest.raises(RuntimeError, match="combine disabled"):
        candidate.make_combine(None)(module, None, None)
