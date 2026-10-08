# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tools.qwen4exp.resident_candidates import prefill_hc_addcmul as candidate
from vllm_ascend.models.qwen4_exp import model


@pytest.mark.parametrize(
    "tokens,dtype,grad",
    [
        (3, torch.float16, False),
        (2560, torch.float32, False),
        (2560, torch.float16, True),
        (2560, torch.float16, False),
    ],
)
def test_prefill_fusion_preserves_inputs_and_avoids_aliasing(monkeypatch, tokens, dtype, grad):
    monkeypatch.setattr(candidate, "_supports_device", lambda device: True)
    monkeypatch.setattr(model, "_linear_operand_dtype", lambda device, weight, compute: weight)
    module = model._GatedResidual(
        hc_count=2, hidden_size=16, lowrank=8, eps=1e-6, params_dtype=torch.float16, compute_dtype=torch.float32
    )
    with torch.no_grad():
        module.block_inject_weight.zero_()
    # Unit injection and powers of two make the host arithmetic exact. The
    # real-weight NPU probe exercises nonzero injection and rounding separately.
    hyper = torch.arange(tokens * 32).remainder(16).reshape(tokens, 32).to(dtype) / 4
    normalized = hyper.float()
    block = torch.ones(tokens, 16, dtype=torch.float16) / 2
    before = [tensor.clone() for tensor in (hyper, normalized, block)]
    calls = []
    original = torch.Tensor.addcmul_

    def record(self, *args, **kwargs):
        calls.append(self.data_ptr())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "addcmul_", record)
    with torch.set_grad_enabled(grad):
        expected = module.combine(block, (hyper, normalized))
        actual = candidate.combine(module, block, (hyper, normalized))
    assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    assert all(torch.equal(a, b) for a, b in zip((hyper, normalized, block), before))
    assert bool(calls) == (tokens == 2560 and dtype == torch.float16 and not grad)
    assert all(pointer != hyper.data_ptr() for pointer in calls)


def test_mix_only_module_rejects_combine():
    module = model._GatedResidual(
        hc_count=2,
        hidden_size=16,
        lowrank=8,
        eps=1e-6,
        params_dtype=torch.float16,
        compute_dtype=torch.float32,
        use_combine=False,
    )
    with pytest.raises(RuntimeError, match="combine disabled"):
        candidate.combine(module, None, None)
