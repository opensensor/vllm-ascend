# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tools.qwen4exp.resident_candidates import padded_hc_injection as candidate
from tools.qwen4exp.resident_candidates import release_padded_injection as cleanup
from vllm_ascend.models.qwen4_exp import model


def make_module():
    module = model._GatedResidual(
        hc_count=2, hidden_size=16, lowrank=8, eps=1e-6, params_dtype=torch.float16, compute_dtype=torch.float32
    )
    with torch.no_grad():
        module.block_inject_weight.copy_(torch.arange(64).remainder(4).reshape(2, 32).half() / 8)
    return module


def host_backend(monkeypatch):
    monkeypatch.setattr(candidate, "_supports_device", lambda device: True)
    monkeypatch.setattr(model, "_linear_operand_dtype", lambda device, weight, compute: weight)


@pytest.mark.parametrize("tokens", [3, 6, 2560])
@torch.no_grad()
def test_cached_and_composed_outputs_preserve_checkpoint_and_inputs(monkeypatch, tokens):
    host_backend(monkeypatch)
    module = make_module()
    generator = torch.Generator().manual_seed(42)
    for _ in range(3):
        hyper = torch.randint(-4, 5, (tokens, 32), generator=generator).half() / 4
        normalized = hyper.float()
        block = torch.ones(tokens, 16).half() / 4
        saved = [tensor.clone() for tensor in (hyper, normalized, block, module.block_inject_weight)]
        expected = module.combine(block, (hyper, normalized))
        for function in (candidate.combine, candidate.combine_with_prefill_addcmul):
            actual = function(module, block, (hyper, normalized))
            assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
        assert all(torch.equal(a, b) for a, b in zip((hyper, normalized, block, module.block_inject_weight), saved))
    padded = candidate.prepare_injection(module)
    assert padded.shape == (16, 32)
    assert torch.equal(padded[:2], module.block_inject_weight)
    assert torch.count_nonzero(padded[2:]) == 0
    assert candidate.prepare_injection(module) is padded


@torch.no_grad()
def test_inplace_parameter_update_rebuilds_cache(monkeypatch):
    host_backend(monkeypatch)
    module = make_module()
    first = candidate.prepare_injection(module)
    module.block_inject_weight.add_(0.125)
    second = candidate.prepare_injection(module)
    assert first is not second
    assert torch.equal(second[:2], module.block_inject_weight)


@torch.no_grad()
def test_capture_cache_miss_and_cleanup_are_explicit(monkeypatch):
    host_backend(monkeypatch)
    module = make_module()
    monkeypatch.setattr(candidate, "_is_capturing", lambda device: True)
    with pytest.raises(RuntimeError, match="warm up"):
        candidate.prepare_injection(module)
    monkeypatch.setattr(candidate, "_is_capturing", lambda device: False)
    candidate.prepare_injection(module)
    hyper = torch.zeros(3, 32).half()
    block = torch.zeros(3, 16).half()
    monkeypatch.setattr(cleanup, "_is_capturing", lambda device: True)
    with pytest.raises(RuntimeError, match="before graph capture"):
        cleanup.combine(module, block, (hyper, hyper.float()))
    monkeypatch.setattr(cleanup, "_is_capturing", lambda device: False)
    expected = module.combine(block, (hyper, hyper.float()))
    assert torch.equal(cleanup.combine(module, block, (hyper, hyper.float())), expected)
    assert not hasattr(module, candidate.CACHE_ATTRIBUTE)


def test_training_and_inference_parameters_do_not_cache(monkeypatch):
    host_backend(monkeypatch)
    module = make_module()
    with torch.enable_grad():
        assert candidate.prepare_injection(module) is None
    with torch.inference_mode():
        inference_module = make_module()
        assert candidate.prepare_injection(inference_module) is None
