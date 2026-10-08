# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Joint hyperconnection projection math, cache lifetime, and resident rollback."""

from pathlib import Path
from uuid import uuid4

import pytest
import torch

from tools.glm_perf.resident_control import PatchSession
from tools.qwen4exp.resident_candidates import fused_hc_projection as candidate
from tools.qwen4exp.resident_candidates import release_hc_projection as cleanup
from vllm_ascend.models.qwen4_exp import model


def make_module(*, use_combine=True):
    torch.manual_seed(1065)
    module = model._GatedResidual(
        hc_count=2,
        hidden_size=16,
        lowrank=8,
        eps=1e-6,
        params_dtype=torch.float16,
        compute_dtype=torch.float32,
        use_combine=use_combine,
    )
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)
    return module


def enable_host_fusion(monkeypatch):
    # Test cache/routing logic and FP16 rounding on CPU. This does not qualify
    # CANN's different GEMM shape/format; the 310P probe must pass separately.
    monkeypatch.setattr(candidate, "_supports_device", lambda device: True)
    monkeypatch.setattr(model, "_linear_operand_dtype", lambda device, weight, compute: weight)


@pytest.mark.parametrize("tokens", [0, 1, 3, 6, 17])
@torch.no_grad()
def test_fused_outputs_and_compact_residual(monkeypatch, tokens):
    enable_host_fusion(monkeypatch)
    module = make_module()
    values = torch.randn(tokens, 32).half()
    block = torch.randn(tokens, 16).half()
    expected, original_residual = module.mix(values)
    actual, residual = candidate.mix(module, values)
    assert torch.equal(actual, expected)
    assert torch.equal(candidate.combine(module, block, residual), module.combine(block, original_residual))
    assert residual.hyper_input is values
    assert residual.injection.shape == (tokens, 2)
    assert residual.injection.dtype == torch.float16
    assert residual.injection.untyped_storage().nbytes() == tokens * 2 * 2
    joined = candidate.prepare_projection(module)
    assert joined.shape == (16, 32)
    assert torch.equal(joined[:8], module.input_mix_weight_down)
    assert torch.equal(joined[8:10], module.block_inject_weight)
    assert torch.count_nonzero(joined[10:]) == 0


@torch.no_grad()
def test_mix_combine_removes_one_linear_call(monkeypatch):
    enable_host_fusion(monkeypatch)
    module = make_module()
    values, block = torch.randn(3, 32).half(), torch.randn(3, 16).half()
    original_linear = candidate.F.linear
    shapes = []

    def record(value, weight):
        shapes.append(tuple(weight.shape))
        return original_linear(value, weight)

    monkeypatch.setattr(candidate.F, "linear", record)
    mixed, residual = candidate.mix(module, values)
    candidate.combine(module, block, residual)
    assert mixed.shape == (3, 16)
    assert shapes == [(16, 32), (32, 8)]


@pytest.mark.parametrize("mutation", ["down_inplace", "injection_inplace", "down_storage", "injection_parameter"])
@torch.no_grad()
def test_cache_rebuilds_after_weight_changes(monkeypatch, mutation):
    enable_host_fusion(monkeypatch)
    module = make_module()
    down_pointer = module.input_mix_weight_down.data_ptr()
    first = candidate.prepare_projection(module)
    assert candidate.prepare_projection(module) is first
    assert module.input_mix_weight_down.data_ptr() == down_pointer
    if mutation == "down_inplace":
        module.input_mix_weight_down.add_(0.5)
    elif mutation == "injection_inplace":
        module.block_inject_weight.mul_(2)
    elif mutation == "down_storage":
        module.input_mix_weight_down.data = module.input_mix_weight_down.data.clone() + 0.5
    else:
        module.block_inject_weight = torch.nn.Parameter(module.block_inject_weight.clone() * 2)
    second = candidate.prepare_projection(module)
    assert second is not first
    assert torch.equal(second[:8], module.input_mix_weight_down)
    assert torch.equal(second[8:10], module.block_inject_weight)


@torch.no_grad()
def test_capture_requires_valid_prepared_cache(monkeypatch):
    enable_host_fusion(monkeypatch)
    module = make_module()
    monkeypatch.setattr(candidate, "_is_capturing", lambda device: True)
    with pytest.raises(RuntimeError, match="before graph capture"):
        candidate.prepare_projection(module)
    monkeypatch.setattr(candidate, "_is_capturing", lambda device: False)
    prepared = candidate.prepare_projection(module)
    monkeypatch.setattr(candidate, "_is_capturing", lambda device: True)
    assert candidate.prepare_projection(module) is prepared
    module.block_inject_weight.add_(0.5)
    with pytest.raises(RuntimeError, match="after weight changes"):
        candidate.prepare_projection(module)


@pytest.mark.parametrize("fallback", ["cpu", "dtype", "compute_dtype", "mix_only", "inference_parameter"])
@torch.no_grad()
def test_unsupported_cases_keep_original_math(monkeypatch, fallback):
    module = make_module(use_combine=fallback != "mix_only")
    if fallback != "cpu":
        enable_host_fusion(monkeypatch)
    if fallback == "dtype":
        module.block_inject_weight.data = module.block_inject_weight.data.float()
    elif fallback == "compute_dtype":
        module.compute_dtype = torch.float16
    elif fallback == "inference_parameter":
        with torch.inference_mode():
            module.input_mix_weight_down = torch.nn.Parameter(module.input_mix_weight_down.clone())
    values, block = torch.randn(3, 32).half(), torch.randn(3, 16).half()
    expected, expected_residual = module.mix(values)
    actual, residual = candidate.mix(module, values)
    assert torch.equal(actual, expected)
    assert isinstance(residual, tuple)
    assert not hasattr(module, candidate.CACHE_ATTRIBUTE)
    if module.use_combine:
        assert torch.equal(candidate.combine(module, block, residual), module.combine(block, expected_residual))
    else:
        with pytest.raises(RuntimeError, match="combine disabled"):
            candidate.combine(module, block, residual)


def test_training_retains_gradients_and_original_residual(monkeypatch):
    enable_host_fusion(monkeypatch)
    module = make_module()
    values = torch.randn(3, 32).half().requires_grad_()
    block = torch.randn(3, 16).half().requires_grad_()
    parameters = [values, block, *module.parameters()]
    expected, original_residual = module.mix(values)
    expected_combined = module.combine(block, original_residual)
    expected_gradients = torch.autograd.grad(expected.float().sum() + expected_combined.float().sum(), parameters)
    actual, residual = candidate.mix(module, values)
    actual_combined = candidate.combine(module, block, residual)
    actual_gradients = torch.autograd.grad(actual.float().sum() + actual_combined.float().sum(), parameters)
    assert isinstance(residual, tuple)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_combined, expected_combined)
    assert all(torch.equal(left, right) for left, right in zip(expected_gradients, actual_gradients))
    assert not hasattr(module, candidate.CACHE_ATTRIBUTE)


@torch.no_grad()
def test_release_allows_cache_rebuild(monkeypatch):
    enable_host_fusion(monkeypatch)
    module = make_module()
    first = candidate.prepare_projection(module)
    candidate.release_projection(module)
    candidate.release_projection(module)
    assert not hasattr(module, candidate.CACHE_ATTRIBUTE)
    assert candidate.prepare_projection(module) is not first


@torch.no_grad()
def test_cleanup_restores_original_residual_and_releases_cache(monkeypatch):
    enable_host_fusion(monkeypatch)
    module = make_module()
    values, block = torch.randn(3, 32).half(), torch.randn(3, 16).half()
    expected, expected_residual = module.mix(values)
    candidate.prepare_projection(module)
    actual, residual = cleanup.mix(module, values)
    assert not hasattr(module, candidate.CACHE_ATTRIBUTE)
    assert torch.equal(actual, expected)
    assert isinstance(residual, tuple)
    assert residual[1].dtype == torch.float32
    assert torch.equal(module.combine(block, residual), module.combine(block, expected_residual))


@torch.no_grad()
def test_cleanup_rejects_freeing_graph_weights_during_capture(monkeypatch):
    enable_host_fusion(monkeypatch)
    module = make_module()
    candidate.prepare_projection(module)
    monkeypatch.setattr(cleanup, "_is_capturing", lambda device: True)
    with pytest.raises(RuntimeError, match="after clearing old graphs"):
        cleanup.mix(module, torch.randn(3, 32).half())


@torch.no_grad()
def test_resident_reprepare_and_restore(monkeypatch):
    monkeypatch.setattr(model, "_linear_operand_dtype", lambda device, weight, compute: weight)
    module = make_module()
    values, block = torch.randn(3, 32).half(), torch.randn(3, 16).half()
    original_mix, original_combine = model._GatedResidual.mix, model._GatedResidual.combine
    expected, original_residual = module.mix(values)
    expected_combined = module.combine(block, original_residual)
    source = Path(candidate.__file__).read_text()
    session = PatchSession()
    try:
        for revision in range(2):
            generation = uuid4().hex
            session.prepare(
                {"generation": generation, "candidate": "fused_hc_projection", "source": source + f"\n# {revision}\n"}
            )
            session.apply(generation)
            # The exec module has its own globals and is not registered in
            # sys.modules. It must still support cache reuse and pair its mix
            # and combine functions after a second preparation.
            monkeypatch.setitem(model._GatedResidual.mix.__globals__, "_supports_device", lambda device: True)
            actual, residual = module.mix(values)
            assert torch.equal(actual, expected)
            assert torch.equal(module.combine(block, residual), expected_combined)
        generation = uuid4().hex
        session.prepare(
            {
                "generation": generation,
                "candidate": "release_hc_projection",
                "source": Path(cleanup.__file__).read_text(),
            }
        )
        session.apply(generation)
        actual, residual = module.mix(values)
        assert not hasattr(module, candidate.CACHE_ATTRIBUTE)
        assert model._GatedResidual.combine is original_combine
        assert torch.equal(actual, expected)
        assert torch.equal(module.combine(block, residual), expected_combined)
        generation = uuid4().hex
        session.prepare({"generation": generation, "candidate": "baseline", "source": ""})
        session.apply(generation)
        assert model._GatedResidual.mix is original_mix
        assert model._GatedResidual.combine is original_combine
    finally:
        model._GatedResidual.mix, model._GatedResidual.combine = original_mix, original_combine
        candidate.release_projection(module)
