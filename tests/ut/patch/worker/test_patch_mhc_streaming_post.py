# SPDX-License-Identifier: Apache-2.0
"""CPU math and dispatch checks without importing the NPU worker package."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_ascend.models.glm5next.ops.mhc_ops import mhc_post_streaming


def reference_post(x, residual, post_mix, comb_mix):
    mixed = torch.einsum("...ij,...ih->...jh", comb_mix.float(), residual.float())
    return (mixed + post_mix.float() * x.unsqueeze(-2).float()).to(residual.dtype)


@pytest.mark.parametrize("leading", [(1,), (4,), (640,), (2, 3)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_streaming_math_and_input_preservation(leading, dtype):
    generator = torch.Generator().manual_seed(310)
    shapes = [(*leading, 32), (*leading, 4, 32), (*leading, 4, 1), (*leading, 4, 4)]
    inputs = [torch.randn(shape, generator=generator).half().to(dtype) for shape in shapes]
    before = [value.clone() for value in inputs]
    actual = mhc_post_streaming(*inputs)
    assert actual.dtype == dtype and actual.shape == inputs[1].shape
    torch.testing.assert_close(actual, reference_post(*inputs), rtol=1e-3, atol=2e-6)
    for value, saved in zip(inputs, before, strict=True):
        assert torch.equal(value, saved)


def test_streaming_rejects_empty_streams():
    with pytest.raises(ValueError, match="at least one"):
        mhc_post_streaming(torch.ones(1, 32), torch.empty(1, 0, 32), torch.empty(1, 0, 1), torch.empty(1, 0, 0))


@pytest.fixture
def mhc_patch(monkeypatch):
    # Load the real patch against narrow upstream stubs. No device discovery,
    # native extensions, or NPU initialization belongs in these host checks.
    kernels = ModuleType("vllm.model_executor.kernels.mhc.torch")
    kernels.mhc_post_torch = reference_post
    kernels.mhc_pre_torch = lambda *args: None
    layers = ModuleType("vllm.model_executor.layers")
    layers.mhc = SimpleNamespace(MHCPreOp=type("Pre", (), {}), MHCFusedPostPreOp=type("Fused", (), {}))
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    monkeypatch.setitem(sys.modules, layers.__name__, layers)
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/worker/patch_mhc_norm.py"
    spec = importlib.util.spec_from_file_location("mhc_streaming_patch_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "tokens,enabled,fp16,streams,expected",
    [
        (1, True, True, 4, "reference"),
        (4, True, True, 4, "reference"),
        (639, True, True, 4, "reference"),
        (640, True, True, 4, "streaming"),
        (1280, True, True, 4, "streaming"),
        (640, False, True, 4, "reference"),
        (640, True, False, 4, "reference"),
        (640, True, True, 3, "reference"),
    ],
)
def test_dispatch_and_round_before_pre(mhc_patch, monkeypatch, tokens, enabled, fp16, streams, expected):
    calls = []
    after_post = torch.full((tokens, streams, 8), 1.006)

    def mixer(name):
        def run(*args):
            calls.append(name)
            return after_post

        return run

    monkeypatch.setattr(mhc_patch, "_mhc_post_torch", mixer("reference"))
    monkeypatch.setattr(mhc_patch, "mhc_post_streaming", mixer("streaming"))
    rounded = mhc_patch._round_mhc_state(after_post, use_fp16=fp16)

    def pre(residual, *args):
        torch.testing.assert_close(residual, rounded, rtol=0, atol=0)
        return residual[..., :1], torch.ones(tokens, streams, streams), residual[:, 0, :]

    monkeypatch.setattr(mhc_patch, "_mhc_pre_torch", pre)
    op = SimpleNamespace(use_310p_fp16_mhc_state=fp16)
    if enabled:
        op.use_310p_prefill_mhc_post = True
    output = mhc_patch._mhc_fused_post_pre_npu(
        op, torch.empty(tokens, 8), after_post, None, None, None, None, None, None, None, None, None, None
    )
    assert calls == [expected]
    torch.testing.assert_close(output[0], rounded, rtol=0, atol=0)
