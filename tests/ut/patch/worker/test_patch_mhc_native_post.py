# SPDX-License-Identifier: Apache-2.0
"""Host dispatch and pre/post ordering without loading an NPU extension."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_ascend.models.glm5next.ops.mhc_native import native_mhc_post_enabled, use_native_mhc_post


def inputs(tokens=640, width=32):
    return (
        torch.empty(tokens, width, dtype=torch.float16),
        torch.empty(tokens, 4, width),
        torch.empty(tokens, 4, 1),
        torch.empty(tokens, 4, 4),
    )


def config(**overrides):
    return SimpleNamespace(**dict(dict(use_310p_native_mhc_post=True, use_310p_fp16_mhc_state=True), **overrides))


@pytest.mark.parametrize(
    "tokens,expected", [(1, False), (4, False), (639, False), (640, True), (1280, True), (32768, True), (32769, False)]
)
def test_prefill_dispatch(tokens, expected):
    assert use_native_mhc_post(config(), *inputs(tokens)) == expected


@pytest.mark.parametrize(
    "width,expected", [(0, False), (31, False), (32, True), (1056, True), (16384, True), (16416, False)]
)
def test_width_boundaries(width, expected):
    # Meta tensors allow checking large allocations without reserving RAM.
    with torch.device("meta"):
        assert use_native_mhc_post(config(), *inputs(width=width)) == expected


@pytest.mark.parametrize("field", ["use_310p_native_mhc_post", "use_310p_fp16_mhc_state"])
def test_disabled_state(field):
    assert not use_native_mhc_post(config(**{field: False}), *inputs())


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("problem", ["dtype", "shape", "layout", "device"])
def test_unsupported_inputs_fall_back(index, problem):
    args = list(inputs())
    value = args[index]
    if problem == "dtype":
        args[index] = value.double()
    elif problem == "shape":
        args[index] = value.unsqueeze(0)
    elif problem == "device":
        args[index] = value.to("meta")
    else:
        # A stride-two view with the same logical shape (including width=1).
        backing = torch.empty((*value.shape[:-1], value.shape[-1] * 2), dtype=value.dtype)
        args[index] = backing[..., ::2]
    assert not use_native_mhc_post(config(), *args)


@pytest.fixture
def patch(monkeypatch):
    kernels = ModuleType("vllm.model_executor.kernels.mhc.torch")
    kernels.mhc_post_torch = lambda *args: None
    kernels.mhc_pre_torch = lambda *args: None
    layers = ModuleType("vllm.model_executor.layers")
    layers.mhc = SimpleNamespace(MHCPreOp=type("Pre", (), {}), MHCFusedPostPreOp=type("Fused", (), {}))
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    monkeypatch.setitem(sys.modules, layers.__name__, layers)
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/worker/patch_mhc_norm.py"
    spec = importlib.util.spec_from_file_location("native_mhc_patch_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invoke(patch, op, args):
    return patch._mhc_fused_post_pre_npu(op, *args, None, None, None, None, None, None, None, None)


def test_native_output_reaches_pre_without_second_round(patch, monkeypatch):
    args = inputs()
    rounded = torch.full_like(args[1], 1.0078125)
    calls = []

    def native(*values):
        assert all(a is b for a, b in zip(values, args, strict=True))
        calls.append("native")
        return rounded

    def pre(residual, *unused):
        assert residual is rounded  # no second cast pair or intermediate copy
        calls.append("pre")
        return torch.ones(640, 4, 1), torch.ones(640, 4, 4), torch.ones(640, 32)

    monkeypatch.setattr(torch.ops._C_ascend, "npu_glm_mhc_post_310", native, raising=False)
    monkeypatch.setattr(patch, "_mhc_pre_torch", pre)
    output = invoke(patch, config(), args)
    assert output[0] is rounded
    assert calls == ["native", "pre"]


def test_missing_kernel_is_not_silently_timed_as_reference(patch, monkeypatch):
    def unavailable(*args):
        raise RuntimeError("native mHC operator unavailable")

    monkeypatch.setattr(torch.ops._C_ascend, "npu_glm_mhc_post_310", unavailable, raising=False)
    with pytest.raises(RuntimeError, match="unavailable"):
        invoke(patch, config(), inputs())


def test_decode_keeps_reference_rounding(patch, monkeypatch):
    args = inputs(tokens=4)
    raw = torch.full_like(args[1], 1.0006)
    expected = raw.half().float()
    monkeypatch.setattr(patch, "_mhc_post_torch", lambda *unused: raw)

    def pre(residual, *unused):
        torch.testing.assert_close(residual, expected, rtol=0, atol=0)
        return torch.ones(4, 4, 1), torch.ones(4, 4, 4), torch.ones(4, 32)

    monkeypatch.setattr(patch, "_mhc_pre_torch", pre)
    torch.testing.assert_close(invoke(patch, config(), args)[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "native,fp16,streaming,expected",
    [
        (False, False, False, False),
        (False, True, True, False),
        (True, True, False, True),
        (True, False, False, "requires"),
        (True, True, True, "only one"),
    ],
)
def test_experiment_configuration(native, fp16, streaming, expected):
    cfg = SimpleNamespace(
        ascend_glm_native_mhc_post=native, ascend_glm_mhc_fp16_state=fp16, ascend_glm_prefill_mhc_post=streaming
    )
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected):
            native_mhc_post_enabled(cfg)
    else:
        assert native_mhc_post_enabled(cfg) is expected


def test_experiment_defaults_off():
    assert native_mhc_post_enabled(SimpleNamespace()) is False
