# SPDX-License-Identifier: Apache-2.0
"""FP32 expert outputs stay FP32 when selecting the native post mixer."""

import json

import pytest
import torch

from tools.glm_perf.mhc_post_native import supported_inputs, wrap_post


def inputs(rows=2, dtype=torch.float32, width=4096):
    return (
        torch.empty(rows, width, dtype=dtype),
        torch.empty(rows, 4, width),
        torch.empty(rows, 4, 1),
        torch.empty(rows, 4, 4),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("rows", [1, 2, 8, 640])
def test_qualified_precision_and_rows(dtype, rows):
    assert supported_inputs(*inputs(rows, dtype))


@pytest.mark.parametrize("rows", [0, 9, 1280])
def test_other_rows_use_original(rows):
    calls = []
    old = lambda *args: calls.append(args) or "original"
    wrapped = wrap_post(old, lambda *args: pytest.fail("unqualified native dispatch"))
    assert wrapped(*inputs(rows)) == "original"
    assert len(calls) == 1


def test_fp32_expert_input_is_forwarded_without_conversion():
    args = inputs()
    calls = []
    native = lambda *values: calls.append(values) or "native"
    wrapped = wrap_post(lambda *values: pytest.fail("qualified fallback"), native)
    assert wrapped(*args) == "native"
    assert calls[0][0] is args[0]
    assert calls[0][0].dtype == torch.float32


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("problem", ["dtype", "shape", "layout", "device"])
def test_unqualified_tensor_contract(index, problem):
    args = list(inputs())
    value = args[index]
    if problem == "dtype":
        args[index] = value.double()
    elif problem == "shape":
        args[index] = value.unsqueeze(0)
    elif problem == "device":
        args[index] = value.to("meta")
    else:
        backing = torch.empty((*value.shape[:-1], value.shape[-1] * 2), dtype=value.dtype)
        args[index] = backing[..., ::2]
    assert not supported_inputs(*args)


def test_other_model_width_rejected():
    assert not supported_inputs(*inputs(width=2048))


def test_reapplying_wrapper_retains_unqualified_original():
    old = lambda *args: "original"
    wrapped = wrap_post(wrap_post(old, lambda *args: "first"), lambda *args: "second")
    assert wrapped(*inputs()) == "second"
    assert wrapped(*inputs(9)) == "original"


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_finish_only_retains_einsum_reduction_and_input_precision(dtype):
    from tools.glm_perf.mhc_post_native import NativeMhcPost

    generator = torch.Generator().manual_seed(7305)
    x = torch.randn(2, 4096, dtype=dtype, generator=generator)
    residual = torch.randn(2, 4, 4096, generator=generator)
    post = torch.randn(2, 4, 1, generator=generator)
    comb = torch.randn(2, 4, 4, generator=generator)
    native = NativeMhcPost.__new__(NativeMhcPost)
    native.finish_only = True
    native.configs = {(x.shape, x.device): torch.tensor(x.shape)}
    native.kernels = {dtype: "kernel"}
    calls = []
    native.launch = lambda *args: calls.append(args)
    native(x, residual, post, comb)
    kernel, arguments, blocks = calls[0]
    assert kernel == "kernel" and blocks == 8
    assert arguments[0] is x and arguments[0].dtype == dtype
    assert torch.equal(arguments[1], torch.einsum("nij,nih->njh", comb, residual))
    assert arguments[1].is_contiguous()
    assert arguments[2] is post and arguments[3] is comb


def test_new_shape_descriptor_uses_device_fills(monkeypatch):
    from tools.glm_perf.mhc_post_native import NativeMhcPost

    args = inputs()
    native = NativeMhcPost.__new__(NativeMhcPost)
    native.finish_only = False
    native.configs = {}
    native.kernels = {torch.float32: "kernel"}
    calls = []
    native.launch = lambda *values: calls.append(values)
    monkeypatch.setattr(torch, "tensor", lambda *a, **k: pytest.fail("host descriptor in captured path"))
    native(*args)
    descriptor = calls[0][1][-1]
    assert descriptor.tolist() == [2, 4096]
    assert descriptor.dtype == torch.int64 and descriptor.device == args[0].device
    native(*args)
    assert calls[1][1][-1] is descriptor


@pytest.mark.parametrize("rows", [1, 8, 33, 639, 641, 2560])
def test_final_mixer_cannot_enter_decode_or_unqualified_prefill(rows):
    assert not supported_inputs(*inputs(rows), final_only=True)


@pytest.mark.parametrize("rows", [640, 1280])
def test_final_mixer_keeps_original_fp32_inputs_and_has_no_einsum(monkeypatch, rows):
    from tools.glm_perf.mhc_post_native import NativeMhcFinalPost

    args = inputs(rows)
    native = NativeMhcFinalPost.__new__(NativeMhcFinalPost)
    native.finish_only = False
    native.configs = {(args[0].shape, args[0].device): torch.tensor(args[0].shape)}
    native.kernels = {torch.float32: "final kernel"}
    calls = []
    native.launch = lambda *values: calls.append(values)
    monkeypatch.setattr(torch, "einsum", lambda *a, **k: pytest.fail("large final mixer temporary"))
    output = native(*args)
    assert output.dtype == torch.float32 and output.shape == args[1].shape
    assert all(actual is expected for actual, expected in zip(calls[0][1][:4], args, strict=True))


@pytest.mark.parametrize(
    "provenance",
    [
        {},
        {"state_rounding": "fp16_in_fp32_storage", "finish_only": False},
        {"state_rounding": "none_fp32", "finish_only": True},
    ],
)
def test_final_mixer_rejects_rounded_or_partial_build_before_device_access(tmp_path, provenance):
    from tools.glm_perf.mhc_post_native import NativeMhcFinalPost

    (tmp_path / "mhc-provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="complete, unrounded FP32"):
        NativeMhcFinalPost(tmp_path, "unused")
