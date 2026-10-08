# SPDX-License-Identifier: Apache-2.0
"""Keep private metadata dispatch separate from the process-wide torch module."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf.integer_divide import (
    DIVISORS,
    DivisionTorch,
    NativeIntegerDivide,
    prepare_counts,
    private_divisions,
)
from tools.glm_perf.integer_divide_control import manifest


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("divisor", DIVISORS)
def test_signed_metadata_dispatch_and_globals_preserved(dtype, divisor):
    values = torch.tensor([torch.iinfo(dtype).min, -641, -5, -1, 0, 1, 5, 641, torch.iinfo(dtype).max], dtype=dtype)
    calls = []

    class Native:
        device = torch.device("cpu")

        def __call__(self, value, denominator):
            calls.append((value, denominator))
            return torch.div(value, denominator, rounding_mode="floor")

    def original(value, denominator=divisor, *, offset=3):
        return torch.div(value, denominator, rounding_mode="floor") + offset

    proxy = DivisionTorch(Native())
    candidate = private_divisions(original, proxy)
    assert candidate.__globals__["torch"] is proxy
    assert original.__globals__["torch"] is torch
    assert candidate.__code__ is original.__code__ and candidate.__closure__ is original.__closure__
    assert torch.equal(candidate(values), original(values))
    assert len(calls) == 1
    assert candidate.__kwdefaults__ == original.__kwdefaults__


@pytest.mark.parametrize(
    "dtype,divisor,mode",
    [(torch.float32, 4, "floor"), (torch.int64, 3, "floor"), (torch.int64, 4, "trunc"), (torch.int64, 4, None)],
)
def test_unsupported_calls_delegate(dtype, divisor, mode):
    native = SimpleNamespace(device=torch.device("cpu"))
    proxy = DivisionTorch(native)
    value = torch.tensor([-7, 0, 7], dtype=dtype)
    assert torch.equal(proxy.div(value, divisor, rounding_mode=mode), torch.div(value, divisor, rounding_mode=mode))
    assert proxy.cat is torch.cat


def test_output_argument_is_not_intercepted():
    proxy = DivisionTorch(SimpleNamespace(device=torch.device("cpu")))
    value = torch.tensor([-7, 7], dtype=torch.int64)
    output = torch.empty_like(value)
    assert proxy.div(value, 4, rounding_mode="floor", out=output) is output
    assert output.tolist() == [-2, 1]


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(0,), (2,), (2, 4), (3, 17), (640,)])
def test_descriptor_protocol_owns_padding_and_preserves_input_layout(dtype, shape):
    native = object.__new__(NativeIntegerDivide)
    native.device, native.kernel, native.configs, native.calls = torch.device("cpu"), object(), {}, 0
    calls = []

    def launch(kernel, arguments, cores):
        value, output, config = arguments
        assert value.is_contiguous() and config.tolist() == [value.numel(), 160, value.element_size()]
        assert output._base.numel() >= output.numel()
        assert output._base.numel() * output.element_size() % 32 == 0
        output.copy_(torch.div(value, 160, rounding_mode="floor"))
        output._base[output.numel() :].zero_()
        calls.append(cores)

    native.launch = launch
    value = torch.arange(torch.tensor(shape).prod().item() * 2, dtype=dtype)[::2].reshape(shape)
    output = native(value, 160)
    assert torch.equal(output, torch.div(value, 160, rounding_mode="floor")) and output.shape == value.shape
    assert bool(calls) == bool(value.numel())


@pytest.fixture
def bundle(tmp_path):
    package = tmp_path / "glm_reconstruction_v927_helpers"
    package.mkdir()
    helper = package / "integer_divide.py"
    helper.write_text("# frozen helper\n")
    options = {
        "helper_package": package.name,
        "namespace": "glm_reconstruction_v927",
        "version": 927,
        "integer_metadata_divide": True,
    }
    binary, bridge = tmp_path / "glm_integer_divide.bin", tmp_path / "glm_reconstruction_bridge_v927.so"
    for file in (binary, bridge):
        file.write_bytes(b"isolated fixture")
    digest = lambda file: hashlib.sha256(file.read_bytes()).hexdigest()
    (tmp_path / "provenance.json").write_text(
        json.dumps({"_build": options, "_helpers": {helper.name: digest(helper)}})
    )
    records = [
        {
            "dtype": dtype,
            "divisor": divisor,
            "count": count,
            "passed": True,
            "changed_input_replay": True,
            "signed_extremes": True,
            "owned_padding_checked": True,
        }
        for dtype in ("torch.int32", "torch.int64")
        for divisor in DIVISORS
        for count in (2, 8, 640)
    ]
    report = tmp_path / "gates.json"
    report.write_text(
        json.dumps(
            {
                "complete": True,
                "build_options": options,
                "records": records,
                "binaries": {p.name: digest(p) for p in (binary, bridge)},
            }
        )
    )
    return tmp_path, report, helper, binary


def test_complete_signed_manifest(bundle):
    root, report, _, _ = bundle
    value = manifest(root, report)
    value.verify_files()
    assert value.name == "integer_divide_v927"


@pytest.mark.parametrize("field", ["passed", "changed_input_replay", "owned_padding_checked", "signed_extremes"])
def test_incomplete_hardware_gate_rejected(bundle, field):
    root, report, _, _ = bundle
    data = json.loads(report.read_text())
    data["records"][-1][field] = False
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="signed extremes"):
        manifest(root, report)


@pytest.mark.parametrize("index", [2, 3])
def test_changed_binary_or_helper_rejected(bundle, index):
    bundle[index].write_bytes(b"different content")
    with pytest.raises(ValueError, match="different binaries|helper changed"):
        manifest(bundle[0], bundle[1])


def test_prepared_descriptors_cover_shapes_without_replacing_existing_storage():
    native = SimpleNamespace(device=torch.device("cpu"), configs={})
    prepare_counts(native, [2, 8, 2, 640])
    assert len(native.configs) == 18
    for (count, divisor, dtype), descriptor in native.configs.items():
        assert descriptor.tolist() == [count, divisor, dtype.itemsize] and descriptor.is_contiguous()
    held = dict(native.configs)
    prepare_counts(native, [2, 8, 640])
    assert all(native.configs[key] is value for key, value in held.items())


@pytest.mark.parametrize("counts", [[0], [-1], [2.0], [True], [1, True]])
def test_descriptor_counts_require_positive_host_integers(counts):
    with pytest.raises(ValueError, match="positive host integers"):
        prepare_counts(SimpleNamespace(device=torch.device("cpu"), configs={}), counts)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("divisor", DIVISORS)
def test_native_remainder_handles_signed_extremes_with_integer_wrap(dtype, divisor):
    limits = torch.iinfo(dtype)
    values = torch.tensor([limits.min, limits.max, -641, -1, 0, 1, 641], dtype=dtype)

    class Native:
        device = torch.device("cpu")

        def __call__(self, value, denominator):
            return torch.div(value, denominator, rounding_mode="floor")

    assert torch.equal(DivisionTorch(Native()).remainder(values, divisor), torch.remainder(values, divisor))


@pytest.mark.parametrize("dtype,divisor", [(torch.float32, 4), (torch.int64, 3)])
def test_unsupported_remainders_delegate(dtype, divisor):
    values = torch.tensor([-7, 7], dtype=dtype)
    assert torch.equal(
        DivisionTorch(SimpleNamespace(device=torch.device("cpu"))).remainder(values, divisor),
        torch.remainder(values, divisor),
    )
