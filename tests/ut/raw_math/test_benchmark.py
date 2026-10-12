# SPDX-License-Identifier: Apache-2.0
"""CPU regression gates for the independently timed hardware harness."""

import math

import pytest
import torch

from tools.raw_math.benchmark import changed_input, inputs, run_case, validate
from tools.raw_math.card import aggregate_rate
from tools.raw_math.cases import Case, copy_cases, matrix_cases, rates, summary, vector_cases


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(op="gemm", dtype="uint8", m=1, n=4, k=4),
        dict(op="gemm", dtype="fp16", m=True, n=4, k=4),
        dict(op="gemm", dtype="fp16", m=8193, n=4, k=4),
        dict(op="copy", dtype="uint8", m=1024**3 + 1),
        dict(op="copy", dtype="uint8", m=4, pinned=True),
        dict(op="add", dtype="fp16", m=4),
        dict(op="softmax", dtype="fp32", m=1, n=0),
    ],
)
def test_invalid_contracts(kwargs):
    with pytest.raises(ValueError):
        Case("invalid", **kwargs)


def test_suites_and_rate_units():
    assert len(matrix_cases()) == 36
    assert len(vector_cases()) == 16
    assert len(copy_cases()) == 25
    matrix = Case("matrix", "gemm", "fp16", 4096, 4096, 4096)
    assert rates(matrix, 1)["tera_ops_per_second"] == pytest.approx(137.438953472)
    copy = Case("copy", "copy", "uint8", 1024**3)
    rate = rates(copy, 1000)
    assert rate["payload_gbytes_per_second"] == pytest.approx(1.073741824)
    assert rate["read_plus_write_gbytes_per_second"] == pytest.approx(2.147483648)
    # Both chips each run four GEMMs inside one measured coordinator interval.
    assert aggregate_rate(matrix, 4, 8) == pytest.approx(137.438953472)
    with pytest.raises(ValueError):
        aggregate_rate(matrix, 0, 8)


@pytest.mark.parametrize("values", [[], [0], [-1], [math.nan], [math.inf]])
def test_invalid_timings(values):
    with pytest.raises(ValueError):
        summary(values)
    if values:
        with pytest.raises(ValueError):
            rates(Case("copy", "copy", "uint8", 4), values[0])


@pytest.mark.parametrize("dtype", ["fp16", "fp32", "bf16", "int8"])
@pytest.mark.parametrize("shape", [(5, 7, 32), (1, 1, 8)])
def test_independent_matrix_reference(dtype, shape):
    case = Case("small", "gemm", dtype, *shape)
    x, weight = inputs(case, torch)
    actual = (x.double() @ weight.double().T).to(torch.float16 if dtype == "int8" else x.dtype)
    assert validate(case, (x, weight), actual, torch)["passed"]
    with pytest.raises(AssertionError):
        validate(case, (x, weight), actual + 1, torch)
    with pytest.raises(AssertionError):
        validate(case, (x, weight), actual.to(torch.float64), torch)


@pytest.mark.parametrize("op", ["add", "exp", "softmax", "rmsnorm"])
def test_vector_reference_and_nonfinite_gate(op):
    case = Case("small", op, "fp32", 3, 17) if op in ("softmax", "rmsnorm") else Case("small", op, "fp32", 17)
    source = inputs(case, torch)
    x = source[0]
    actual = (
        x + source[1]
        if op == "add"
        else x.exp()
        if op == "exp"
        else x.softmax(-1)
        if op == "softmax"
        else x * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    )
    assert validate(case, source, actual, torch)["passed"]
    actual.flatten()[-1] = math.nan
    with pytest.raises(AssertionError, match="nonfinite"):
        validate(case, source, actual, torch)


def test_copy_checks_all_bytes_and_handles_partial_pattern():
    case = Case("copy", "copy", "uint8", 257)
    source = inputs(case, torch)
    assert source[0].numel() == 257
    assert validate(case, source, source[0].clone(), torch)["passed"]
    wrong = source[0].clone()
    wrong[193] ^= 1
    with pytest.raises(AssertionError, match="all source bytes"):
        validate(case, source, wrong, torch)


def test_large_vector_sample_indices_do_not_round_past_end():
    # FP32 linspace rounds 2**26-1 up to 2**26. Sampling indices need FP64.
    case = Case("large", "exp", "fp32", 2**26)
    source = (torch.zeros(case.m),)
    assert validate(case, source, torch.ones(case.m), torch)["passed"]


def test_fp32_input_rounding_is_explicit_and_cannot_pass_strict_gate():
    case = Case("rounding", "gemm", "fp32", 3, 7, 4096)
    source = inputs(case, torch)
    rounded = (source[0].half().double() @ source[1].half().double().T).float()
    with pytest.raises(AssertionError):
        validate(case, source, rounded, torch)
    diagnostic = validate(case, source, rounded, torch, reference_fp16_inputs=True)
    assert diagnostic["passed"]
    assert not diagnostic["strict_fp32_passed"]
    assert diagnostic["strict_fp32_max_absolute_error"] > 0


class CopyBackend:
    def __init__(self, source, stale=False):
        self.x = source[0].clone()
        self.result = None
        self.stale = stale

    def measure(self, repeats):
        self.result = self.x.clone()
        return 0.1, 0.2

    def read(self):
        return self.result

    def refresh(self, value):
        if not self.stale:
            self.x.copy_(value)


def test_changed_input_gate_rejects_stale_output():
    case = Case("copy", "copy", "uint8", 257)
    source = inputs(case, torch)
    assert not torch.equal(source[0], changed_input(case, source[0]))
    with pytest.raises(AssertionError, match="all source bytes"):
        run_case(case, source, CopyBackend(source, stale=True), torch, 1, 3)
    result = run_case(case, source, CopyBackend(source), torch, 1, 3)
    assert result["final_validation"]["passed"]
    assert result["timings"]["queued"]["device"]["samples_ms"] == [0.1] * 3
