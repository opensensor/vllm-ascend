# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tools.qwen4exp.benchmark_math_paths_310 import compare_outputs, gate_passed
from tools.qwen4exp.math_probe_cases import CASE_NAMES, build_case, changed_inputs


def fake_swiglu(x, dim=-1):
    gate, up = x.chunk(2, dim=dim)
    return F.silu(gate) * up


def fake_rms(x, weight, eps):
    rstd = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return (x.float() * rstd * weight).to(x.dtype), rstd


def fake_add_rms(x, residual, weight, eps):
    total = x + residual
    normalized, rstd = fake_rms(total, weight, eps)
    return normalized, rstd, total


@pytest.mark.parametrize("name", CASE_NAMES)
def test_cases_change_inputs_without_mutating_arguments(name):
    native = SimpleNamespace(npu_swiglu=fake_swiglu, npu_rms_norm=fake_rms, npu_add_rms_norm=fake_add_rms)
    case = build_case(name, 2, native=native)
    first = changed_inputs(case.inputs, "random", 1)
    second = changed_inputs(case.inputs, "random", 2)
    saved = tuple(x.clone() for x in first)
    expected = case.baseline(*first)
    actual = case.candidate(*first)
    result = compare_outputs(expected, actual)
    assert result and all(item["finite"] for item in result)
    assert all(torch.equal(left, right) for left, right in zip(first, saved))
    assert not torch.equal(first[0], second[0])
    assert not torch.equal(
        case.baseline(*first)[0] if isinstance(expected, tuple) else expected,
        case.baseline(*second)[0] if isinstance(expected, tuple) else case.baseline(*second),
    )


def test_byte_parity_distinguishes_signed_zero():
    outputs = compare_outputs(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert not gate_passed(outputs)
    assert outputs[0]["changed_elements"] == 0


def test_nonfinite_outputs_never_qualify():
    outputs = compare_outputs(torch.tensor([float("inf")]), torch.tensor([float("inf")]))
    assert not gate_passed(outputs)


def test_router_tie_indices_are_checked_separately_from_weights():
    case = build_case("router_unsorted_then_sort", 2)
    values = changed_inputs(case.inputs, "ties", 4)
    outputs = compare_outputs(case.baseline(*values), case.candidate(*values))
    assert outputs[1]["dtype"] == "torch.int64"
    assert outputs[1]["fp16_round_equal"] is None


@pytest.mark.parametrize("name", ["hc_down_reciprocal", "hc_injection_reciprocal_inplace", "route_weight_inplace"])
def test_simple_exact_candidates_pass_host_gate(name):
    case = build_case(name, 3)
    for pattern in ("random", "zeros", "large", "ties", "near_ties"):
        values = changed_inputs(case.inputs, pattern, 5)
        assert gate_passed(compare_outputs(case.baseline(*values), case.candidate(*values)))


@pytest.mark.parametrize("name", ["hc_injection_fp16_reciprocal_inplace", "hc_down_fp16_reciprocal"])
def test_actual_hc_pointwise_probes_keep_fp16_intermediate_rounding(name):
    case = build_case(name, 3)
    for pattern in ("random", "zeros", "large", "ties", "near_ties"):
        values = changed_inputs(case.inputs, pattern, 13)
        expected, actual = case.baseline(*values), case.candidate(*values)
        assert expected.dtype == actual.dtype == torch.float16
        assert gate_passed(compare_outputs(expected, actual))
