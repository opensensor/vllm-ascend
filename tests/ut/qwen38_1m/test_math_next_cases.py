# SPDX-License-Identifier: Apache-2.0
"""Contracts of the new stock math screening probes."""

from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp.benchmark_math_paths_310 import compare_outputs, gate_passed
from tools.qwen4exp.math_next_cases import CASE_NAMES, build_case
from tools.qwen4exp.math_probe_cases import changed_inputs


def rms(x, gamma, eps):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * gamma, None


def router(x, k):
    weights, ids = x.float().softmax(-1).topk(k, -1)
    return weights.half(), ids.int(), ids.int()


def norm_quant(x, gamma, beta, scale, offset, epsilon):
    y = rms(x.float(), gamma.float(), epsilon)[0] + beta.float()
    return (y * scale.float() + offset).round().clamp(-128, 127).to(torch.int8)


@pytest.mark.parametrize("name", CASE_NAMES)
def test_candidates_do_not_mutate_inputs_and_have_correct_output_contract(name):
    native = SimpleNamespace(npu_rms_norm=rms, npu_moe_gating_top_k_softmax=router, npu_rms_norm_quant=norm_quant)
    case = build_case(name, 3, native=native)
    values = changed_inputs(case.inputs, "random", 13)
    saved = [x.clone() for x in values]
    checks = compare_outputs(case.baseline(*values), case.candidate(*values))
    assert checks and all(check["finite"] for check in checks)
    assert all(torch.equal(old, x) for old, x in zip(saved, values))
    if name in ("glu_fp32", "gdn_rms_affine_gate", "rms_quant_static"):
        assert gate_passed(checks)
    if name == "router_fused_fp16":
        assert checks[0]["dtype"] == "torch.float32" and checks[1]["dtype"] == "torch.int64"
        assert not gate_passed(checks)


def test_bad_shapes_rejected_before_native_calls():
    with pytest.raises(ValueError):
        build_case("glu_fp32", 0)
    with pytest.raises(ValueError):
        build_case("unknown", 3)
