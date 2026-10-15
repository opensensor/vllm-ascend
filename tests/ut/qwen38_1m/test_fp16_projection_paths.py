# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Operator probe inputs stay dynamic and preparation is accounted for."""

import pytest
import torch

from tools.qwen4exp.benchmark_fp16_projection_paths_310 import operator_functions


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
@torch.no_grad()
def test_every_arm_observes_changed_inputs_with_same_operand_rounding(input_dtype):
    torch.manual_seed(1075)
    inputs = torch.randn(3, 32, dtype=input_dtype)
    weight = torch.randn(16, 32).half()
    functions = operator_functions(inputs, weight, weight.t().contiguous(), lambda value: value)
    first = functions["linear_nz_weight"]().clone()
    for phase in range(3):
        inputs.copy_(torch.randn_like(inputs) + phase)
        expected = functions["linear_nz_weight"]()
        assert not torch.equal(first, expected)
        for function in functions.values():
            assert torch.equal(function(), expected)


@torch.no_grad()
def test_dynamic_format_conversion_is_inside_the_timed_callable():
    inputs = torch.randn(3, 32)
    weight = torch.randn(16, 32).half()
    seen = []

    def format_input(value):
        seen.append(value.dtype)
        return value

    functions = operator_functions(inputs, weight, weight.t().contiguous(), format_input)
    assert seen == []
    functions["linear_nz_weight"]()
    functions["mm_cached_transpose"]()
    assert seen == []
    for _ in range(2):
        functions["linear_nz_input"]()
        functions["mm_nz_input_cached_transpose"]()
    assert seen == [torch.float16] * 4
