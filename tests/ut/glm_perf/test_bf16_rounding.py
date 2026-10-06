# SPDX-License-Identifier: Apache-2.0
"""Round-through-BF16 contract, including ties, signed zero and overflow."""

import pytest
import torch

from tools.glm_perf.bf16_cast import NativeBF16Cast


@pytest.mark.parametrize("dtype", (torch.float16, torch.float32, torch.bfloat16))
def test_rounding_contract_preserves_intermediate_bf16_precision(dtype):
    values = torch.tensor([0.0, -0.0, 1.00390625, 1.01171875, -1.00390625, 70000.0, 1e-30, -1e-30])
    native = object.__new__(NativeBF16Cast)
    actual = native.round(values, dtype)
    expected = values.bfloat16().to(dtype)
    integer = torch.int32 if dtype == torch.float32 else torch.int16
    assert torch.equal(actual.view(integer), expected.view(integer))
    if dtype == torch.float32:
        assert actual[2] != values[2]


def test_rounding_contract_rejects_unsupported_precision():
    native = object.__new__(NativeBF16Cast)
    with pytest.raises(ValueError, match="rounding requires"):
        native.round(torch.ones(2, dtype=torch.float16))
    with pytest.raises(ValueError, match="rounding requires"):
        native.round(torch.ones(2), torch.int32)
