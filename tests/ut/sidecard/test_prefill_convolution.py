# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tools.sidecard.layer_math import prefill_convolution


@pytest.mark.parametrize("rows", [1, 4, 17, 32, 64, 128])
@pytest.mark.parametrize("scale", [0.01, 1.0, 40.0])
def test_vectorized_windows_match_incremental_convolution_and_half_rounding(rows, scale):
    torch.manual_seed(91 + rows)
    history = (torch.randn(96, rows + 3) * scale).half()
    kernel = torch.randn(96, 4)
    expected = torch.stack(
        [
            torch.nn.functional.silu((history[:, offset : offset + 4].float() * kernel).sum(-1).half())
            for offset in range(rows)
        ]
    )
    actual = prefill_convolution(history, kernel, torch)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
