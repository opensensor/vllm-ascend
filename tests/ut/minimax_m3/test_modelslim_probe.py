# SPDX-License-Identifier: Apache-2.0
"""Optional CPU gate; run in an isolated process with the ModelSlim checkout."""

import pytest
import torch

from tools.minimax_m3.modelslim_probe import quantize_modelslim
from vllm_ascend._310p.ops.minimax_m3 import unpack_w4


def test_modelslim_bridge_preserves_stored_scale_quantizer():
    pytest.importorskip("msmodelslim")
    weight = torch.randn(64, 96, generator=torch.Generator().manual_seed(17))
    codes, scales = quantize_modelslim(weight)
    expanded = scales.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
    expected = (weight / expanded).round().clamp(-8, 7) * expanded
    torch.testing.assert_close(unpack_w4(codes, scales), expected, atol=0, rtol=0)
    codes, scales = quantize_modelslim(torch.zeros(32, 32))
    assert torch.count_nonzero(codes) == 0 and torch.all(scales > 0)


@pytest.mark.parametrize("weight", [torch.ones(31, 32), torch.full((32, 32), torch.nan), torch.full((32, 32), 1e10)])
def test_modelslim_bridge_rejects_bad_shapes_and_values(weight):
    pytest.importorskip("msmodelslim")
    with pytest.raises(ValueError):
        quantize_modelslim(weight)
