# SPDX-License-Identifier: Apache-2.0
"""Direct GDN fusion rejects incompatible geometry before device submission."""

import pytest
import torch

from tools.qwen4exp.direct_gdn_output import DirectGDNNormGate


@pytest.mark.parametrize("fault", ["rows", "width", "dtype", "gamma", "gate", "cpu"])
def test_shape_dtype_and_device_rejected_without_launch(fault):
    op = object.__new__(DirectGDNNormGate)
    x = torch.ones(36, 128, dtype=torch.float32)
    gamma = torch.ones(128, dtype=torch.float32)
    gate = torch.ones_like(x, dtype=torch.float16)
    if fault == "rows":
        x = x[:0]
        gate = gate[:0]
    elif fault == "width":
        x = x[:, :64]
        gate = gate[:, :64]
    elif fault == "dtype":
        x = x.half()
    elif fault == "gamma":
        gamma = gamma[:64]
    elif fault == "gate":
        gate = gate[:3]
    with pytest.raises(ValueError):
        op(x, gamma, gate)
