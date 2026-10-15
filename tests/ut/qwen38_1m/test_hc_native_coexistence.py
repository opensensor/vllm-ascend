# SPDX-License-Identifier: Apache-2.0
"""Host contract tests for the deferred two-library gate."""

import pytest
import torch

from tools.qwen4exp.benchmark_hc_native_coexistence_310 import check_pair
from tools.qwen4exp.benchmark_native_hc_mean_310 import reference as mean_reference
from tools.qwen4exp.benchmark_native_hc_residual_310 import reference as residual_reference


@pytest.mark.parametrize("fault", [None, "residual", "mean"])
def test_composition_checks_both_outputs_and_rejects_corruption(fault):
    hyper = torch.ones(3, 16, dtype=torch.float16)
    block = torch.ones(3, 4, dtype=torch.float16)
    injection = torch.ones(3, 4, dtype=torch.float32)
    gate = torch.full_like(hyper, 0.5)
    calls = []

    def residual(*tensors):
        calls.append("residual")
        output = residual_reference(*tensors)
        return output + 1 if fault == "residual" else output

    def mean(*tensors):
        calls.append("mean")
        output = mean_reference(*tensors)
        return output + 1 if fault == "mean" else output

    if fault is None:
        checks = check_pair(residual, mean, hyper, block, injection, gate)
        assert len(checks) == 2 and all(check["bitwise_equal"] for check in checks)
    else:
        with pytest.raises(AssertionError, match="composition failed"):
            check_pair(residual, mean, hyper, block, injection, gate)
    assert calls == ["residual", "mean"]
