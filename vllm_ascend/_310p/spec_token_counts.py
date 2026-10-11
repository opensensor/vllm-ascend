# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact short speculative row counts without the 310P INT64 reduction path."""

import torch

MAX_FAST_COUNT_WIDTH = 9


def _supports_device(device):
    return device.type == "npu"


def count_valid_spec_tokens(valid_mask: torch.Tensor) -> torch.Tensor:
    # Wider reductions use the established implementation. The short 1/6-row
    # MTP shapes were measured on all six chips; large INT32 reductions were
    # slower, so do not generalize this path to arbitrary routing matrices.
    if (
        _supports_device(valid_mask.device)
        and valid_mask.dtype == torch.bool
        and valid_mask.ndim == 2
        and 0 < valid_mask.shape[1] <= MAX_FAST_COUNT_WIDTH
    ):
        # Each count is at most nine: accumulation is exact in INT32. Keep the
        # public INT64 result used by gather and the accepted-token accounting.
        return valid_mask.sum(dim=1, dtype=torch.int32).to(torch.int64)
    return valid_mask.sum(dim=1)
