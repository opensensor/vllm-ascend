# SPDX-License-Identifier: Apache-2.0
"""Device checkpoint pre-copy decisions using the qualified grouped-copy ABI.

Page ownership must be admitted before invoking this transform. This preserves
align-mode Mamba semantics; it does not introduce a replacement cache policy.
"""

from typing import NamedTuple

import torch

from tools.qwen4exp.mamba_commit import PADDED_ROWS
from tools.qwen4exp.runtime_geometry import MAX_REQUESTS

MTP_WIDTH = 3

NO_COPY_COMPUTED = 10


class CheckpointPreparation(NamedTuple):
    state_indices: torch.Tensor
    normalized_counts: torch.Tensor
    copy_required: torch.Tensor
    error: torch.Tensor


def prepare_checkpoint(previous, computed, accepted, inputs, *, block_size):
    """Encode pre-copy as an exact grouped-copy operation, without host reads.

    For destination D and acceptance A, choose running=(D+1)*B-(A-1).
    The grouped kernel then copies previous -> D with bias A-1. Non-copy
    rows use a valid below-boundary tuple, so no bank address is accessed.
    Error rows are inert and must prevent continuation at the window boundary.
    Admission bounds computed tokens below INT32_MAX-block_size-MTP_WIDTH.
    """
    n = previous.numel()
    if (
        not 1 <= n <= MAX_REQUESTS
        or previous.shape != (n,)
        or any(
            t.shape != (n,) or t.dtype != torch.int32 or t.device != previous.device
            for t in (previous, computed, accepted)
        )
        or inputs.shape != (5, PADDED_ROWS)
        or inputs.dtype != torch.int32
        or inputs.device != previous.device
        or type(block_size) is not int
        or not NO_COPY_COMPUTED + MTP_WIDTH < block_size < (1 << 30)
    ):
        raise ValueError("checkpoint preparation requires admitted INT32 MTP2 rows")
    current = torch.div(computed + MTP_WIDTH - 1, block_size, rounding_mode="floor")
    error = (previous < -1) | (computed < 0) | (accepted < 1) | (accepted > MTP_WIDTH)
    copy = (previous >= 0) & (previous != current) & ~error
    bias = accepted - 1
    aligned = (current + 1) * block_size
    inputs[0, :n].copy_(torch.where(copy, previous, 0))
    inputs[1, :n].copy_(torch.where(copy, aligned - bias - 1, NO_COPY_COMPUTED))
    inputs[2, :n].fill_(MTP_WIDTH)
    inputs[3, :n].fill_(MTP_WIDTH - 1)
    inputs[4, :n].copy_(torch.where(copy, accepted, 0))
    return CheckpointPreparation(current, torch.where(copy, 1, accepted), copy, error)
