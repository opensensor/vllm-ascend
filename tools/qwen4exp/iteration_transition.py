# SPDX-License-Identifier: Apache-2.0
"""Device-only acceptance and next-input construction for a bounded MTP2 step.

These functions never observe tensor values on the host. Allocation, request
identity, and page ownership are admission responsibilities. The caller orders
acceptance -> grouped state commit -> next-input preparation on its stream.
This module alone does not replace the engine's synchronous scheduler.
"""

from typing import NamedTuple

import torch

from tools.qwen4exp.mamba_commit import PADDED_ROWS
from tools.qwen4exp.mamba_commit_plan import plan_aligned_commit

MTP_WIDTH = 3
MAX_REQUESTS = 4


class AcceptedStep(NamedTuple):
    counts: torch.Tensor
    next_counts: torch.Tensor
    rejected: torch.Tensor
    error: torch.Tensor


class NextStep(NamedTuple):
    input_ids: torch.Tensor
    positions: torch.Tensor
    computed: torch.Tensor
    seq_lens: torch.Tensor
    active: torch.Tensor
    error: torch.Tensor


def accept_step(sampled, inputs, *, block_size):
    """Fill native commit metadata directly from sampler output on device.

    ``inputs`` is owned INT32 [5,8]; its first four rows are admitted metadata.
    Valid sampler IDs form a contiguous prefix followed by -1. Zero acceptance
    is permitted but cannot continue. Error rows receive zero accepted count;
    admission must guarantee their static metadata is a no-copy case at zero.
    Error receipts must gate continuation and be checked at window retirement.
    """
    n = sampled.shape[0]
    if (
        sampled.ndim != 2
        or sampled.shape[1] != MTP_WIDTH
        or not 1 <= n <= MAX_REQUESTS
        or sampled.dtype not in (torch.int32, torch.int64)
        or inputs.shape != (5, PADDED_ROWS)
        or inputs.dtype != torch.int32
        or sampled.device != inputs.device
    ):
        raise ValueError("requires MTP2 integer samples and owned INT32 [5,8] metadata")
    valid = sampled >= 0
    counts = valid.sum(dim=1, dtype=torch.int32)
    columns = torch.arange(MTP_WIDTH, device=sampled.device, dtype=torch.int32)
    prefix = columns.unsqueeze(0) < counts.unsqueeze(1)
    error = ((valid != prefix) | (sampled < -1)).any(dim=1)
    error = error | (counts > inputs[3, :n] + 1)
    accepted = torch.where(error, 0, counts)
    inputs[4, :n].copy_(accepted)
    plan = plan_aligned_commit(*inputs[:, :n].unbind(), block_size=block_size)
    return AcceptedStep(accepted, plan.next_accepted_counts, inputs[3, :n] + 1 - accepted, error)


def prepare_next_step(sampled, drafts, inputs, accepted, *, max_model_len, eos_token_id=None):
    """Build next target token IDs/positions without a host token round trip.

    Uniform MTP2 admission reserves all three next-query slots and leaves the
    batch in the same row order. EOS or invalid rows are inactive and publish
    safe zero IDs/positions, never executable work. The model consumer MUST
    honor active/error; zero padding alone is not cancellation or termination.
    No physical page allocation or mapping is inferred here.
    """
    n = sampled.shape[0]
    if (
        sampled.shape != (n, MTP_WIDTH)
        or not 1 <= n <= MAX_REQUESTS
        or drafts.shape != (n, MTP_WIDTH - 1)
        or drafts.dtype not in (torch.int32, torch.int64)
        or sampled.dtype not in (torch.int32, torch.int64)
        or inputs.shape != (5, PADDED_ROWS)
        or inputs.dtype != torch.int32
        or sampled.device != drafts.device
        or sampled.device != inputs.device
        or type(max_model_len) is not int
        or not MTP_WIDTH <= max_model_len < (1 << 30)
        or (eos_token_id is not None and (type(eos_token_id) is not int or eos_token_id < 0))
    ):
        raise ValueError("invalid admitted MTP2 next-step geometry")
    if any(t.shape != (n,) or t.device != sampled.device for t in accepted):
        raise ValueError("acceptance receipt does not match request rows/device")
    columns = torch.arange(MTP_WIDTH, device=sampled.device, dtype=torch.int32)
    # The last accepted token becomes the first input of the next target step.
    # Raw sampler counts, not normalized Mamba counts, advance positions.
    computed = inputs[1, :n] + inputs[2, :n] - inputs[3, :n] + accepted.counts - 1
    positions = computed.unsqueeze(1) + columns.unsqueeze(0)
    index = (accepted.counts - 1).clamp(0, MTP_WIDTH - 1).to(torch.int64).unsqueeze(1)
    last = sampled.gather(1, index)
    ids = torch.cat((last, drafts), dim=1)
    error = accepted.error | (drafts < 0).any(dim=1) | (computed < 0)
    active = (accepted.counts > 0) & ~error & (computed + MTP_WIDTH <= max_model_len)
    if eos_token_id is not None:
        generated = columns.unsqueeze(0) < accepted.counts.unsqueeze(1)
        stopped = ((sampled == eos_token_id) & generated).any(dim=1)
        active = active & ~stopped
    return NextStep(
        torch.where(active.unsqueeze(1), ids, 0),
        torch.where(active.unsqueeze(1), positions, 0),
        torch.where(active, computed, 0),
        torch.where(active, computed + MTP_WIDTH, 0),
        active,
        error,
    )


def advance_ple_history(history, first_input, sampled, accepted):
    """Advance pre-query n-gram history by accepted target inputs on device.

    The final sampled token is the next query's first input, so it must NOT be
    included in its preceding history. For acceptance A, append this query's
    first input plus the first A-1 sampled tokens. Rejected draft tokens never
    enter history. Zero/error rows retain their previous history.
    """
    n = sampled.shape[0]
    if (
        history.ndim != 2
        or history.shape[0] != n
        or history.shape[1] < 1
        or sampled.shape != (n, MTP_WIDTH)
        or first_input.shape != (n,)
        or not 1 <= n <= MAX_REQUESTS
        or any(t.dtype not in (torch.int32, torch.int64) for t in (history, first_input, sampled))
        or any(t.device != history.device for t in (first_input, sampled, *accepted))
        or any(t.shape != (n,) for t in accepted)
    ):
        raise ValueError("history, target input, samples and acceptance must share admitted request rows")
    accepted_count = accepted.counts.clamp(0, MTP_WIDTH)
    available = torch.cat((history, first_input.unsqueeze(1), sampled), dim=1)
    offsets = torch.arange(history.shape[1], dtype=torch.int64, device=history.device)
    indices = offsets.unsqueeze(0) + accepted_count.to(torch.int64).unsqueeze(1)
    advanced = available.gather(1, indices).to(history.dtype)
    valid = ~accepted.error & (accepted.counts > 0) & (accepted.counts <= MTP_WIDTH)
    return torch.where(valid.unsqueeze(1), advanced, history)
