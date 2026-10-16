# SPDX-License-Identifier: Apache-2.0
"""Deferred private-tensor gates. Importing this file never opens a device.

Call only in an explicitly available hardware window, with a read-only real
bank and privately loaded resources. No model patch, server control or default
CLI is provided. A passing CPU simulation cannot authorize these experiments.
"""

import torch

from tools.qwen4exp.local_route_epilogue import EXPERTS, MAX_ROUTES, TOP_K, WIDTH


@torch.no_grad()
def validate_pair(pair, bank, pack, reference_projection, *, rows=(650, 5150, 20480)):
    """Compare the complete private pair with the retained FP32 backend sum.

    Timing deliberately comes later, after all ranks and both real-bank ends
    pass. The caller must retain the pair, kernels and bridge across launches.
    Only scalar gate results are copied to the CPU, outside any serving path.
    """
    if not rows or any(type(n) is not int or not 128 < n <= MAX_ROUTES or n % TOP_K for n in rows):
        raise ValueError("invalid private gate capacities")
    device = bank.weight.device
    generator = torch.Generator().manual_seed(160219)
    records = []
    for count in rows:
        inputs = torch.randn(count, 640, generator=generator, dtype=torch.float16).to(device)
        packed = pack(inputs)
        for active, skew in ((0, False), (1, False), (count // 4, False), (count // 4, True), (count, False)):
            counts = [active // EXPERTS + (i < active % EXPERTS) for i in range(EXPERTS)]
            if skew:
                counts = [active] + [0] * (EXPERTS - 1)
            ends = torch.tensor(counts, dtype=torch.int64, device=device).cumsum(0)
            inverse = torch.randperm(count, generator=generator).to(device)
            weights = torch.rand(count, 1, generator=generator).to(device)
            full = reference_projection(bank, packed, ends)
            expected = (full.index_select(0, inverse).float() * weights).reshape(-1, TOP_K, WIDTH).sum(1)
            actual = pair(bank, packed, ends, inverse, weights)
            exact = torch.equal(actual.view(torch.int32), expected.view(torch.int32))
            record = dict(
                rows=count,
                active=active,
                skew=skew,
                exact=exact,
                max_abs_error=float((actual - expected).abs().max().cpu()),
            )
            records.append(record)
            if not exact:
                return dict(passed=False, mode=pair.epilogue.mode, records=records, serving_enabled=False)
            del full, expected, actual
    return dict(passed=True, mode=pair.epilogue.mode, records=records, serving_enabled=False)
