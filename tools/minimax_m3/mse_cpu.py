# SPDX-License-Identifier: Apache-2.0
"""Optional compiled CPU implementation of the existing 48-candidate W4 search.

Keep float64 arithmetic, the original candidate grid, and round-to-even. Close
candidate ties fall back to the original Torch reduction order so existing
export receipts remain usable. No NPU runtime or device kernels are involved.
"""

import numpy as np
import torch
from numba import njit, prange, set_num_threads

from tools.deepseek_w2.w2_format import (
    SCALE_FLOOR,
    W4_BITS,
    W4_CODE_MAX,
    W4_CODE_MIN,
    broadcast_block_scales,
    compute_block_scales,
)

BLOCK_WIDTH = 32
BLOCK_ELEMENTS = BLOCK_WIDTH**2
MSE_CANDIDATES = 48
# Both error sums contain 1024 nonnegative float64 terms. This margin is much
# larger than their O(n * epsilon) reduction-order error. Ambiguous candidates
# are recomputed with Torch; changing summation order must not change winners.
MSE_TIE_RTOL = 1e-10


@njit(cache=True, parallel=True, fastmath=False)
def _search_scales(blocks, fractions):
    scales = np.empty(blocks.shape[0], dtype=np.float64)
    ambiguous = np.zeros(blocks.shape[0], dtype=np.bool_)
    for block in prange(blocks.shape[0]):
        absmax = 0.0
        for element in range(BLOCK_ELEMENTS):
            absmax = max(absmax, abs(blocks[block, element]))
        best_error = np.inf
        second_error = np.inf
        best_scale = SCALE_FLOOR
        for fraction in fractions:
            scale = max(absmax * fraction, SCALE_FLOOR)
            error = 0.0
            for element in range(BLOCK_ELEMENTS):
                weight = blocks[block, element]
                code = min(float(W4_CODE_MAX), max(float(W4_CODE_MIN), np.rint(weight / scale)))
                difference = code * scale - weight
                error += difference * difference
            if error < best_error:
                second_error = best_error
                best_error = error
                best_scale = scale
            elif scale != best_scale and error < second_error:
                second_error = error
        scales[block] = best_scale
        ambiguous[block] = second_error - best_error <= MSE_TIE_RTOL * best_error
    return scales, ambiguous


def quantize_mse_cpu(weight):
    """Return canonical codes/scales, accelerating only the candidate search."""
    if weight.device.type != "cpu" or weight.ndim != 2 or any(size % BLOCK_WIDTH for size in weight.shape):
        raise ValueError("Compiled M3 MSE requires a CPU matrix tiled into 32x32 blocks")
    if not torch.isfinite(weight).all():
        raise ValueError("Compiled M3 MSE requires finite weights")
    # Respect --threads; Numba's thread pool is independent of Torch's.
    set_num_threads(torch.get_num_threads())
    weight = weight.double()
    rows, cols = weight.shape
    blocks = weight.reshape(rows // BLOCK_WIDTH, BLOCK_WIDTH, cols // BLOCK_WIDTH, BLOCK_WIDTH)
    blocks = blocks.permute(0, 2, 1, 3).reshape(-1, BLOCK_ELEMENTS).contiguous()
    fractions = torch.linspace(1 / MSE_CANDIDATES, 1, MSE_CANDIDATES, dtype=torch.float64)
    scales, ambiguous = _search_scales(blocks.numpy(), fractions.numpy())
    scales = torch.from_numpy(scales)
    indices = torch.from_numpy(np.flatnonzero(ambiguous))
    if indices.numel():
        tied = blocks.index_select(0, indices).reshape(-1, BLOCK_WIDTH)
        canonical = compute_block_scales(tied, n_bits=W4_BITS, method="mse", mse_num_candidates=MSE_CANDIDATES)
        scales[indices] = canonical.flatten()
    scales = scales.reshape(rows // BLOCK_WIDTH, cols // BLOCK_WIDTH)
    expanded = broadcast_block_scales(scales, rows, cols)
    codes = torch.round(weight / expanded).clamp(W4_CODE_MIN, W4_CODE_MAX).to(torch.int8)
    return codes, scales
