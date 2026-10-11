# SPDX-License-Identifier: Apache-2.0
"""Combine compact unweighted expert rows with separate ordered FP32 steps."""

COMBINE_BLOCK_TOKENS = 32


def combine_compact(routes, output_bytes, weights, width, *, reply_fp16=False):
    # Loaded only in the serving worker; offline planning needs no NumPy.
    import numpy as np

    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim != 2 or not weights.size or width < 1 or not np.isfinite(weights).all():
        raise ValueError("invalid compact combination weights or geometry")
    if (weights < 0).any() or len(output_bytes) != len(routes) * width * (2 if reply_fp16 else 4):
        raise ValueError("invalid compact combination weights or output length")
    outputs = np.frombuffer(output_bytes, dtype="<f2" if reply_fp16 else "<f4").reshape(len(routes), width)
    if not np.isfinite(outputs).all():
        raise ValueError("nonfinite compact expert output")
    tokens = np.asarray([route.token for route in routes], dtype=np.int64)
    slots = np.asarray([route.slot for route in routes], dtype=np.int64)
    if (tokens < 0).any() or (tokens >= len(weights)).any() or (slots < 0).any() or (slots >= weights.shape[1]).any():
        raise ValueError("compact combination route is outside the batch")
    if len({(route.token, route.slot) for route in routes}) != len(routes):
        raise ValueError("duplicate compact combination route")
    result = np.zeros((len(weights), width), dtype=np.float32)
    # Small CPU tiles keep products and gathers within cache even when IPC
    # packets carry an entire prefill batch. Token tiles are independent.
    for start in range(0, len(weights), COMBINE_BLOCK_TOKENS):
        active = (tokens >= start) & (tokens < start + COMBINE_BLOCK_TOKENS)
        for slot in range(weights.shape[1]):
            indices = np.flatnonzero(active & (slots == slot))
            selected = tokens[indices]
            # A materialized product prevents contraction into fused multiply-add.
            product = np.multiply(outputs[indices], weights[selected, slot, None], dtype=np.float32)
            result[selected] = np.add(result[selected], product, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite compact combination result")
    return result
