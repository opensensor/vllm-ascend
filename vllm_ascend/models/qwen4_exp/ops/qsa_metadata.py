# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host query boundaries for the rows actually consumed by QSA."""

from __future__ import annotations


def packed_qsa_query_lens(metadata: object, num_tokens: int) -> list[int] | None:
    """Exclude padded suffix rows without reading a device tensor.

    FULL draft metadata describes the complete capture bucket. Direct draft
    execution slices QSA inputs to ``num_actual_tokens`` instead. Keep one
    length per block-table row, including empty requests, but exclude query
    rows beyond that packed input. Graph callbacks pass their captured row
    count, so they retain the complete bucket's boundaries.
    """
    lengths = getattr(metadata, "query_lens_cpu", None)
    if lengths is None:
        return None
    if lengths.device.type != "cpu":
        raise ValueError("QSA host query lengths must reside on CPU")
    values = lengths.tolist()
    if num_tokens < 0 or any(length < 0 for length in values) or sum(values) < num_tokens:
        raise ValueError("QSA query boundaries do not cover the packed input")
    remaining = num_tokens
    result = []
    for length in values:
        used = min(length, remaining)
        result.append(used)
        remaining -= used
    return result
