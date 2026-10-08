# SPDX-License-Identifier: Apache-2.0
"""CPU preparation of the native projection's invariant UB gather tables.

The first eight INT64 fields retain the existing launch ABI. A tagged, aligned
UINT32 payload follows; all per-token arithmetic remains in the native kernel.
"""

HEADER_WORDS = 12
LAYOUT_TAG = 0x474C4D4F464631
COLUMNS = 128
GROUP = 32
MAX_GROUPS = 128
LANES = 8
BULK_TOKENS = 16


def projection_offset_tables(header, *, gate_up, options):
    """Return immutable UINT32 tables in the native DMA order, without device reads."""
    if len(header) != 8 or any(type(v) is not int for v in header):
        raise ValueError("projection offsets require eight integer geometry fields")
    rows, experts, n, k, bits, activation_bits, tokens, top_k = header
    if (
        not 0 < rows <= 65536
        or experts <= 0
        or n <= 0
        or n % COLUMNS
        or not 0 < k <= 4096
        or k % 256
        or bits not in (2, 3, 4)
        or activation_bits not in (4, 8)
        or tokens <= 0
        or top_k <= 0
        or rows != tokens * top_k
        or type(gate_up) is not bool
    ):
        raise ValueError("invalid projection offset geometry")
    m = 32 if options.get("prefill_rows_32", False) else 16
    layouts = tuple(range(16, 2 * m + 1, 16)) if options.get("active_cube_rows", False) else (m, 2 * m)
    output_bytes = 4 if not gate_up and options.get("native_route_columns", False) else 2
    output = tuple((c // 2 + (COLUMNS // 2 if c % 2 else 0)) * output_bytes for c in range(COLUMNS))
    products = tuple((c // 16 * width * 16 + c % 16) * 4 for width in layouts for c in range(COLUMNS))
    if options.get("nz_prefill_accumulator", False):
        products = products[:COLUMNS] + tuple(b * 16 * 4 for b in range(4)) + products[COLUMNS + 4 :]
    scales = tuple((c % (COLUMNS // 2) // 16) * (k // GROUP) * 4 for c in range(COLUMNS))
    compact_down = (
        not gate_up
        and tokens > BULK_TOKENS
        and activation_bits == 4
        and options.get("route_compact_down_scales", False)
    )
    down_lanes = 4 if options.get("raw_hidden_scales", False) else LANES
    activation = tuple(
        (g // 4 * m * down_lanes + g % 4) * 4 if compact_down else g * LANES * 4 for g in range(MAX_GROUPS)
    )
    quant = ()
    if gate_up:
        batch = 16 if options.get("quad_hidden_quant", False) else 8
        elements = GROUP * batch
        quant = tuple(
            (g * GROUP + c - g % 2 * GROUP if g % 2 * GROUP <= c < (g % 2 + 1) * GROUP else elements) * 2
            for g in range(batch)
            for c in range(2 * GROUP)
        ) + tuple(g * 2 * 4 for g in range(batch))
    tables = (output, products, scales, activation, quant)
    if any(len(t) % LANES or any(not 0 <= v < 2**32 for v in t) for t in tables):
        raise ValueError("unaligned or overflowing projection offset table")
    return tables


def projection_descriptor(header, *, gate_up, options):
    """Pack the qualified table payload into an aligned INT64 config tensor."""
    tables = projection_offset_tables(header, gate_up=gate_up, options=options)
    payload = sum(tables, ())
    packed = tuple(payload[i] | payload[i + 1] << 32 for i in range(0, len(payload), 2))
    return (*header, LAYOUT_TAG, len(payload), 0, 0, *packed)
