# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense W3 byte-boundary and NZ-fragment contract for the 310P Cube path."""

import torch

from tools.deepseek_w2.w2_format import pack_codes, unpack_codes


def test_w3_field_vectors_restore_signed_codes_across_byte_boundaries():
    rows, tile_k = 16, 256
    expected = ((torch.arange(rows * tile_k).reshape(rows, tile_k) * 5 + 3) % 8 - 4).to(torch.int8)
    packed = pack_codes(expected, 3)
    flat = packed.flatten().to(torch.int16)
    field_vectors = []
    for field in range(8):
        bit = field * 3
        byte, shift = divmod(bit, 8)
        low_bits = min(3, 8 - shift)
        low_mask = (1 << low_bits) - 1
        groups = torch.arange(32)
        offsets = torch.arange(rows)[:, None] * 96 + groups[None, :] * 3 + byte
        values = (flat[offsets] >> shift) & low_mask
        if low_bits != 3:
            values |= (flat[offsets + 1] & ((1 << (3 - low_bits)) - 1)) << low_bits
        field_vectors.append(((values + 4) & 7) - 4)

    field_major = torch.stack(field_vectors).reshape(8, rows, 32)
    restored = field_major.permute(1, 2, 0).reshape(rows, tile_k)
    torch.testing.assert_close(restored, expected.to(torch.int16))
    torch.testing.assert_close(unpack_codes(packed, tile_k, 3), expected)

    # The existing Cube body transposes each [16,16] input fragment to NZ.
    fragments = restored.reshape(rows, tile_k // 16, 16).permute(1, 0, 2)
    torch.testing.assert_close(fragments[0], expected[:, :16].to(torch.int16))
    torch.testing.assert_close(fragments[-1], expected[:, -16:].to(torch.int16))
