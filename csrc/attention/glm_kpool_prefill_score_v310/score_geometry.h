// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
namespace NsGlmKpoolPrefillScore {
constexpr int64_t HEADS = 32, DIM = 128, TILE = 64, POOL = 4, QUERY_TILE = 4, MAX_ROWS = 128;
inline bool ValidLayout(int64_t rows, int64_t pools, int64_t blocks, int64_t blockRows,
                        int64_t blockStride, int64_t rowStride, int64_t offset, int64_t elements) {
    return rows > 0 && rows <= MAX_ROWS && rows % QUERY_TILE == 0 && pools > 0 && pools % 8 == 0 &&
        blocks > 0 && blockRows > 0 && blockRows <= INT32_MAX && rowStride >= DIM &&
        rowStride % 16 == 0 && rowStride / 16 <= UINT16_MAX &&
        blockStride >= blockRows * rowStride && offset >= 0 && offset % 16 == 0 &&
        blockStride % 16 == 0 && elements > offset &&
        (blocks - 1) <= (elements - offset - 1) / blockStride &&
        (blocks - 1) * blockStride + (blockRows - 1) * rowStride + DIM <= elements - offset;
}
}
