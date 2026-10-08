// SPDX-License-Identifier: Apache-2.0
#ifndef W2_ROW_REUSE_GEOMETRY_H
#define W2_ROW_REUSE_GEOMETRY_H
namespace NsW2Rows {
constexpr unsigned TILE_ROWS = 128;
constexpr unsigned TILES = 2;
constexpr unsigned WINDOW_ROWS = TILE_ROWS * TILES;
__aicore__ inline unsigned WindowRows(unsigned total, unsigned first) {
    if (first >= total) return 0;
    return total - first < WINDOW_ROWS ? total - first : WINDOW_ROWS;
}
__aicore__ inline unsigned TileRows(unsigned total, unsigned first, unsigned tile) {
    const unsigned rows = WindowRows(total, first);
    if (tile >= TILES || tile * TILE_ROWS >= rows) return 0;
    return rows - tile * TILE_ROWS < TILE_ROWS ? rows - tile * TILE_ROWS : TILE_ROWS;
}
}  // namespace NsW2Rows
#endif
