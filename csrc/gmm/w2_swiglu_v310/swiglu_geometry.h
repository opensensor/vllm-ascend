// SPDX-License-Identifier: Apache-2.0
#ifndef W2_SWIGLU_GEOMETRY_H
#define W2_SWIGLU_GEOMETRY_H
#include <cstdint>
namespace NsW2Swiglu {
constexpr int64_t MAX_ROWS = 32768;
constexpr int64_t MAX_WIDTH = 16384;
constexpr int64_t ALIGNMENT = 32;
constexpr uint32_t TILE = 1024;
inline bool ValidGeometry(int64_t rows, int64_t gateUpWidth) {
    return rows > 0 && rows <= MAX_ROWS && gateUpWidth > 0 &&
           gateUpWidth <= 2 * MAX_WIDTH && gateUpWidth % (2 * ALIGNMENT) == 0;
}
}  // namespace NsW2Swiglu
#endif
