// SPDX-License-Identifier: Apache-2.0
#ifndef GLM_MHC_POST_GEOMETRY_H
#define GLM_MHC_POST_GEOMETRY_H
#include <cstdint>
namespace NsGlmMhcPost {
constexpr int64_t STREAMS = 4;
constexpr int64_t MAX_ROWS = 32768;
constexpr int64_t MAX_WIDTH = 16384;
constexpr int64_t ALIGNMENT = 32;
constexpr uint32_t TILE = 1024;
inline bool ValidGeometry(int64_t rows, int64_t width) {
    return rows > 0 && rows <= MAX_ROWS && width > 0 && width <= MAX_WIDTH && width % ALIGNMENT == 0;
}
}  // namespace NsGlmMhcPost
#endif
