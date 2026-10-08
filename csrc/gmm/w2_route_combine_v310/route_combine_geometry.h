// SPDX-License-Identifier: Apache-2.0
#ifndef W2_ROUTE_COMBINE_GEOMETRY_H
#define W2_ROUTE_COMBINE_GEOMETRY_H
#include <cstdint>
namespace NsW2Combine {
constexpr int64_t MAX_ROUTES = 32768;
constexpr int64_t MAX_WIDTH = 16384;
constexpr int64_t MAX_TOP_K = 32;
constexpr int64_t MAX_EXPERTS = 1024;
constexpr int64_t CHANNEL_ALIGNMENT = 32;
constexpr int64_t CHANNEL_TILE = 1024;
constexpr bool ValidGeometry(int64_t rows, int64_t hidden, int64_t tokens, int64_t topK, int64_t experts) {
    return rows > 0 && rows <= MAX_ROUTES && hidden > 0 && hidden <= MAX_WIDTH &&
           hidden % CHANNEL_ALIGNMENT == 0 && tokens > 0 && tokens <= MAX_ROUTES &&
           topK > 0 && topK <= MAX_TOP_K && rows == tokens * topK &&
           experts > 0 && experts <= MAX_EXPERTS;
}
}  // namespace NsW2Combine
#endif
