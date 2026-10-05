// SPDX-License-Identifier: Apache-2.0
#ifndef QSA_POSITION_METADATA_AICPU_V310_POSITION_METADATA_H
#define QSA_POSITION_METADATA_AICPU_V310_POSITION_METADATA_H

#include <algorithm>
#include <cstdint>
#include <limits>

namespace qsa_position_metadata {

// C++ integer division truncates toward zero; QSA uses mathematical floor.
inline int64_t FloorDiv(int64_t value, int64_t divisor) {
    const int64_t quotient = value / divisor;
    return quotient - (value < 0 && value % divisor != 0);
}

// Output is four contiguous [rows] planes: visible groups, selected group
// count, causal-tail start, and causal-tail count. It matches
// _qsa_position_geometry plus group_counts.clamp_max(selected_width).
inline bool Compute(const int32_t* positions, int64_t rows, int64_t ratio,
                    int64_t capacity, int64_t selected_width,
                    int32_t* metadata) {
    if (rows < 0 || ratio <= 0 || capacity < 0 || selected_width < 0 ||
        selected_width > capacity ||
        (rows > 0 && (positions == nullptr || metadata == nullptr))) {
        return false;
    }
    for (int64_t row = 0; row < rows; ++row) {
        const int64_t next = static_cast<int64_t>(positions[row]) + 1;
        const int64_t complete = FloorDiv(next, ratio);
        const int64_t tail_start = complete * ratio;
        const int64_t tail_count = next - tail_start;
        const int64_t visible = std::min(complete, capacity);
        const int64_t selected = std::min(visible, selected_width);
        if (tail_start < std::numeric_limits<int32_t>::min() ||
            tail_start > std::numeric_limits<int32_t>::max() ||
            visible < std::numeric_limits<int32_t>::min() ||
            visible > std::numeric_limits<int32_t>::max() ||
            tail_count > std::numeric_limits<int32_t>::max()) {
            return false;
        }
        metadata[row] = static_cast<int32_t>(visible);
        metadata[rows + row] = static_cast<int32_t>(selected);
        metadata[2 * rows + row] = static_cast<int32_t>(tail_start);
        metadata[3 * rows + row] = static_cast<int32_t>(tail_count);
    }
    return true;
}

}  // namespace qsa_position_metadata

#endif  // QSA_POSITION_METADATA_AICPU_V310_POSITION_METADATA_H
