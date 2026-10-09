// SPDX-License-Identifier: Apache-2.0
// Retained routing metadata occupies unused mask scratch, not a new UB buffer.
#pragma once
#include <cstdint>
namespace GlmFusedRouteCache {
template <uint32_t Rows>
struct Layout {
  static_assert(Rows == 16 || Rows == 32, "route cache requires the existing Cube schedules");
  static constexpr uint32_t MASK_BYTES = 128 * 256;
  static constexpr uint32_t RECONSTRUCTION_BYTES = MASK_BYTES * 3 / 4;
  static constexpr uint32_t ROW_INDEX_BYTES = Rows * sizeof(uint32_t);
  static constexpr uint32_t DMA_BYTES = 32;
  static constexpr uint32_t MAX_EXPERTS = 288;
  static constexpr uint32_t END_OFFSET =
      (RECONSTRUCTION_BYTES + ROW_INDEX_BYTES + DMA_BYTES - 1) / DMA_BYTES * DMA_BYTES;
  static constexpr uint32_t END_BYTES = MAX_EXPERTS * sizeof(int64_t);
  static constexpr uint32_t ENDS_PER_DMA = DMA_BYTES / sizeof(int64_t);
  static_assert(END_OFFSET % DMA_BYTES == 0 && END_OFFSET + END_BYTES <= MASK_BYTES,
                "expert boundaries exceed retained mask scratch");
};
}  // namespace GlmFusedRouteCache
