// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

#ifndef QSA_GEOMETRY_INLINE
  #define QSA_GEOMETRY_INLINE inline
#endif

namespace qwen_geometry_v2 {
struct Geometry {
  int64_t groups;
  int64_t start;
  int64_t count;
};

// Preserve PyTorch INT64 wrapping add, then exact signed floor by four.
// The INT64_MIN multiple of four keeps every product within signed range.
QSA_GEOMETRY_INLINE Geometry Position(int64_t position) {
  const int64_t maximum = 9223372036854775807LL;
  const int64_t minimum = -9223372036854775807LL - 1;
  const int64_t next = position == maximum ? minimum : position + 1;
  int64_t groups = next / 4;
  if (next < 0 && next % 4 != 0) --groups;
  const int64_t start = groups * 4;
  return {groups, start, next - start};
}
}  // namespace qwen_geometry_v2
