// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

#ifndef QSA_GEOMETRY_INLINE
  #define QSA_GEOMETRY_INLINE inline
#endif

namespace qwen_geometry {
struct Geometry {
  int32_t groups;
  int32_t start;
  int32_t count;
};

// Match PyTorch INT32 add overflow before signed floor division by four.
// Widen before arithmetic so C++ signed overflow is never invoked.
QSA_GEOMETRY_INLINE Geometry Position(int32_t position) {
  int64_t next = static_cast<int64_t>(position) + 1;
  if (next > 2147483647LL) next -= 4294967296LL;
  int64_t groups = next / 4;
  if (next < 0 && next % 4 != 0) --groups;
  const int64_t start = groups * 4;
  return {static_cast<int32_t>(groups), static_cast<int32_t>(start), static_cast<int32_t>(next - start)};
}
}  // namespace qwen_geometry
