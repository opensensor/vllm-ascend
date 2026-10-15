// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

#ifndef CHECKPOINT_INLINE
#define CHECKPOINT_INLINE inline
#endif

namespace checkpoint_preparation {
struct Decision {
  int64_t destination;
  int64_t bias;
  int32_t normalized;
  bool copy;
  int32_t error;
};

CHECKPOINT_INLINE Decision Plan(int32_t previous, int32_t computed, int32_t accepted, int64_t block_size) {
  if (previous < -1 || computed < 0 || accepted < 1 || accepted > 3 || block_size < 1)
    return {0, 0, 1, false, 1};
  const int64_t current = (static_cast<int64_t>(computed) + 2) / block_size;
  const bool copy = previous >= 0 && previous != current;
  return {current, accepted - 1, copy ? 1 : accepted, copy, 0};
}
}  // namespace checkpoint_preparation
