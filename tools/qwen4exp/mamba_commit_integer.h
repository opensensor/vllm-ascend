// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

#ifndef MAMBA_COMMIT_INLINE
  #define MAMBA_COMMIT_INLINE inline
#endif

namespace mamba_commit {
struct Decision {
  int64_t destination;
  int64_t bias;
  int32_t next_count;
  bool copy;
};

// Inputs are admitted nonnegative INT32 values; use INT64 intermediates.
MAMBA_COMMIT_INLINE Decision Plan(int32_t source, int32_t computed, int32_t scheduled, int32_t drafts, int32_t accepted,
                                  int64_t block_size) {
  const int64_t running = static_cast<int64_t>(computed) + scheduled - drafts;
  const int64_t aligned = (running + accepted - 1) / block_size * block_size;
  const int64_t destination = aligned / block_size - 1;
  const int64_t bias = aligned - running;
  const bool reaches = aligned >= running;
  return {destination, bias, reaches && source == destination ? 1 : accepted,
          reaches && (source != destination || bias != 0)};
}
}  // namespace mamba_commit
