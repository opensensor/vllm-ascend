// SPDX-License-Identifier: Apache-2.0
#include "../../csrc/attention/qsa_position_metadata_aicpu_v310/position_metadata.h"

#include <cstdint>

extern "C" int qsa_position_metadata_host(const int32_t* positions, int64_t rows, int64_t ratio, int64_t capacity,
                                          int64_t selected_width, int32_t* metadata) {
  return qsa_position_metadata::Compute(positions, rows, ratio, capacity, selected_width, metadata) ? 0 : 1;
}
