// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"
namespace GlmFusedCompactScales {
constexpr uint32_t ROWS = 32, GROUPS_PER_TILE = 4;
__aicore__ inline uint32_t GroupOffset(uint32_t group) {
  return group / GROUPS_PER_TILE * ROWS * GROUPS_PER_TILE + group % GROUPS_PER_TILE;
}
}  // namespace GlmFusedCompactScales
