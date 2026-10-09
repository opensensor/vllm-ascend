// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"
namespace GlmRouteInput {
constexpr unsigned ROWS = 32, ACTIVE_ROWS = ROWS - 1, MAX_GROUPS = 128;
constexpr unsigned ROW_BYTES = 32, PAIR_BYTES = 2 * ROWS * ROW_BYTES;
constexpr unsigned GROUP_MAJOR_MIN_ROWS = 5;
constexpr unsigned SCALE_ELEMENTS = ROWS * MAX_GROUPS;
constexpr unsigned GROUP_MAJOR_TILE_BYTES = SCALE_ELEMENTS * sizeof(float);
constexpr unsigned GROUP_MAJOR_INDEX_BYTES = ROWS * sizeof(unsigned);
// Dense batches store [group, row]; sparse batches retain [row, group].
// Both stages derive the same choice from device-resident expert boundaries.
__aicore__ inline bool GroupMajorScales(unsigned count) { return count >= GROUP_MAJOR_MIN_ROWS; }
// Batch slots first/31 + expert + batch are disjoint for monotonic expert ends.
// Consumers derive this index on device; no route counts travel to the host.
}  // namespace GlmRouteInput
