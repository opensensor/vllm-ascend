// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace GlmRouteInput {
constexpr unsigned ROWS = 32, ACTIVE_ROWS = ROWS - 1, MAX_GROUPS = 128;
constexpr unsigned ROW_BYTES = 32, PAIR_BYTES = 2 * ROWS * ROW_BYTES;
// Batch slots first/31 + expert + batch are disjoint for monotonic expert ends.
// Consumers derive this index on device; no route counts travel to the host.
}  // namespace GlmRouteInput
