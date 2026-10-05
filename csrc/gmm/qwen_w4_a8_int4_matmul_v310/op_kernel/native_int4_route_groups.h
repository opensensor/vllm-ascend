// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef NATIVE_INT4_ROUTE_GROUPS_H
#define NATIVE_INT4_ROUTE_GROUPS_H

#include <cstdint>

namespace native_int4 {
constexpr uint32_t MAX_DIRECT_LOOKUP_EXPERTS = 128;
constexpr uint32_t MIN_DIRECT_LOOKUP_GROUPS = 32;
constexpr uint32_t MAX_ROUTE_ROWS = 128;

// Preserve first-seen expert order and original route order within each expert.
// Dense expert fan-out avoids scanning the growing expert list for every row.
template <bool ENABLE_DIRECT_LOOKUP, typename RouteIds>
__aicore__ inline uint32_t BuildRouteGroups(RouteIds cachedIds, uint32_t rows, int64_t experts,
                                             uint32_t broadcastFactor, int32_t* expertIds, uint32_t* counts,
                                             uint32_t* groupOfRow, uint32_t* starts, uint32_t* routeRows,
                                             uint32_t* sourceRows) {
  const bool canUseDirectLookup = ENABLE_DIRECT_LOOKUP && experts <= MAX_DIRECT_LOOKUP_EXPERTS;
  bool directLookup = false;
  int16_t groupByExpert[MAX_DIRECT_LOOKUP_EXPERTS];
  uint32_t groups = 0;
  for (uint32_t row = 0; row < rows; ++row) {
    const int32_t expert = cachedIds.GetValue(row);
    groupOfRow[row] = MAX_ROUTE_ROWS;
    if (expert < 0 || expert >= experts) continue;
    if (!directLookup && canUseDirectLookup && groups >= MIN_DIRECT_LOOKUP_GROUPS) {
      for (uint32_t index = 0; index < static_cast<uint32_t>(experts); ++index) groupByExpert[index] = -1;
      for (uint32_t group = 0; group < groups; ++group) groupByExpert[expertIds[group]] = static_cast<int16_t>(group);
      directLookup = true;
    }
    uint32_t group = 0;
    if (directLookup) {
      const int16_t knownGroup = groupByExpert[expert];
      group = knownGroup < 0 ? groups : static_cast<uint32_t>(knownGroup);
      if (knownGroup < 0) groupByExpert[expert] = static_cast<int16_t>(groups);
    } else {
      while (group < groups && expertIds[group] != expert) ++group;
    }
    if (group == groups) {
      expertIds[group] = expert;
      counts[group] = 0;
      ++groups;
    }
    groupOfRow[row] = group;
    ++counts[group];
  }
  starts[0] = 0;
  for (uint32_t group = 0; group < groups; ++group) {
    starts[group + 1] = starts[group] + counts[group];
    counts[group] = starts[group];
  }
  for (uint32_t row = 0; row < rows; ++row) {
    sourceRows[row] = row / broadcastFactor;
    if (groupOfRow[row] != MAX_ROUTE_ROWS) routeRows[counts[groupOfRow[row]]++] = row;
  }
  return groups;
}
}  // namespace native_int4

#endif  // NATIVE_INT4_ROUTE_GROUPS_H
