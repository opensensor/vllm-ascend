// SPDX-License-Identifier: Apache-2.0
#ifndef GLM_ADAPTIVE_EXPERT_CORE_GROUPS_H
#define GLM_ADAPTIVE_EXPERT_CORE_GROUPS_H
namespace GlmAdaptive {
constexpr unsigned DECODE_ROUTE_LIMIT = 64;  // MTP1: 4 requests * 2 rows * top-8.
constexpr unsigned LARGE_EXPERT_ROWS = 128;
struct Team { unsigned group, groups, lane, lanes; };
__aicore__ inline Team Plan(unsigned core, unsigned cores, unsigned requestedLanes,
                           unsigned totalRows, unsigned expertRows) {
  const bool split = totalRows > DECODE_ROUTE_LIMIT && expertRows < LARGE_EXPERT_ROWS &&
                     requestedLanes > 0 && requestedLanes <= cores && cores % requestedLanes == 0;
  const unsigned lanes = split ? requestedLanes : cores;
  return {core / lanes, cores / lanes, core % lanes, lanes};
}
}  // namespace GlmAdaptive
#endif
