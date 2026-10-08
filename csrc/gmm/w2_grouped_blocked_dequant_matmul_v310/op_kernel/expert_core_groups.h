// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef GLM_EXPERT_CORE_GROUPS_H
#define GLM_EXPERT_CORE_GROUPS_H

namespace NsW2 {

constexpr unsigned EXPERT_GROUP_MIN_ROWS = 256;
// The qualified MTP1 graphs verify up to four requests with two rows each.
// Keep all of their top-8 routes on the prior W3 schedule while testing
// resident prefill. This shape guard also retains it for short prefills;
// larger speculative graph profiles must revisit the bound explicitly.
constexpr unsigned W3_DECODE_MAX_REQUESTS = 4;
constexpr unsigned W3_DECODE_VERIFIER_ROWS = 2;
constexpr unsigned W3_ROUTES_PER_TOKEN = 8;
constexpr unsigned W3_DECODE_MAX_ROUTES =
    W3_DECODE_MAX_REQUESTS * W3_DECODE_VERIFIER_ROWS * W3_ROUTES_PER_TOKEN;

__aicore__ inline bool AllowResidentW3Prefill(unsigned codesPerByte, unsigned rows)
{
    return codesPerByte != 3 || rows > W3_DECODE_MAX_ROUTES;
}

struct ExpertCoreGroup {
    unsigned group;
    unsigned groups;
    unsigned lane;
    unsigned lanes;
};

// Each team visits a disjoint subset of active experts and stripes all their
// output tiles over its lanes. Private GM workspace remains indexed by the
// physical core, not the team lane. Small decode calls keep all cores together.
__aicore__ inline ExpertCoreGroup MakeExpertCoreGroup(
    unsigned core, unsigned cores, unsigned requestedLanes, unsigned rows)
{
    const unsigned lanes = rows >= EXPERT_GROUP_MIN_ROWS && requestedLanes > 0 &&
                               requestedLanes <= cores && cores % requestedLanes == 0
                               ? requestedLanes : cores;
    return {core / lanes, cores / lanes, core % lanes, lanes};
}

}  // namespace NsW2
#endif
