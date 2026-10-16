// SPDX-License-Identifier: Apache-2.0
// K32 scale replication only; caller retains indices across every batch reader.
#pragma once
#include "kernel_operator.h"
#include "glm_fused_route_cache.h"
#include "glm_streaming_contract.h"

namespace GlmStreamingScaleGather {
using namespace AscendC;
constexpr uint32_t ROWS = GlmStreamingContract::ROWS;
constexpr uint32_t LANES = 8;
constexpr uint32_t ELEMENTS = ROWS * LANES;
constexpr uint32_t EXPANDED_BYTES = ELEMENTS * sizeof(uint32_t);
constexpr uint32_t GROUP_SOURCE_BYTES = ROWS * sizeof(uint32_t);
using RouteCache = GlmFusedRouteCache::Layout<ROWS>;
constexpr uint32_t EXPANDED_OFFSET = RouteCache::END_OFFSET + RouteCache::END_BYTES;
constexpr uint32_t GROUP_SOURCE_OFFSET = EXPANDED_OFFSET + EXPANDED_BYTES;
static_assert(ROWS == 32 && ELEMENTS == 256, "scale Gather uses the frozen M32 broadcast geometry");
static_assert(GlmStreamingContract::BROADCAST_OFFSET + ELEMENTS * sizeof(float) <=
                  GlmStreamingContract::RETAINED_INDEX_OFFSET,
              "scale broadcast overlaps retained row indices");
static_assert(RouteCache::END_OFFSET >= GlmStreamingContract::RETAINED_INDEX_OFFSET + GROUP_SOURCE_BYTES,
              "expert ends overlap retained row indices");
static_assert(GlmStreamingContract::VERSION == 2 &&
                  EXPANDED_OFFSET == GlmStreamingContract::SCALE_GATHER_INDEX_OFFSET &&
                  GROUP_SOURCE_OFFSET == GlmStreamingContract::GROUP_SCALE_SOURCE_OFFSET &&
                  EXPANDED_BYTES == GlmStreamingContract::SCALE_GATHER_INDEX_BYTES &&
                  GROUP_SOURCE_BYTES == GlmStreamingContract::GROUP_SCALE_SOURCE_BYTES,
              "scale indices must follow the complete retained expert-end cache");
static_assert(EXPANDED_OFFSET % RouteCache::DMA_BYTES == 0 && GROUP_SOURCE_OFFSET % RouteCache::DMA_BYTES == 0 &&
                  GROUP_SOURCE_OFFSET + GROUP_SOURCE_BYTES <= RouteCache::MASK_BYTES,
              "scale Gather indices exceed aligned retained mask scratch");

// Source contains 32 byte offsets relative to each sx view. Row-major callers
// provide their existing stride/count-clamped row indices. Group-major callers
// provide immutable row*4 indices, preserving all literal padded source rows.
// Drain ALL old consumers before changing source or expanded; prepare once per
// batch/layout, then retain expanded across both groups and every K pair/up.
__aicore__ inline void Prepare(const LocalTensor<uint32_t>& expanded, const LocalTensor<uint32_t>& sourceIndices) {
  // Source may have scalar or DMA writers; this fence is outside the K loop.
  PipeBarrier<PIPE_ALL>();
  Brcb(expanded, sourceIndices, ROWS / LANES, {1, LANES});
  PipeBarrier<PIPE_V>();
}

// All arguments are disjoint. Source sx already selects the current K32 group;
// expanded offsets use bytes and dst has exactly 32 rows of eight FP32 lanes.
// No FP32 arithmetic, host reads, scalar index construction or allocation.
// Gather changes vector masks; caller must configure subsequent masked Casts.
__aicore__ inline void Broadcast(const LocalTensor<float>& dst, const LocalTensor<float>& sx,
                                 const LocalTensor<uint32_t>& expanded) {
  Gather(dst, sx, expanded, static_cast<uint32_t>(0), ELEMENTS);
  PipeBarrier<PIPE_V>();
}
}  // namespace GlmStreamingScaleGather
