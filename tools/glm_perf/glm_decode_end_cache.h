// SPDX-License-Identifier: Apache-2.0
// Launch-local M16 expert boundaries; caller owns the existing mask subview.
#pragma once
#include "kernel_operator.h"
#include "glm_fused_route_cache.h"

namespace GlmDecodeEndCache {
using namespace AscendC;
using Layout = GlmFusedRouteCache::Layout<16>;
constexpr uint32_t END_OFFSET = Layout::END_OFFSET;
constexpr uint32_t END_BYTES = Layout::END_BYTES;
constexpr uint32_t MAX_EXPERTS = Layout::MAX_EXPERTS;
constexpr uint32_t ENDS_PER_DMA = Layout::ENDS_PER_DMA;
static_assert(END_OFFSET == 24640 && END_OFFSET + END_BYTES == 26944,
              "decode boundaries must retain the audited M16 mask geometry");

// Caller admits 1..MAX_EXPERTS and prepared M16/no-LUT mask lifetimes. Reload
// before tile scans on every launch/replay, including all-peer routes. cached
// already selects mask_[END_OFFSET]; no buffer allocation or GM overread.
__aicore__ inline void Prepare(const LocalTensor<int64_t>& cached, const GlobalTensor<int64_t>& ends,
                               uint32_t experts) {
  const uint32_t aligned = experts / ENDS_PER_DMA * ENDS_PER_DMA;
  if (aligned) DataCopy(cached, ends, aligned);
  PipeBarrier<PIPE_ALL>();
  for (uint32_t expert = aligned; expert < experts; ++expert) cached.SetValue(expert, ends.GetValue(expert));
  PipeBarrier<PIPE_ALL>();
}

// This removes repeated GM reads, not scalar instructions: GetValue remains UB
// scalar work. Caller restricts expert to the freshly prepared live count.
__aicore__ inline int64_t Read(const LocalTensor<int64_t>& cached, uint32_t expert) { return cached.GetValue(expert); }
}  // namespace GlmDecodeEndCache
