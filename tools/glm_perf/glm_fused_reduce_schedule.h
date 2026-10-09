// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
namespace GlmFusedReduceSchedule {
constexpr uint32_t MIN_TOKENS = 64, MAX_TOP_K = 8;
constexpr uint32_t METADATA_BYTES = MAX_TOP_K * (sizeof(int32_t) + sizeof(float));
}  // namespace GlmFusedReduceSchedule
