// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

#ifndef PLE_INLINE
#define PLE_INLINE inline
#endif

namespace ple_request {
constexpr int32_t kHeads = 16;
constexpr int32_t kHeadsPerOrder = 8;
constexpr int64_t kRowBytes = 320;

PLE_INLINE bool ValidToken(int64_t token, int64_t vocabulary) {
    return token >= 0 && token < vocabulary;
}

PLE_INLINE uint64_t Mix(int64_t current, int64_t recent, int64_t older,
                       int64_t eos, uint64_t m0, uint64_t m1, uint64_t m2, int32_t head) {
    if (recent == eos) older = eos;
    uint64_t mixed = static_cast<uint64_t>(current) * m0 ^ static_cast<uint64_t>(recent) * m1;
    if (head >= kHeadsPerOrder) mixed ^= static_cast<uint64_t>(older) * m2;
    return mixed;
}

PLE_INLINE int64_t Row(uint64_t mixed, int64_t size, int64_t offset) {
    return static_cast<int64_t>(mixed % static_cast<uint64_t>(size)) + offset;
}
}  // namespace ple_request
