// SPDX-License-Identifier: Apache-2.0
// Execute the real producer and consumer helpers with CPU copy/vector semantics.
#include "kernel_operator.h"
#include "glm_fused_input_scales.h"
#include "glm_route_input_layout.h"
#include <cstring>
#include <iostream>
#include <numeric>
extern "C" void legacy_route(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
extern "C" void grouped_route(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
using namespace AscendC;
using namespace GlmRouteInput;
using RouteFn = void (*)(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
template <typename T>
GM_ADDR address(std::vector<T>& value) {
  return reinterpret_cast<GM_ADDR>(value.data());
}
uint32_t bits(float value) {
  uint32_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}
float fromBits(uint32_t value) {
  float result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}
int main() {
  for (unsigned groups : {8u, 16u, 64u, 128u}) {
    for (int64_t tokens : {64, 640, 1280}) {
      constexpr int64_t topK = 8, experts = 72;
      const int64_t routes = tokens * topK, slots = routes / ACTIVE_ROWS + experts + 1;
      std::vector<int64_t> config{routes, experts, 4096, groups * 32, 3, 4, tokens, topK};
      std::vector<int64_t> counts(experts, 0), ends(experts), order(routes);
      std::vector<int8_t> low(tokens * groups * ROW_BYTES);
      std::vector<float> scales(tokens * groups);
      const int initialCounts[] = {0, 1, 3, 4, 5, 7, 16, 17, 30, 31, 32, 61, 62};
      std::copy(std::begin(initialCounts), std::end(initialCounts), counts.begin());
      for (size_t i = 0; i < low.size(); ++i) low[i] = static_cast<int8_t>(i * 37);
      for (unsigned replay = 0; replay < 3; ++replay) {
        if (replay == 2) std::fill(counts.begin(), counts.end(), 0);
        if (replay) std::rotate(counts.begin(), counts.begin() + 4, counts.end());
        std::partial_sum(counts.begin(), counts.end(), ends.begin());
        for (int64_t i = 0; i < routes; ++i) order[i] = ((i * 13 + replay * 17) % tokens) * topK + i % topK;
        for (size_t i = 0; i < scales.size(); ++i)
          scales[i] = fromBits(static_cast<uint32_t>(i * 2654435761u + replay));
        const uint32_t special[] = {0, 0x80000000u, 0x7f800000u, 0x7fc12345u};
        for (int64_t token = 0; token < tokens; ++token)
          for (unsigned g = 0; g < 4; ++g) scales[token * groups + g] = fromBits(special[g]);
        std::vector<int8_t> packedOld(slots * (groups / 2) * PAIR_BYTES, 93), packedNew(packedOld);
        std::vector<float> sxOld(slots * SCALE_ELEMENTS, fromBits(0x7fc98765u)), sxNew(sxOld);
        for (auto variant : {0, 1}) {
          auto& packed = variant ? packedNew : packedOld;
          auto& sx = variant ? sxNew : sxOld;
          RouteFn operation = variant ? grouped_route : legacy_route;
          outputBegin = sx.data();
          outputWrites.assign(sx.size(), 0);
          for (blockIndex = 0; blockIndex < blockCount; ++blockIndex) {
            allocatedBytes = 0;
            operation(address(low), address(scales), address(order), address(ends), address(packed), address(sx),
                      address(config));
            assert(allocatedBytes == (variant ? 2 * GROUP_MAJOR_TILE_BYTES + GROUP_MAJOR_INDEX_BYTES
                                              : PAIR_BYTES + GROUP_MAJOR_TILE_BYTES));
          }
          std::vector<bool> writtenSlots(slots, false);
          int64_t first = 0;
          for (int64_t expert = 0; expert < experts; ++expert) {
            for (int64_t row = first; row < ends[expert]; row += ACTIVE_ROWS) {
              const unsigned count = std::min<int64_t>(ACTIVE_ROWS, ends[expert] - row);
              const int64_t slot = first / ACTIVE_ROWS + expert + (row - first) / ACTIVE_ROWS;
              assert(!writtenSlots[slot]);
              writtenSlots[slot] = true;
              for (unsigned i = 0; i < SCALE_ELEMENTS; ++i) {
                const bool grouped = variant && count >= GROUP_MAJOR_MIN_ROWS;
                const unsigned g = grouped ? i / ROWS : i % MAX_GROUPS;
                const unsigned m = grouped ? i % ROWS : i / MAX_GROUPS;
                const unsigned selected = grouped && m >= count ? 0 : m;
                const uint32_t expected =
                    g < groups && selected < count ? bits(scales[(order[row + selected] / topK) * groups + g]) : 0;
                assert(bits(sx[slot * SCALE_ELEMENTS + i]) == expected);
                assert(outputWrites[slot * SCALE_ELEMENTS + i] == 1);
              }
              for (unsigned pair = 0; pair < groups / 2; ++pair)
                for (unsigned i = 0; i < PAIR_BYTES; ++i) {
                  const unsigned codeRow = i / ROW_BYTES;
                  const int8_t expected =
                      codeRow < 2 * count
                          ? low[((order[row + codeRow % count] / topK) * groups + pair * 2 + codeRow / count) *
                                    ROW_BYTES +
                                i % ROW_BYTES]
                          : 0;
                  assert(packed[(slot * (groups / 2) + pair) * PAIR_BYTES + i] == expected);
                }
              if (variant && count >= GROUP_MAJOR_MIN_ROWS) {
                uint32_t indices[ROWS];
                float oldRows[ROWS], newRows[ROWS], oldBroadcast[ROWS * 8], newBroadcast[ROWS * 8];
                for (unsigned m = 0; m < ROWS; ++m) indices[m] = (m < count ? m : 0) * MAX_GROUPS * sizeof(float);
                for (unsigned g = 0; g < groups; ++g) {
                  auto old = LocalTensor<float>{sxOld.data() + slot * SCALE_ELEMENTS + g, SCALE_ELEMENTS - g};
                  auto current =
                      LocalTensor<float>{sx.data() + slot * SCALE_ELEMENTS + g * ROWS, SCALE_ELEMENTS - g * ROWS};
                  auto offsets = LocalTensor<uint32_t>{indices, ROWS};
                  gatherCalls = 0;
                  GlmFusedInputScales::BroadcastRows({oldBroadcast, ROWS * 8}, {oldRows, ROWS}, old, offsets, ROWS,
                                                     false);
                  assert(gatherCalls == 1);
                  GlmFusedInputScales::BroadcastRows({newBroadcast, ROWS * 8}, {newRows, ROWS}, current, offsets, ROWS,
                                                     true);
                  assert(gatherCalls == 1);  // Direct consumer performs no Gather.
                  assert(std::memcmp(oldBroadcast, newBroadcast, sizeof(oldBroadcast)) == 0);
                  GlmFusedInputScales::CopyRows({oldRows, ROWS}, old, offsets, ROWS, false);
                  GlmFusedInputScales::CopyRows({newRows, ROWS}, current, offsets, ROWS, true);
                  assert(std::memcmp(oldRows, newRows, sizeof(oldRows)) == 0);
                }
              }
            }
            first = ends[expert];
          }
          for (int64_t slot = 0; slot < slots; ++slot) {
            if (writtenSlots[slot]) continue;
            for (unsigned i = 0; i < SCALE_ELEMENTS; ++i) {
              assert(outputWrites[slot * SCALE_ELEMENTS + i] == 0);
              assert(bits(sx[slot * SCALE_ELEMENTS + i]) == 0x7fc98765u);
            }
            for (unsigned i = 0; i < (groups / 2) * PAIR_BYTES; ++i)
              assert(packed[slot * (groups / 2) * PAIR_BYTES + i] == 93);
          }
        }
        assert(packedOld == packedNew);
        outputBegin = nullptr;
      }
    }
  }
  std::cout << "producer bits, paired consumers, changed metadata, UB bounds and packed codes passed\n";
}
