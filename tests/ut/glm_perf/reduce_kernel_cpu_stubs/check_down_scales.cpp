// SPDX-License-Identifier: Apache-2.0
// CPU helper semantics; quantizer/Cube execution and device replay remain untested.
#include "glm_fused_down_scales.h"
#include "glm_fused_input_scales.h"
#include <cstring>
#include <iostream>
using namespace AscendC;
using namespace GlmFusedDownScales;
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
  static_assert(STORAGE_BYTES == 1152, "staging must use only the declared dead scratch slice");
  assert(MIN_ROWS == 5);
  for (unsigned groups : {4u, 8u, 16u, 64u, 128u}) {
    const unsigned elements = groups * ROWS, tiles = groups / GROUPS;
    // Repeat the sparse/dense/sparse boundary and then every usable density.
    std::vector<unsigned> counts{4, 5, 4, 31, 5, 31, 1, 2, 3};
    for (unsigned count = 6; count <= 30; ++count) counts.push_back(count);
    // Same GM and UB storage reused across launches. Poison every unused slice.
    std::vector<float> bank(elements + 16, fromBits(0x7fcabcdeu));
    std::vector<float> scratch(32 * 128 + 128, fromBits(0x7fcfedcbu));
    for (unsigned replay = 0; replay < counts.size(); ++replay) {
      const unsigned count = counts[replay];
      std::fill(bank.begin(), bank.end(), fromBits(0x7fcabcdeu));
      outputBegin = bank.data();
      outputWrites.assign(bank.size(), 0);
      std::vector<float> expected(elements, fromBits(0x7fcabcdeu));
      for (unsigned tile = 0; tile < tiles; ++tile) {
        std::fill(scratch.begin(), scratch.end(), fromBits(0x7fcfedcbu));
        LocalTensor<float> storage{scratch.data(), scratch.size()};
        std::vector<float> rows(TILE_ELEMENTS, 1.0f);
        for (unsigned row = 0; row < count; ++row)
          for (unsigned group = 0; group < GROUPS; ++group) {
            uint32_t raw = (row * groups + tile * GROUPS + group + replay * 17) * 2654435761u;
            const uint32_t special[] = {0, 0x80000000u, 0x7f800000u, 0x7fc12345u};
            if (row == 0) raw = special[group];
            rows[row * GROUPS + group] = fromBits(raw);
          }
        if (GroupMajor(count)) Prepare(storage);
        for (unsigned row = 0; row < count; row += QUANT_ROWS) {
          float quantizerScales[QUANT_ROWS * GROUPS];
          std::copy_n(rows.data() + row * GROUPS, QUANT_ROWS * GROUPS, quantizerScales);
          auto source = LocalTensor<float>{quantizerScales, QUANT_ROWS * GROUPS};
          if (GroupMajor(count))
            Stage(storage, source, row);
          else
            DataCopy(GlobalTensor<float>{bank.data() + tile * TILE_ELEMENTS + row * GROUPS}, source,
                     QUANT_ROWS * GROUPS);
          // Source quantization scratch is overwritten before the next block.
          std::fill_n(quantizerScales, QUANT_ROWS * GROUPS, fromBits(0x7fc55555u));
        }
        if (GroupMajor(count)) {
          gatherCalls = 0;
          Finish(storage, count);
          assert(gatherCalls == GROUPS);
          DataCopy(GlobalTensor<float>{bank.data() + tile * TILE_ELEMENTS}, storage[TRANSPOSE_OFFSET], TILE_ELEMENTS);
          for (unsigned group = 0; group < GROUPS; ++group)
            for (unsigned row = 0; row < ROWS; ++row)
              expected[tile * TILE_ELEMENTS + group * ROWS + row] = rows[(row < count ? row : 0) * GROUPS + group];
        } else {
          for (unsigned row = 0; row < (count + QUANT_ROWS - 1) / QUANT_ROWS * QUANT_ROWS; ++row)
            for (unsigned group = 0; group < GROUPS; ++group)
              expected[tile * TILE_ELEMENTS + row * GROUPS + group] = rows[row * GROUPS + group];
        }
        // Staging must not overwrite the rest of the input-scale bank or tables.
        for (unsigned i = STORAGE_BYTES / sizeof(float); i < scratch.size(); ++i)
          assert(bits(scratch[i]) == 0x7fcfedcbu);
      }
      assert(std::memcmp(bank.data(), expected.data(), elements * sizeof(float)) == 0);
      for (unsigned i = 0; i < elements; ++i) assert(outputWrites[i] == (bits(expected[i]) == 0x7fcabcdeu ? 0 : 1));
      for (unsigned i = elements; i < bank.size(); ++i) {
        assert(bits(bank[i]) == 0x7fcabcdeu);
        assert(outputWrites[i] == 0);
      }
      // Consume actual stored scales with the same helpers used by down's K loop.
      if (GroupMajor(count)) {
        uint32_t offsets[ROWS];
        float oldRows[ROWS], newRows[ROWS], oldBroadcast[ROWS * 8], newBroadcast[ROWS * 8];
        for (unsigned row = 0; row < ROWS; ++row) offsets[row] = (row < count ? row : 0) * GROUPS * sizeof(float);
        for (unsigned group = 0; group < groups; ++group) {
          std::vector<float> oldTile(TILE_ELEMENTS);
          for (unsigned row = 0; row < count; ++row)
            for (unsigned part = 0; part < GROUPS; ++part)
              oldTile[row * GROUPS + part] = bank[(group / GROUPS * GROUPS + part) * ROWS + row];
          auto old = LocalTensor<float>{oldTile.data() + group % GROUPS, TILE_ELEMENTS - group % GROUPS};
          auto current = LocalTensor<float>{bank.data() + group * ROWS, elements - group * ROWS};
          auto indices = LocalTensor<uint32_t>{offsets, ROWS};
          gatherCalls = 0;
          GlmFusedInputScales::BroadcastRows({oldBroadcast, ROWS * 8}, {oldRows, ROWS}, old, indices, ROWS, false);
          GlmFusedInputScales::BroadcastRows({newBroadcast, ROWS * 8}, {newRows, ROWS}, current, indices, ROWS, true);
          assert(gatherCalls == 1);
          assert(std::memcmp(oldBroadcast, newBroadcast, sizeof(oldBroadcast)) == 0);
          GlmFusedInputScales::CopyRows({oldRows, ROWS}, old, indices, ROWS, false);
          GlmFusedInputScales::CopyRows({newRows, ROWS}, current, indices, ROWS, true);
          assert(std::memcmp(oldRows, newRows, sizeof(oldRows)) == 0);
        }
      }
      outputBegin = nullptr;
    }
  }
  std::cout << "down scale bits, reused scratch, sparse/dense/sparse and consumer helpers passed\n";
}
