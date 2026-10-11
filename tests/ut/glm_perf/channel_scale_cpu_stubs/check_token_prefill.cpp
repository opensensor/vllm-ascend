// SPDX-License-Identifier: Apache-2.0
#include "production_down_token.cpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
int main() {
  for (long tokens : {17, 37, 640})
    for (long hidden : {256, 4096})
      for (bool noLocal : {false, true}) {
        const long rows = tokens * 8, valid = noLocal ? 0 : rows - 11;
        std::vector<int32_t> ends = {int32_t(valid / 3), int32_t(valid / 3), int32_t(valid * 3 / 4), int32_t(valid)};
        std::vector<int32_t> keys(rows), ranks(rows);
        for (long row = 0; row < rows; ++row) {
          ranks[rows - 1 - row] = row;
          keys[row] = row < ends[0] ? 0 : (row < ends[2] ? 2 : 3);
        }
        const long windows = (tokens + 15) / 16;
        std::vector<int32_t> sortedRows(windows * 128, rows);
        std::vector<int64_t> positions(windows * 128);
        for (long window = 0; window < windows; ++window) {
          auto start = positions.begin() + window * 128;
          std::iota(start, start + 128, 0);
          std::stable_sort(start, start + 128, [&](long a, long b) {
            long ga = window * 128 + a, gb = window * 128 + b;
            return (ga < rows ? ranks[ga] : rows) < (gb < rows ? ranks[gb] : rows);
          });
          for (long i = 0; i < 128; ++i) {
            long original = window * 128 + positions[window * 128 + i];
            sortedRows[window * 128 + i] = original < rows ? ranks[original] : rows;
          }
        }
        std::vector<float> weights(rows), output(tokens * hidden + 16, 991);
        for (long slot = 0; slot < rows; ++slot)
          weights[slot] = ranks[slot] < valid ? float(slot % 8 + 1) / 8 : std::numeric_limits<float>::quiet_NaN();
        std::vector<half> unusedTiles(16, 991);
        int64_t config[] = {tokens, 8, 4, hidden, 256, 256, 8, 0, 0};
        auto pointer = [](auto& vector) { return reinterpret_cast<GM_ADDR>(vector.data()); };
        for (unsigned core = 0; core < 8; ++core) {
          config[8] = core;
          glm_channel_down_prefill_v1(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, pointer(ends),
                                      pointer(sortedRows), pointer(keys), nullptr, pointer(unusedTiles),
                                      pointer(positions), pointer(weights), pointer(output),
                                      reinterpret_cast<GM_ADDR>(config));
        }
        for (long t = 0; t < tokens; ++t)
          for (long c = 0; c < hidden; ++c) {
            float expected = 0;
            for (long slot = 0; slot < 8; ++slot) {
              const long row = ranks[t * 8 + slot];
              if (row < valid) expected += projection_value(row, keys[row], c / 128, c % 128) * weights[t * 8 + slot];
            }
            if (output[t * hidden + c] != expected) {
              std::fprintf(stderr, "T=%ld H=%ld t=%ld c=%ld noLocal=%d\n", tokens, hidden, t, c, noLocal);
              return 1;
            }
          }
        for (long i = tokens * hidden; i < long(output.size()); ++i)
          if (output[i] != 991) return 2;
        for (half value : unusedTiles)
          if (value != 991) return 3;
      }
}
