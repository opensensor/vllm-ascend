// SPDX-License-Identifier: Apache-2.0
#include "production_down_prefill.cpp"
#include <cstdio>
int main() {
  for (long tokens : {17, 37, 640, 641})
    for (long hidden : {256, 4096})
      for (long topK : {4, 8}) {
        const long rows = tokens * topK, valid = rows - 3, activeCores = hidden / 128 < 8 ? hidden / 128 : 8;
        std::vector<int32_t> ends = {int32_t(valid / 3), int32_t(valid / 3), int32_t(valid * 3 / 4), int32_t(valid)};
        std::vector<int64_t> order(rows);
        std::vector<int32_t> keys(rows), ranks(rows);
        for (long row = 0; row < rows; ++row) {
          order[row] = rows - 1 - row;
          ranks[order[row]] = row;
          keys[row] = row < ends[0] ? 0 : (row < ends[2] ? 2 : 3);
        }
        std::vector<float> weights(tokens * 8, 0), output(tokens * hidden + 16, 991);
        for (long t = 0; t < tokens; ++t)
          for (long slot = 0; slot < topK; ++slot) weights[t * 8 + slot] = float(slot + 1) / 8;
        std::vector<half> tiles(activeCores * rows * 128 + 16, 991);
        int64_t config[] = {tokens, topK, 4, hidden, 256, 256, 8, 0, 0};
        auto pointer = [](auto& vector) { return reinterpret_cast<GM_ADDR>(vector.data()); };
        auditRanks = ranks.data();
        gmRankReads = 0;
        for (unsigned core = 0; core < 8; ++core) {
          config[8] = core;  // CPU fixture owns the simulated hardware core register.
          glm_channel_down_prefill_v1(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, pointer(ends),
                                      pointer(order), pointer(keys), nullptr, pointer(tiles), pointer(ranks),
                                      pointer(weights), pointer(output), reinterpret_cast<GM_ADDR>(config));
        }
#ifdef GLM_CHANNEL_CACHE_PREFILL_ROUTES
        if (tokens <= 640 && topK == 8 && gmRankReads != 0) return 4;
        if ((tokens > 640 || topK != 8) && gmRankReads == 0) return 5;
#endif
        for (long t = 0; t < tokens; ++t)
          for (long c = 0; c < hidden; ++c) {
            float expected = 0;
            for (long slot = 0; slot < topK; ++slot) {
              const long row = ranks[t * topK + slot];
              if (row < valid) expected += projection_value(row, keys[row], c / 128, c % 128) * weights[t * 8 + slot];
            }
            if (output[t * hidden + c] != expected) {
              std::fprintf(stderr, "T=%ld H=%ld t=%ld c=%ld\n", tokens, hidden, t, c);
              return 1;
            }
          }
        for (long i = tokens * hidden; i < long(output.size()); ++i)
          if (output[i] != 991) return 2;
        for (long i = activeCores * rows * 128; i < long(tiles.size()); ++i)
          if (tiles[i] != 991) return 3;
      }
}
