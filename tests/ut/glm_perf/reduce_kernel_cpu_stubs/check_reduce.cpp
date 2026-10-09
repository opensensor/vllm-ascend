// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
using Kernel = void (*)(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
extern "C" void legacy_reduce(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
extern "C" void cached_reduce(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
template <typename T>
GM_ADDR address(std::vector<T>& values) {
  return reinterpret_cast<GM_ADDR>(values.data());
}
void check(unsigned tokens, unsigned topK, unsigned width, bool empty) {
  const unsigned routes = tokens * topK, localRows = empty ? 0 : routes - 3;
  std::vector<half> workspace(routes * width, static_cast<half>(std::numeric_limits<float>::quiet_NaN()));
  std::vector<int32_t> ranks(routes);
  std::vector<int64_t> ends{localRows}, order(routes), config{routes, 1, width, 256, 4, 4, tokens, topK};
  std::vector<float> weights(routes), expected(tokens * width, 0), output(tokens * width);
  for (unsigned row = 0; row < routes; ++row) {
    ranks[row] = row % 11 == 0 ? -1 : row;
    order[row] = routes - row - 1;
    weights[row] = static_cast<float>(static_cast<int>(row % 13) - 6) * 0.01703089f;
    if (row < localRows)
      for (unsigned column = 0; column < width; ++column)
        workspace[row * width + column] = static_cast<half>((static_cast<int>((row + column) % 31) - 15) / 7.01f);
  }
  // Replay-equivalent inputs change in place between calls, including an all-peer
  // suffix. This checks CPU kernel semantics only; it does not emulate ACL graphs.
  for (unsigned change = 0; change < 2; ++change) {
    if (change) {
      std::reverse(weights.begin(), weights.end());
      for (auto& rank : ranks)
        if (rank >= 0 && static_cast<unsigned>(rank) < localRows) rank = localRows - 1 - rank;
    }
    std::fill(expected.begin(), expected.end(), 0);
    for (unsigned token = 0; token < tokens; ++token)
      for (unsigned slot = 0; slot < topK; ++slot) {
        int32_t row = ranks[token * topK + slot];
        if (row < 0 || static_cast<unsigned>(row) >= localRows) continue;
        for (unsigned column = 0; column < width; ++column) {
          unsigned physical = column;
#ifdef GLM_NATIVE_ROUTE_COLUMNS
          unsigned lane = column % 128;
          physical = column / 128 * 128 + lane / 2 + (lane % 2 ? 64 : 0);
#endif
          float product = static_cast<float>(workspace[row * width + physical]) * weights[order[row]];
          expected[token * width + column] += product;
        }
      }
    uint64_t reads[2];
    Kernel kernels[2] = {legacy_reduce, cached_reduce};
    for (unsigned mode = 0; mode < 2; ++mode) {
      std::fill(output.begin(), output.end(), std::numeric_limits<float>::quiet_NaN());
      AscendC::outputBegin = output.data();
      AscendC::outputWrites.assign(output.size(), 0);
      AscendC::scalarReads = 0;
      for (unsigned core = 0; core < AscendC::blockCount; ++core) {
        AscendC::blockIndex = core;
        kernels[mode](address(workspace), address(ranks), address(ends), address(output), address(config),
                      address(order), address(weights));
      }
      assert(std::memcmp(output.data(), expected.data(), output.size() * sizeof(float)) == 0);
      assert(
          std::all_of(AscendC::outputWrites.begin(), AscendC::outputWrites.end(), [](unsigned n) { return n == 1; }));
      reads[mode] = AscendC::scalarReads;
    }
    bool eligible = tokens >= 64 && tokens % 8 == 0 && topK <= 8;
    unsigned tiles = width / 128;
    // Eight invariant localRows reads remain. All other scalar GM metadata reads
    // are performed once per token instead of once per token/column tile.
    assert(eligible ? reads[0] - 8 == (reads[1] - 8) * tiles : reads[0] == reads[1]);
  }
}
int main() {
  for (unsigned tokens : {17u, 31u, 64u, 65u, 128u, 640u, 1280u})
    for (unsigned topK : {1u, 2u, 8u, 9u})
      for (bool empty : {false, true}) check(tokens, topK, tokens == 64 ? 4096 : 256, empty);
  std::cout << "bit-exact outputs, single writes, poisoned peers, changed metadata, metadata read counts passed\n";
}
