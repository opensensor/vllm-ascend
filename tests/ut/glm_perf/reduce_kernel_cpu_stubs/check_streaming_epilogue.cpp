// SPDX-License-Identifier: Apache-2.0
// Compile the actual helper; model the old GM clear/store independently.
#define GLM_STREAMING_DEVICE_API
#include "glm_streaming_epilogue.h"
#include <cstring>
#include <iostream>
#include <string>
using namespace AscendC;
using namespace GlmStreamingEpilogue;

void check(unsigned count, unsigned replay, std::vector<uint64_t>& backing, std::vector<int8_t>& gm,
           bool zeroCodes = false) {
  assert(count <= GlmStreamingContract::MAX_ROWS);
  // A full 16 KiB conversion allocation: only its dead first 4 KiB may change.
  auto bytes = reinterpret_cast<int8_t*>(backing.data());
  LocalTensor<int8_t> tile{bytes + sizeof(uint64_t), GlmStreamingContract::CONVERTED_BYTES};
  std::vector<int8_t> expected(TILE_BYTES, 0);
  trace = {};
  Clear(tile);
  for (unsigned row = 0; row < count; row += QUANT_ROWS) {
    std::vector<int8_t> source(QUANT_ROWS * GROUPS * RECORD_BYTES + RECORD_BYTES, static_cast<int8_t>(0xd7));
    for (unsigned part = 0; part < QUANT_ROWS; ++part)
      for (unsigned group = 0; group < GROUPS; ++group)
        for (unsigned b = 0; b < RECORD_BYTES; ++b) {
          const unsigned sourceOffset = (part * GROUPS + group) * RECORD_BYTES + b;
          const unsigned value = zeroCodes ? 0 : (replay * 43 + (row + part) * 73 + group * 89 + b * 7) & 255;
          source[sourceOffset] = static_cast<int8_t>(value);
          if (row + part < count) {
            // Qualified old StoreHiddenCodes after a full zero GM pair clear.
            const unsigned oldOffset = group / 2 * PAIR_BYTES + (group % 2 * count + row + part) * RECORD_BYTES + b;
            expected[oldOffset] = static_cast<int8_t>(value);
          }
        }
    Stage(tile, {source.data(), source.size()}, row, count);
    // The next quantizer may overwrite its entire source; no tile alias exists.
    std::fill(source.begin(), source.end(), static_cast<int8_t>(0xee));
  }
  // No empty transfer or unsigned count-row underflow at/after the tail.
  const auto beforeEmpty = trace;
  Stage(tile, {nullptr, 0}, count, count);
  Stage(tile, {nullptr, 0}, count + QUANT_ROWS, count);
  assert(trace.localCalls == beforeEmpty.localCalls && trace.localBytes == beforeEmpty.localBytes);
  assert(trace.vectorBarriers == beforeEmpty.vectorBarriers && trace.allBarriers == beforeEmpty.allBarriers);
  assert(std::memcmp(tile.data, expected.data(), TILE_BYTES) == 0);
  assert(backing.front() == 0xa5a5a5a5a5a5a5a5ull && backing.back() == 0xa5a5a5a5a5a5a5a5ull);
  for (unsigned i = TILE_BYTES; i < GlmStreamingContract::CONVERTED_BYTES; ++i)
    assert(static_cast<uint8_t>(tile.data[i]) == 0xa5);
  const auto initial = gm;
  globalBase = gm.data();
  globalWrites.assign(gm.size(), 0);
  // Two nonadjacent regions prove that Store uses caller offsets independently.
  const uint64_t first = PAIR_BYTES, second = 5 * PAIR_BYTES;
  Store({gm.data()}, tile, first, second);
  assert(std::memcmp(gm.data() + first, expected.data(), PAIR_BYTES) == 0);
  assert(std::memcmp(gm.data() + second, expected.data() + PAIR_BYTES, PAIR_BYTES) == 0);
  for (unsigned i = 0; i < gm.size(); ++i) {
    const bool owned = (i >= first && i < first + PAIR_BYTES) || (i >= second && i < second + PAIR_BYTES);
    assert(globalWrites[i] == (owned ? 1u : 0u));
    if (!owned) assert(gm[i] == initial[i]);
  }
  const unsigned quantBatches = (count + QUANT_ROWS - 1) / QUANT_ROWS;
  assert(trace.localCalls == GROUPS * quantBatches && trace.localBytes == GROUPS * count * RECORD_BYTES);
  assert(trace.globalCalls == PAIRS && trace.globalBytes == TILE_BYTES);
  assert(trace.allBarriers == 2 && trace.vectorBarriers == 1 + quantBatches);
  // Another output core owns disjoint pairs and cannot change the first core.
  const auto previous = gm;
  Store({gm.data()}, tile, 2 * PAIR_BYTES, 6 * PAIR_BYTES);
  assert(std::memcmp(gm.data() + first, previous.data() + first, PAIR_BYTES) == 0);
  assert(std::memcmp(gm.data() + second, previous.data() + second, PAIR_BYTES) == 0);
  globalBase = nullptr;
  std::cout << "{\"count\":" << count << ",\"old_gm_calls\":" << PAIRS + GROUPS * count
            << ",\"old_gm_bytes\":" << TILE_BYTES + GROUPS * count * RECORD_BYTES
            << ",\"new_gm_calls\":2,\"new_gm_bytes\":4096,\"old_local_calls\":" << GROUPS * count
            << ",\"local_calls\":" << GROUPS * quantBatches << ",\"local_bytes\":" << GROUPS * count * RECORD_BYTES
            << "}\n";
}

int main(int argc, char** argv) {
  std::vector<uint64_t> backing(GlmStreamingContract::CONVERTED_BYTES / sizeof(uint64_t) + 2, 0xa5a5a5a5a5a5a5a5ull);
  std::vector<int8_t> gm(8 * PAIR_BYTES, static_cast<int8_t>(0xc3));
  if (argc == 2 && std::string(argv[1]) == "zero") {
    check(31, 0, backing, gm, true);
    check(5, 1, backing, gm, true);
    return 0;
  }
  if (argc == 2 && std::string(argv[1]) != "replay") {
    check(static_cast<unsigned>(std::stoul(argv[1])), 1, backing, gm);
    return 0;
  }
  // Reuse one logical launch geometry, including all-peer/zero-local drain.
  unsigned replay = 0;
  for (unsigned count : {31u, 5u, 4u, 0u, 1u, 4u, 5u, 15u, 16u, 17u, 30u, 31u}) check(count, replay++, backing, gm);
}
