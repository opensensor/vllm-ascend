// SPDX-License-Identifier: Apache-2.0
#include "glm_decode_end_cache.h"
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <string>
#include <type_traits>

namespace {
uint64_t allocations = 0;
constexpr uint32_t GUARD_BYTES = 32;
using Cache = GlmDecodeEndCache::Layout;
static_assert(
    std::is_same_v<decltype(&GlmDecodeEndCache::Prepare),
                   void (*)(const AscendC::LocalTensor<int64_t>&, const AscendC::GlobalTensor<int64_t>&, uint32_t)>);
static_assert(
    std::is_same_v<decltype(&GlmDecodeEndCache::Read), int64_t (*)(const AscendC::LocalTensor<int64_t>&, uint32_t)>);

uint64_t Pattern(uint32_t expert, uint32_t epoch, uint32_t pattern) {
  if (pattern == 0) return 0;                     // All-peer/zero boundaries.
  if (pattern == 1) return (expert + epoch) / 3;  // Monotonic with empty experts.
  if (expert == 0) return uint64_t(std::numeric_limits<int64_t>::min());
  if (expert == 1) return uint64_t(std::numeric_limits<int64_t>::max());
  return uint64_t(0x8123456789abcdef) ^ (uint64_t(expert) << 33) ^ (uint64_t(epoch) * 0x9e3779b97f4a7c15);
}

void Check(uint32_t initialExperts, uint32_t pattern, bool replay) {
  using namespace AscendC;
  alignas(32) std::array<uint8_t, GUARD_BYTES + Cache::MASK_BYTES + GUARD_BYTES> mask;
  alignas(32) std::array<uint8_t, GUARD_BYTES + Cache::END_BYTES + GUARD_BYTES> source;
  mask.fill(0xa5);
  source.fill(0x5a);
  auto* cacheStart = mask.data() + GUARD_BYTES + Cache::END_OFFSET;
  LocalTensor<int64_t> cached{cacheStart, Cache::END_BYTES};
  for (uint32_t epoch = 0; epoch < (replay ? 3u : 1u); ++epoch) {
    const uint32_t experts =
        epoch == 1 ? (initialExperts == Cache::MAX_EXPERTS ? 1 : initialExperts + 1) : initialExperts;
    for (uint32_t expert = 0; expert < experts; ++expert) {
      const uint64_t bits = Pattern(expert, epoch, pattern);
      std::memcpy(source.data() + GUARD_BYTES + expert * sizeof(int64_t), &bits, sizeof(bits));
    }
    const auto previousMask = mask;
    const auto previousSource = source;
    GlobalTensor<int64_t> ends{source.data() + GUARD_BYTES, experts * uint32_t(sizeof(int64_t))};
    Model state;
    state.cacheBegin = cacheStart;
    state.liveBytes = experts * sizeof(int64_t);
    model = &state;
    const auto beforePrepare = allocations;
    GlmDecodeEndCache::Prepare(cached, ends, experts);
    Require(allocations == beforePrepare, "helper allocated memory");
    const uint32_t aligned = experts / Cache::ENDS_PER_DMA * Cache::ENDS_PER_DMA;
    Require(state.dmaCopies == uint32_t(aligned != 0) && state.dmaBytes == aligned * sizeof(int64_t),
            "wrong DMA prefix");
    Require(state.gmReads == experts - aligned && state.ubWrites == experts - aligned, "wrong scalar tail count");
    Require(state.barriers == 2 && state.pending.bytes == 0, "incomplete prologue fences");
    for (uint32_t i = 0; i < state.gmReads; ++i)
      Require(state.gmIndices[i] == aligned + i && state.ubWriteIndices[i] == aligned + i, "wrong scalar tail index");
    for (uint32_t byte = 0; byte < mask.size(); ++byte) {
      const uint32_t first = GUARD_BYTES + Cache::END_OFFSET;
      const uint8_t expected = byte >= first && byte < first + experts * sizeof(int64_t)
                                   ? source[GUARD_BYTES + byte - first]
                                   : previousMask[byte];
      Require(mask[byte] == expected, "cache copy or mask neighbor corrupted");
    }
    Require(source == previousSource, "metadata source changed");
    // Audited prepared W2/W3 reconstruction may rewrite the first 16 KiB;
    // this deliberately does not emulate the complete production kernels.
    for (uint32_t byte = 0; byte < 16384; ++byte) mask[GUARD_BYTES + byte] ^= 0x3c;
    const uint32_t gmBeforeRead = state.gmReads;
    const auto beforeReads = allocations;
    for (uint32_t tile = 0; tile < 32; ++tile)
      for (uint32_t expert = 0; expert < experts; ++expert) {
        const int64_t value = GlmDecodeEndCache::Read(cached, expert);
        uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        Require(bits == Pattern(expert, epoch, pattern), "stale or altered cache read");
      }
    Require(allocations == beforeReads, "reader allocated memory");
    Require(state.gmReads == gmBeforeRead && state.ubReads == experts * 32, "reader returned to GM");
    Require(source == previousSource, "reader changed source metadata");
    model = nullptr;
  }
}
}  // namespace

void* operator new(std::size_t bytes) {
  ++allocations;
  if (auto* value = std::malloc(bytes ? bytes : 1)) return value;
  throw std::bad_alloc();
}
void operator delete(void* value) noexcept { std::free(value); }
void operator delete(void* value, std::size_t) noexcept { std::free(value); }

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "sanitizer") {
      for (uint32_t experts : {1u, 2u, 3u, 4u, 5u, 71u, 72u, 73u, 287u, 288u})
        for (uint32_t pattern = 0; pattern < 3; ++pattern) Check(experts, pattern, true);
    } else {
      if (argc != 4) throw std::runtime_error("requires experts pattern replay");
      const uint32_t experts = std::stoul(argv[1]);
      if (experts < 1 || experts > Cache::MAX_EXPERTS) throw std::runtime_error("caller expert admission");
      Check(experts, std::stoul(argv[2]), std::stoul(argv[3]) != 0);
    }
    std::cout << "exact cache bits, DMA bounds/fences, neighbors, replay and no-allocation passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
