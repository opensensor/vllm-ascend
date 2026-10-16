// SPDX-License-Identifier: Apache-2.0
// Executes the production helper with independent bit/bounds/ownership semantics.
#include "glm_streaming_scale_gather.h"
#include <array>
#include <iostream>
#include <string>
using namespace AscendC;
using namespace GlmStreamingScaleGather;
constexpr uint32_t GUARD = 0x7faabcde;
constexpr unsigned REPLAYS = 6, PAIRS = 64;
constexpr std::array<uint32_t, 16> PATTERNS{0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x807fffff,
                                            0x00800000, 0x80800000, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000,
                                            0x7fc12345, 0x7fa12345, 0xffa12345, 0xffffffff};
void Check(bool value, const char* message) { Require(value, message); }
template <typename Action>
void Reject(Action action) {
  bool rejected = false;
  try {
    action();
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  Check(rejected, "invalid tensor operation admitted");
  trace.vectorPending = false;
}
void CheckMask(const Memory& mask, const std::vector<uint32_t>& saved) {
  for (unsigned i = 0; i < mask.words.size(); ++i) {
    const unsigned byte = i * 4;
    const bool expanded = byte >= EXPANDED_OFFSET && byte < EXPANDED_OFFSET + EXPANDED_BYTES;
    const bool broadcast = byte >= GlmStreamingContract::BROADCAST_OFFSET &&
                           byte < GlmStreamingContract::BROADCAST_OFFSET + ELEMENTS * sizeof(float);
    if (!expanded && !broadcast) Check(mask.words[i] == saved[i], "source/factor/route-cache/guard changed");
  }
}
void Run(unsigned firstCount, unsigned firstStride) {
  Check(firstCount >= 1 && firstCount <= 31 && (firstStride == 1 || firstStride == 4 || firstStride == 128),
        "arguments");
  Memory mask(RouteCache::MASK_BYTES + 128, GUARD), sx(2 * ROWS * 128 * 4 + 128, GUARD);
  mask.protectedBytes = {{0, GlmStreamingContract::BROADCAST_OFFSET},
                         {GlmStreamingContract::RETAINED_INDEX_OFFSET, EXPANDED_OFFSET},
                         {GROUP_SOURCE_OFFSET, RouteCache::MASK_BYTES + 128}};
  LocalTensor<uint32_t> expanded{&mask, EXPANDED_OFFSET, ELEMENTS};
  LocalTensor<float> dst{&mask, GlmStreamingContract::BROADCAST_OFFSET, ELEMENTS};
  const std::array<unsigned, REPLAYS> counts{firstCount, 31, 1, 17, 16, firstCount};
  const std::array<unsigned, REPLAYS> strides{firstStride, 1, 128, 4, firstStride, 1};
  trace = {};
  for (unsigned replay = 0; replay < REPLAYS; ++replay) {
    const unsigned count = counts[replay], stride = strides[replay];
    // Caller prepares source indices once after draining old batch readers.
    const unsigned sourceOffset = stride == 1 ? GROUP_SOURCE_OFFSET : GlmStreamingContract::RETAINED_INDEX_OFFSET;
    for (unsigned row = 0; row < ROWS; ++row) {
      const unsigned sourceRow = stride == 1 || row < count ? row : 0;
      mask.words[sourceOffset / 4 + row] = sourceRow * stride * 4;
    }
    const auto saved = mask.words;
    const LocalTensor<uint32_t> sourceIndices{&mask, sourceOffset, ROWS};
    Prepare(expanded, sourceIndices);
    Check(!trace.vectorPending, "Prepare omitted completion fence");
    for (unsigned row = 0; row < ROWS; ++row)
      for (unsigned lane = 0; lane < LANES; ++lane)
        Check(expanded.ReadBits(row * LANES + lane) == sourceIndices.ReadBits(row), "expanded index mismatch");
    const auto savedExpanded = std::vector<uint32_t>(mask.words.begin() + EXPANDED_OFFSET / 4,
                                                     mask.words.begin() + (EXPANDED_OFFSET + EXPANDED_BYTES) / 4);
    for (unsigned pair = 0; pair < PAIRS; ++pair)
      for (unsigned part = 0; part < 2; ++part) {
        // Separate pair/group sx bases; undefined row-major padding gets adversarial canaries.
        const unsigned sourceBase = stride == 1 ? part * ROWS : part;
        for (unsigned row = 0; row < ROWS; ++row)
          sx.words[row * stride + sourceBase] = PATTERNS[(row + pair + part * 3 + replay) % PATTERNS.size()];
        const auto savedSx = sx.words;
        const LocalTensor<float> source{&sx, sourceBase * 4, stride == 1 ? ROWS : ROWS * stride - part};
        Broadcast(dst, source, expanded);
        Check(!trace.vectorPending, "Broadcast omitted completion fence");
        for (unsigned row = 0; row < ROWS; ++row)
          for (unsigned lane = 0; lane < LANES; ++lane) {
            const unsigned sourceRow = stride == 1 || row < count ? row : 0;
            Check(dst.ReadBits(row * LANES + lane) == savedSx[sourceRow * stride + sourceBase],
                  "scale bits or padded rows changed");
          }
        Check(sx.words == savedSx, "Gather modified source scales");
        Check(std::equal(savedExpanded.begin(), savedExpanded.end(), mask.words.begin() + EXPANDED_OFFSET / 4),
              "Gather modified retained expanded indices");
        CheckMask(mask, saved);
      }
  }
  Check(trace.broadcasts == REPLAYS && trace.allFences == REPLAYS, "indices prepared inside K/group loop");
  Check(trace.gathers == REPLAYS * PAIRS * 2, "wrong Gather issue count");
  Check(trace.vectorFences == trace.broadcasts + trace.gathers, "missing or surplus completion fences");
  std::cout << "exact bits, retained route cache and changed batches passed: " << trace.gathers << " broadcasts\n";
}
void Faults() {
  Memory mask(RouteCache::MASK_BYTES + 128, GUARD), src(ROWS * 128 * 4, GUARD);
  mask.protectedBytes = {{RouteCache::END_OFFSET, EXPANDED_OFFSET}};
  LocalTensor<uint32_t> expanded{&mask, EXPANDED_OFFSET, ELEMENTS}, indices{&mask, GROUP_SOURCE_OFFSET, ROWS};
  LocalTensor<float> dst{&mask, GlmStreamingContract::BROADCAST_OFFSET, ELEMENTS}, source{&src, 0, ROWS * 128};
  for (unsigned row = 0; row < ROWS; ++row) mask.words[GROUP_SOURCE_OFFSET / 4 + row] = row * 4;
  Reject([&] { Prepare({&mask, RouteCache::END_OFFSET, ELEMENTS}, indices); });
  Reject([&] { Prepare({&mask, EXPANDED_OFFSET, ELEMENTS - 1}, indices); });
  Reject([&] { Prepare(expanded, {&mask, GROUP_SOURCE_OFFSET, ROWS - 1}); });
  Reject([&] { Prepare({&mask, GROUP_SOURCE_OFFSET, ELEMENTS}, indices); });
  Prepare(expanded, indices);
  Reject([&] { Broadcast({&mask, GlmStreamingContract::BROADCAST_OFFSET, ELEMENTS - 1}, source, expanded); });
  Reject([&] { Broadcast(dst, source, {&mask, EXPANDED_OFFSET, ELEMENTS - 1}); });
  mask.words[EXPANDED_OFFSET / 4] = 3;
  Reject([&] { Broadcast(dst, source, expanded); });
  mask.words[EXPANDED_OFFSET / 4] = source.count * 4;
  Reject([&] { Broadcast(dst, source, expanded); });
  Prepare(expanded, indices);
  Reject([&] { Broadcast({&mask, EXPANDED_OFFSET, ELEMENTS}, source, expanded); });
  // Raw mock instructions expose missing barriers; the real helper adds them.
  Brcb(expanded, indices, 4, {1, 8});
  Reject([&] { Gather(dst, source, expanded, 0, ELEMENTS); });
  std::cout << "10 bounds, alias, retained-cache and dependency faults rejected\n";
}
int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "faults")
      Faults();
    else {
      Check(argc == 3, "usage: rows stride or faults");
      Run(std::stoul(argv[1]), std::stoul(argv[2]));
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
