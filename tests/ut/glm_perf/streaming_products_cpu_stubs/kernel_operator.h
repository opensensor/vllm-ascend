// SPDX-License-Identifier: Apache-2.0
// Bounds, instruction semantics and event accounting; no device timing model.
#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#define __aicore__
namespace AscendC {
template <typename T>
struct GlobalTensor;
constexpr unsigned PIPE_V = 1, PIPE_ALL = 2, EVENT_ID0 = 0, EVENT_ID1 = 1;
enum class HardEvent { M_V, V_M };
enum class RoundMode { CAST_NONE };
enum class BlockMode { BLOCK_MODE_MATRIX };
inline bool events[2][2]{};
inline std::unordered_map<int32_t*, bool> rawReaders;
inline int32_t* activeRaw = nullptr;
inline unsigned activeAdds = 0;
inline unsigned casts = 0, copies = 0, gathers = 0, barriers = 0, multiplies = 0, adds = 0;
inline unsigned castElements = 0;
inline unsigned indexPreparations = 0, floatBroadcasts = 0;
inline std::vector<unsigned> gatherWidths;
inline unsigned maskSetups = 0, maskNormalizations = 0, vectorMask = 64;
inline bool normalMaskMode = true;
inline void SetMaskNorm() {
  normalMaskMode = true;
  ++maskNormalizations;
}
template <typename T>
void SetVectorMask(int32_t mask) {
  static_assert(sizeof(T) == 4);
  assert(normalMaskMode && mask > 0 && mask <= 64);
  vectorMask = mask;
  ++maskSetups;
}
struct CastRange {
  int32_t* source;
  float* destination;
  unsigned mask, repeats, dstRepStride, srcRepStride;
};
inline std::vector<CastRange> castRanges;
inline void ResetCounters() {
  casts = copies = gathers = barriers = multiplies = adds = castElements = 0;
  maskSetups = maskNormalizations = 0;
  floatBroadcasts = 0;
  gatherWidths.clear();
  castRanges.clear();
}
template <unsigned Pipe>
void PipeBarrier() {
  static_assert(Pipe == PIPE_V || Pipe == PIPE_ALL);
  ++barriers;
  if (activeRaw && activeAdds == 2) {
    rawReaders[activeRaw] = false;
    activeRaw = nullptr;
    activeAdds = 0;
  }
}
template <HardEvent Kind>
void SetFlag(unsigned slot) {
  if (slot > 1 || events[static_cast<unsigned>(Kind)][slot]) throw std::logic_error("event already issued");
  events[static_cast<unsigned>(Kind)][slot] = true;
}
template <HardEvent Kind>
void WaitFlag(unsigned slot) {
  if (slot > 1 || !events[static_cast<unsigned>(Kind)][slot]) throw std::logic_error("unmatched wait");
  events[static_cast<unsigned>(Kind)][slot] = false;
}
template <typename T>
struct LocalTensor {
  T* data;
  size_t count;
  LocalTensor operator[](size_t offset) const {
    assert(offset <= count);
    return {data + offset, count - offset};
  }
  template <typename U>
  LocalTensor<U> ReinterpretCast() const {
    return {reinterpret_cast<U*>(data), count * sizeof(T) / sizeof(U)};
  }
};
struct DataCopyParams {
  uint16_t blockCount, blockLen, srcStride, dstStride;
};
struct DataCopyEnhancedParams {
  BlockMode blockMode;
};
struct UnaryRepeatParams {
  unsigned dstBlkStride, srcBlkStride, dstRepStride, srcRepStride;
};
struct BinaryRepeatParams {
  unsigned dstBlkStride, src0BlkStride, src1BlkStride, dstRepStride, src0RepStride, src1RepStride;
};
struct BrcbRepeatParams {
  unsigned dstBlkStride, dstRepStride;
};
template <typename T>
void DataCopy(LocalTensor<T> dst, LocalTensor<T> src, DataCopyParams p) {
  ++copies;
  const size_t elements = p.blockLen * 32 / sizeof(T);
  const size_t sstep = elements + p.srcStride * 32 / sizeof(T), dstep = elements + p.dstStride * 32 / sizeof(T);
  assert((p.blockCount - 1) * sstep + elements <= src.count);
  assert((p.blockCount - 1) * dstep + elements <= dst.count);
  for (unsigned block = 0; block < p.blockCount; ++block)
    std::copy_n(src.data + block * sstep, elements, dst.data + block * dstep);
}
inline void DataCopy(LocalTensor<int32_t> dst, LocalTensor<int32_t> src, DataCopyParams p, DataCopyEnhancedParams e) {
  assert(e.blockMode == BlockMode::BLOCK_MODE_MATRIX && p.srcStride == 0 && p.dstStride == 0);
  if (rawReaders[dst.data]) throw std::logic_error("raw slot overwritten before consumer fence");
  rawReaders[dst.data] = true;
  ++copies;
  const size_t elements = p.blockCount * p.blockLen * 16 * 16;
  assert(elements <= dst.count && elements <= src.count);
  std::copy_n(src.data, elements, dst.data);
}
template <typename Destination, typename Source, bool SetMask = true>
void Cast(LocalTensor<Destination> dst, LocalTensor<Source> src, RoundMode, uint64_t mask, unsigned repeats,
          UnaryRepeatParams p) {
  static_assert(sizeof(Destination) == 4 && sizeof(Source) == 4);
  if constexpr (SetMask) {
    SetMaskNorm();
    SetVectorMask<Destination>(mask);
  }
  if (!normalMaskMode || vectorMask != mask) throw std::logic_error("cast mask was not established");
  ++casts;
  castElements += mask * repeats;
  castRanges.push_back({src.data, dst.data, static_cast<unsigned>(mask), repeats, p.dstRepStride, p.srcRepStride});
  const auto reader = rawReaders.find(src.data);
  if (reader != rawReaders.end()) {
    if (!reader->second || activeRaw) throw std::logic_error("invalid raw consumer lifecycle");
    activeRaw = src.data;
    activeAdds = 0;
  }
  for (unsigned row = 0; row < repeats; ++row)
    for (unsigned lane = 0; lane < mask; ++lane) {
      const unsigned destination = row * p.dstRepStride * 8 + lane / 8 * p.dstBlkStride * 8 + lane % 8;
      const unsigned source = row * p.srcRepStride * 8 + lane / 8 * p.srcBlkStride * 8 + lane % 8;
      assert(destination < dst.count && source < src.count);
      dst.data[destination] = static_cast<float>(src.data[source]);
    }
}
inline void Gather(LocalTensor<float> dst, LocalTensor<float> src, LocalTensor<uint32_t> indices, uint32_t,
                   unsigned n) {
  ++gathers;
  gatherWidths.push_back(n);
  vectorMask = 64;
  assert(n <= dst.count && n <= indices.count);
  for (unsigned i = 0; i < n; ++i) {
    assert(indices.data[i] % 4 == 0 && indices.data[i] / 4 < src.count);
    std::memcpy(dst.data + i, src.data + indices.data[i] / 4, sizeof(float));
  }
}
inline void Brcb(LocalTensor<float> dst, LocalTensor<float> src, unsigned repeats, BrcbRepeatParams p) {
  ++floatBroadcasts;
  vectorMask = 64;
  assert(p.dstBlkStride == 1 && p.dstRepStride == 8);
  assert(repeats * 8 <= src.count && repeats * 64 <= dst.count);
  for (unsigned i = 0; i < repeats * 8; ++i)
    for (unsigned lane = 0; lane < 8; ++lane) std::memcpy(dst.data + i * 8 + lane, src.data + i, sizeof(float));
}
inline void Brcb(LocalTensor<uint32_t> dst, LocalTensor<uint32_t> src, unsigned repeats, BrcbRepeatParams p) {
  ++indexPreparations;
  vectorMask = 64;
  assert(p.dstBlkStride == 1 && p.dstRepStride == 8);
  assert(repeats * 8 <= src.count && repeats * 64 <= dst.count);
  for (unsigned i = 0; i < repeats * 8; ++i) std::fill_n(dst.data + i * 8, 8, src.data[i]);
}
inline void Mul(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, uint64_t mask, unsigned repeats,
                BinaryRepeatParams p) {
  ++multiplies;
  for (unsigned r = 0; r < repeats; ++r)
    for (unsigned lane = 0; lane < mask; ++lane) {
      const unsigned di = r * p.dstRepStride * 8 + lane / 8 * p.dstBlkStride * 8 + lane % 8;
      const unsigned ai = r * p.src0RepStride * 8 + lane / 8 * p.src0BlkStride * 8 + lane % 8;
      const unsigned bi = r * p.src1RepStride * 8 + lane / 8 * p.src1BlkStride * 8 + lane % 8;
      assert(di < dst.count && ai < a.count && bi < b.count);
      volatile float rounded = a.data[ai] * b.data[bi];
      dst.data[di] = rounded;
    }
}
inline void Mul(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, unsigned n) {
  ++multiplies;
  assert(n <= dst.count && n <= a.count && n <= b.count);
  for (unsigned i = 0; i < n; ++i) {
    volatile float rounded = a.data[i] * b.data[i];
    dst.data[i] = rounded;
  }
}
inline void Add(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, unsigned n) {
  ++adds;
  if (activeRaw) ++activeAdds;
  assert(n <= dst.count && n <= a.count && n <= b.count);
  for (unsigned i = 0; i < n; ++i) {
    volatile float rounded = a.data[i] + b.data[i];
    dst.data[i] = rounded;
  }
}
}  // namespace AscendC
