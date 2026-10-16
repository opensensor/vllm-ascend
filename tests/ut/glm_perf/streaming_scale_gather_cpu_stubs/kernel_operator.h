// SPDX-License-Identifier: Apache-2.0
// Dedicated bounds/bit/dependency model; no SDK or hardware timing simulation.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#define __aicore__
namespace AscendC {
constexpr unsigned PIPE_V = 1, PIPE_ALL = 2;
struct Trace {
  unsigned broadcasts = 0, gathers = 0, vectorFences = 0, allFences = 0;
  bool vectorPending = false;
};
inline Trace trace;
inline void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
struct Memory {
  std::vector<uint32_t> words;
  std::vector<std::pair<size_t, size_t>> protectedBytes;
  explicit Memory(size_t bytes, uint32_t fill) : words(bytes / sizeof(uint32_t), fill) {
    Require(bytes % sizeof(uint32_t) == 0, "unaligned CPU memory");
  }
  void CheckWrite(size_t offset, size_t bytes) const {
    Require(offset <= words.size() * 4 && bytes <= words.size() * 4 - offset, "write bounds");
    for (const auto& region : protectedBytes)
      Require(offset + bytes <= region.first || offset >= region.second, "write into protected retained memory");
  }
};
template <typename T>
struct LocalTensor {
  Memory* memory;
  size_t offset, count;
  uint32_t ReadBits(size_t index) const {
    static_assert(sizeof(T) == 4);
    Require(index < count && offset + (index + 1) * 4 <= memory->words.size() * 4, "read bounds");
    return memory->words[offset / 4 + index];
  }
  void WriteBits(size_t index, uint32_t bits) const {
    Require(index < count, "view write bounds");
    memory->CheckWrite(offset + index * 4, 4);
    memory->words[offset / 4 + index] = bits;
  }
};
template <typename T, typename U>
bool Overlap(const LocalTensor<T>& a, const LocalTensor<U>& b) {
  return a.memory == b.memory &&
         std::max(a.offset, b.offset) < std::min(a.offset + a.count * 4, b.offset + b.count * 4);
}
struct BrcbRepeatParams {
  unsigned dstBlkStride, dstRepStride;
};
template <unsigned Pipe>
void PipeBarrier() {
  static_assert(Pipe == PIPE_V || Pipe == PIPE_ALL);
  if constexpr (Pipe == PIPE_V)
    ++trace.vectorFences;
  else
    ++trace.allFences;
  trace.vectorPending = false;
}
inline void Brcb(const LocalTensor<uint32_t>& dst, const LocalTensor<uint32_t>& src, unsigned repeats,
                 BrcbRepeatParams p) {
  Require(!trace.vectorPending, "Brcb source dependency lacks fence");
  Require(p.dstBlkStride == 1 && p.dstRepStride == 8 && repeats == 4, "Brcb geometry");
  Require(src.count >= repeats * 8 && dst.count >= repeats * 64, "Brcb bounds");
  Require(!Overlap(dst, src), "Brcb source/destination alias");
  ++trace.broadcasts;
  for (unsigned row = 0; row < repeats * 8; ++row)
    for (unsigned lane = 0; lane < 8; ++lane) dst.WriteBits(row * 8 + lane, src.ReadBits(row));
  trace.vectorPending = true;
}
inline void Gather(const LocalTensor<float>& dst, const LocalTensor<float>& src, const LocalTensor<uint32_t>& indices,
                   uint32_t base, unsigned count) {
  Require(!trace.vectorPending, "Gather dependency lacks fence");
  Require(count == 256 && base == 0, "Gather geometry");
  Require(dst.count >= count && indices.count >= count, "Gather bounds");
  Require(!Overlap(dst, src) && !Overlap(dst, indices) && !Overlap(src, indices), "Gather alias");
  // Check every source before issuing any write, like a bounded instruction model.
  for (unsigned i = 0; i < count; ++i) {
    const uint32_t offset = indices.ReadBits(i);
    Require(offset % 4 == 0 && offset / 4 < src.count, "Gather source byte offset");
  }
  ++trace.gathers;
  for (unsigned i = 0; i < count; ++i) dst.WriteBits(i, src.ReadBits(indices.ReadBits(i) / 4));
  trace.vectorPending = true;
}
}  // namespace AscendC
