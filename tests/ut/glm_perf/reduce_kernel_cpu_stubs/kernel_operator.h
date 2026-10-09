// SPDX-License-Identifier: Apache-2.0
// CPU-only semantics for testing the actual reducer source, not NPU timing.
#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>
#define __aicore__
#define __gm__
#define __global__
using GM_ADDR = uint8_t*;
using half = _Float16;
namespace AscendC {
enum class TPosition { VECCALC };
enum class RoundMode { CAST_NONE };
constexpr unsigned PIPE_ALL = 0, PIPE_V = 1;
inline unsigned blockIndex = 0, blockCount = 8;
inline uint64_t scalarReads = 0, gatherCalls = 0;
inline size_t allocatedBytes = 0;
inline float* outputBegin = nullptr;
inline std::vector<unsigned> outputWrites;
inline unsigned GetBlockIdx() { return blockIndex; }
inline unsigned GetBlockNum() { return blockCount; }
inline void InitSocState() {}
template <unsigned Pipe>
inline void PipeBarrier() {}
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
  T GetValue(size_t offset) const {
    assert(offset < count);
    return data[offset];
  }
  void SetValue(size_t offset, T value) const {
    assert(offset < count);
    data[offset] = value;
  }
};
template <typename T>
struct GlobalTensor {
  T* data = nullptr;
  void SetGlobalBuffer(T* pointer) { data = pointer; }
  GlobalTensor operator[](size_t offset) const { return {data + offset}; }
  T GetValue(size_t offset) const {
    ++scalarReads;
    return data[offset];
  }
};
template <TPosition Position>
struct TBuf {
  std::vector<uint64_t> words;
  template <typename T>
  LocalTensor<T> Get() {
    return {reinterpret_cast<T*>(words.data()), words.size() * 8 / sizeof(T)};
  }
};
struct TPipe {
  template <TPosition P>
  void InitBuffer(TBuf<P>& buffer, size_t bytes) {
    allocatedBytes += (bytes + 7) / 8 * 8;
    buffer.words.resize((bytes + 7) / 8);
  }
};
template <typename T>
void Duplicate(LocalTensor<T> dst, T value, size_t count) {
  assert(count <= dst.count);
  std::fill_n(dst.data, count, value);
}
template <typename T>
void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, size_t count) {
  assert(count <= dst.count);
  std::copy_n(src.data, count, dst.data);
}
template <typename T>
void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, size_t count) {
  assert(count <= src.count);
  std::copy_n(src.data, count, dst.data);
  if constexpr (std::is_same_v<T, float>) {
    if (outputBegin) {
      size_t offset = dst.data - outputBegin;
      assert(offset + count <= outputWrites.size());
      for (size_t i = 0; i < count; ++i) ++outputWrites[offset + i];
    }
  }
}
template <typename T>
void DataCopy(LocalTensor<T> dst, LocalTensor<T> src, size_t count) {
  assert(count <= dst.count && count <= src.count);
  std::copy_n(src.data, count, dst.data);
}
template <typename T, typename U>
void Cast(LocalTensor<T> dst, LocalTensor<U> src, RoundMode, size_t count) {
  assert(count <= dst.count && count <= src.count);
  for (size_t i = 0; i < count; ++i) dst.data[i] = static_cast<T>(src.data[i]);
}
inline void Muls(LocalTensor<float> dst, LocalTensor<float> src, float factor, size_t count) {
  assert(count <= dst.count && count <= src.count);
  for (size_t i = 0; i < count; ++i) dst.data[i] = src.data[i] * factor;
}
inline void Add(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, size_t count) {
  assert(count <= dst.count && count <= a.count && count <= b.count);
  for (size_t i = 0; i < count; ++i) dst.data[i] = a.data[i] + b.data[i];
}
inline void Gather(LocalTensor<float> dst, LocalTensor<float> src, LocalTensor<uint32_t> offsets, uint32_t,
                   size_t count) {
  ++gatherCalls;
  assert(count <= dst.count && count <= offsets.count);
  for (size_t i = 0; i < count; ++i) {
    size_t index = offsets.data[i] / sizeof(float);
    assert(index < src.count);
    dst.data[i] = src.data[index];
  }
}
struct BrcbRepeatParams {
  unsigned dstBlkStride, dstRepStride;
};
inline void Brcb(LocalTensor<float> dst, LocalTensor<float> src, unsigned repeats, BrcbRepeatParams params) {
  assert(params.dstBlkStride == 1 && params.dstRepStride == 8);
  assert(repeats * 8 <= src.count && repeats * 64 <= dst.count);
  for (unsigned row = 0; row < repeats * 8; ++row) std::fill_n(dst.data + row * 8, 8, src.data[row]);
}
}  // namespace AscendC
