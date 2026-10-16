// SPDX-License-Identifier: Apache-2.0
// Byte-copy semantics only; this mock cannot validate device instruction pipes.
#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
#define __aicore__
namespace AscendC {
constexpr unsigned PIPE_ALL = 0, PIPE_V = 1;
struct CopyTrace {
  uint64_t localCalls = 0, localBytes = 0, globalCalls = 0, globalBytes = 0;
  uint64_t allBarriers = 0, vectorBarriers = 0;
};
inline CopyTrace trace;
inline std::vector<unsigned> globalWrites;
inline int8_t* globalBase = nullptr;
struct DataCopyParams {
  uint16_t blockCount, blockLen, srcStride, dstStride;
};
template <unsigned Pipe>
inline void PipeBarrier() {
  if constexpr (Pipe == PIPE_ALL)
    ++trace.allBarriers;
  else
    ++trace.vectorBarriers;
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
template <typename T>
struct GlobalTensor {
  T* data;
  GlobalTensor operator[](size_t offset) const { return {data + offset}; }
};
template <typename T>
void Duplicate(LocalTensor<T> dst, T value, size_t count) {
  assert(count <= dst.count);
  std::fill_n(dst.data, count, value);
}
template <typename T>
void DataCopy(LocalTensor<T> dst, LocalTensor<T> src, size_t count) {
  assert(count <= dst.count && count <= src.count);
  std::copy_n(src.data, count, dst.data);
  ++trace.localCalls;
  trace.localBytes += count * sizeof(T);
}
template <typename T>
void DataCopy(LocalTensor<T> dst, LocalTensor<T> src, const DataCopyParams& params) {
  constexpr size_t blockBytes = 32;
  assert(blockBytes % sizeof(T) == 0);
  assert(params.blockCount > 0 && params.blockLen > 0);
  const size_t blockElements = params.blockLen * blockBytes / sizeof(T);
  const size_t sourceStep = (params.blockLen + params.srcStride) * blockBytes / sizeof(T);
  const size_t destinationStep = (params.blockLen + params.dstStride) * blockBytes / sizeof(T);
  for (size_t block = 0; block < params.blockCount; ++block) {
    assert(block * sourceStep + blockElements <= src.count);
    assert(block * destinationStep + blockElements <= dst.count);
    std::copy_n(src.data + block * sourceStep, blockElements, dst.data + block * destinationStep);
  }
  ++trace.localCalls;
  trace.localBytes += params.blockCount * blockElements * sizeof(T);
}
template <typename T>
void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, size_t count) {
  assert(count <= src.count);
  const size_t offset = reinterpret_cast<int8_t*>(dst.data) - globalBase;
  assert(offset + count * sizeof(T) <= globalWrites.size());
  std::copy_n(src.data, count, dst.data);
  for (size_t i = 0; i < count * sizeof(T); ++i) ++globalWrites[offset + i];
  ++trace.globalCalls;
  trace.globalBytes += count * sizeof(T);
}
}  // namespace AscendC
