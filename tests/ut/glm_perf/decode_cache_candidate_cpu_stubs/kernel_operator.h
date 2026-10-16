// SPDX-License-Identifier: Apache-2.0
// CPU-only metadata/prologue/dispatch model, not an arithmetic kernel emulator.
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#define __aicore__
#define __gm__
using GM_ADDR = void*;
using half = uint16_t;
namespace AscendC {
struct Context {
  uint8_t* ends = nullptr;
  uint32_t experts = 0, routes = 0, block = 0;
  bool caching = false;
  uint32_t cacheFences = 0, gmReads = 0, ubReads = 0, dmas = 0;
  uint8_t* pendingDestination = nullptr;
  const uint8_t* pendingSource = nullptr;
  uint32_t pendingBytes = 0;
  std::vector<int64_t> trace;
};
inline Context* context = nullptr;
inline void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
template <class T>
struct LocalTensor {
  uint8_t* data;
  uint32_t bytes;
  LocalTensor operator[](uint64_t index) const {
    const uint64_t offset = index * sizeof(T);
    Require(offset <= bytes, "local offset bounds");
    return {data + offset, static_cast<uint32_t>(bytes - offset)};
  }
  void SetValue(uint32_t index, T value) const {
    Require((index + uint64_t(1)) * sizeof(T) <= bytes, "local write bounds");
    Require(context->cacheFences == 1, "tail writer before prefix fence");
    std::memcpy(data + index * sizeof(T), &value, sizeof(T));
  }
  T GetValue(uint32_t index) const {
    Require((index + uint64_t(1)) * sizeof(T) <= bytes, "local read bounds");
    Require(context->cacheFences == 2, "cache reader before final fence");
    ++context->ubReads;
    T value;
    std::memcpy(&value, data + index * sizeof(T), sizeof(T));
    return value;
  }
};
template <class T>
struct GlobalTensor {
  uint8_t* data = nullptr;
  void SetGlobalBuffer(T* value) { data = reinterpret_cast<uint8_t*>(value); }
  GlobalTensor operator[](uint64_t index) const { return {data ? data + index * sizeof(T) : nullptr}; }
  T GetValue(uint32_t index) const {
    if (data == context->ends) {
      Require(index < context->experts, "GM end read bounds");
      ++context->gmReads;
    } else
      Require(index < context->routes, "GM order read bounds");
    T value;
    std::memcpy(&value, data + index * sizeof(T), sizeof(T));
    return value;
  }
};
template <class T>
void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t count) {
  Require(src.data == context->ends, "unexpected metadata DMA");
  Require(count <= context->experts && count * sizeof(T) <= dst.bytes, "end DMA bounds");
  Require(count && count * sizeof(T) % 32 == 0 && reinterpret_cast<uintptr_t>(dst.data) % 32 == 0, "end DMA alignment");
  Require(!context->pendingBytes && context->cacheFences == 0, "duplicate cache DMA");
  context->pendingDestination = dst.data;
  context->pendingSource = src.data;
  context->pendingBytes = count * sizeof(T);
  ++context->dmas;
}
template <class T>
void DataCopy(const GlobalTensor<T>&, const LocalTensor<T>&, uint32_t) {}
template <class T, class U>
void Duplicate(const LocalTensor<T>&, U, uint32_t) {}
template <class T>
void Gather(const LocalTensor<T>&, const LocalTensor<T>&, const LocalTensor<uint32_t>&, uint32_t, uint32_t) {}
constexpr int PIPE_ALL = 0;
template <int>
void PipeBarrier() {
  if (context->caching && context->cacheFences < 2) {
    ++context->cacheFences;
    if (context->pendingBytes) {
      std::memcpy(context->pendingDestination, context->pendingSource, context->pendingBytes);
      context->pendingBytes = 0;
    }
  }
}
inline uint32_t GetBlockIdx() { return context->block; }
inline uint32_t GetBlockNum() { return 8; }
}  // namespace AscendC
