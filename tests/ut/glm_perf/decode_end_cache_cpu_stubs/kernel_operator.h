// SPDX-License-Identifier: Apache-2.0
// Bounded CPU model for decode-only metadata DMA and scalar access ordering.
#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#define __aicore__
namespace AscendC {
inline void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
struct Copy {
  uint8_t* destination = nullptr;
  const uint8_t* source = nullptr;
  uint32_t bytes = 0;
};
struct Model {
  Copy pending;
  uint32_t barriers = 0, dmaCopies = 0, dmaBytes = 0, gmReads = 0, ubReads = 0, ubWrites = 0;
  std::array<uint32_t, 288> gmIndices{}, ubWriteIndices{};
  uint8_t* cacheBegin = nullptr;
  uint32_t liveBytes = 0;
};
inline Model* model = nullptr;
template <class T>
struct LocalTensor {
  uint8_t* data;
  uint32_t bytes;
  LocalTensor operator[](uint32_t index) const {
    const uint64_t offset = uint64_t(index) * sizeof(T);
    Require(offset <= bytes, "local subview bounds");
    return {data + offset, static_cast<uint32_t>(bytes - offset)};
  }
  void SetValue(uint32_t index, T value) const {
    const uint64_t offset = uint64_t(index) * sizeof(T);
    Require(offset + sizeof(T) <= bytes, "local write bounds");
    Require(model->barriers == 1 && model->pending.bytes == 0, "tail write before DMA fence");
    Require(data + offset >= model->cacheBegin && data + offset + sizeof(T) <= model->cacheBegin + model->liveBytes,
            "tail write changes a cache neighbor");
    Require(model->ubWrites < model->ubWriteIndices.size(), "too many tail writes");
    model->ubWriteIndices[model->ubWrites++] = index;
    std::memcpy(data + offset, &value, sizeof(T));
  }
  T GetValue(uint32_t index) const {
    const uint64_t offset = uint64_t(index) * sizeof(T);
    Require(offset + sizeof(T) <= bytes, "local read bounds");
    Require(model->barriers == 2 && model->pending.bytes == 0, "UB read before final fence");
    Require(data + offset >= model->cacheBegin && data + offset + sizeof(T) <= model->cacheBegin + model->liveBytes,
            "read outside freshly prepared experts");
    ++model->ubReads;
    T value;
    std::memcpy(&value, data + offset, sizeof(T));
    return value;
  }
};
template <class T>
struct GlobalTensor {
  const uint8_t* data;
  uint32_t bytes;
  GlobalTensor operator[](uint32_t index) const {
    const uint64_t offset = uint64_t(index) * sizeof(T);
    Require(offset <= bytes, "global subview bounds");
    return {data + offset, static_cast<uint32_t>(bytes - offset)};
  }
  T GetValue(uint32_t index) const {
    const uint64_t offset = uint64_t(index) * sizeof(T);
    Require(offset + sizeof(T) <= bytes, "GM scalar read bounds");
    Require(model->barriers == 1 && model->pending.bytes == 0, "GM tail read before DMA fence");
    Require(model->gmReads < model->gmIndices.size(), "too many GM tail reads");
    model->gmIndices[model->gmReads++] = index;
    T value;
    std::memcpy(&value, data + offset, sizeof(T));
    return value;
  }
};
template <class T>
void DataCopy(const LocalTensor<T>& destination, const GlobalTensor<T>& source, uint32_t elements) {
  const uint64_t bytes = uint64_t(elements) * sizeof(T);
  Require(elements && bytes % 32 == 0, "DMA length alignment");
  Require(reinterpret_cast<uintptr_t>(destination.data) % 32 == 0 && reinterpret_cast<uintptr_t>(source.data) % 32 == 0,
          "DMA address alignment");
  Require(bytes <= destination.bytes && bytes <= source.bytes, "DMA overread or overwrite");
  Require(destination.data == model->cacheBegin && bytes <= model->liveBytes, "DMA changes a cache neighbor");
  Require(model->barriers == 0 && model->pending.bytes == 0, "DMA outside launch prologue");
  ++model->dmaCopies;
  model->dmaBytes += bytes;
  model->pending = {destination.data, source.data, static_cast<uint32_t>(bytes)};
}
constexpr int PIPE_ALL = 0;
template <int Pipe>
void PipeBarrier() {
  static_assert(Pipe == PIPE_ALL, "cache model expects the two complete prologue fences");
  Require(model->barriers < 2, "extra prologue fence");
  ++model->barriers;
  if (model->pending.bytes) {
    std::memcpy(model->pending.destination, model->pending.source, model->pending.bytes);
    model->pending = {};
  }
}
}  // namespace AscendC
