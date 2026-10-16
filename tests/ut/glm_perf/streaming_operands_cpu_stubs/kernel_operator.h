// SPDX-License-Identifier: Apache-2.0
// Operand-only CPU instruction/event model. This does not emulate hardware pipes.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#define __aicore__
namespace AscendC {
using event_t = uint32_t;
struct int4b_t {};
enum class HardEvent { MTE2_MTE1, MTE1_M, M_MTE1, MTE1_MTE2 };
struct LoadData2DParams {
  uint16_t startIndex = 0, repeatTimes = 0, srcStride = 0, dstGap = 0;
  bool ifTranspose = false;
};
inline void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
template <class T>
struct LocalTensor {
  uint8_t* data;
  size_t bytes;
  LocalTensor operator[](size_t index) const {
    const auto offset = index * sizeof(T);
    Require(offset <= bytes, "local offset overflow");
    return {data + offset, bytes - offset};
  }
  template <class U>
  LocalTensor<U> ReinterpretCast() const {
    return {data, bytes};
  }
};
template <class T>
struct GlobalTensor {
  uint8_t* data;
  size_t bytes;
  GlobalTensor operator[](size_t index) const {
    const auto offset = index * sizeof(T);
    Require(offset <= bytes, "global offset overflow");
    return {data + offset, bytes - offset};
  }
};
struct Copy {
  uint8_t* dst;
  const uint8_t* src;
  size_t bytes;
};
struct Model {
  std::array<std::array<bool, 2>, 4> flags{};
  std::array<bool, 2> busy{};
  std::array<uint8_t*, 2> a{}, b{};
  std::array<std::vector<Copy>, 2> l1Readers;
  std::vector<Copy> mte2, mte1;
  uint32_t activationLoads = 0, weightLoads = 0, stages = 0, waits = 0;
  void CheckEmpty() const {
    for (auto& kinds : flags)
      for (bool flag : kinds) Require(!flag, "unconsumed event");
    for (bool value : busy) Require(!value, "Cube still reads operands");
    Require(mte1.empty() && mte2.empty(), "pending copy");
  }
};
inline Model* model = nullptr;
inline bool Overlap(const uint8_t* a, size_t aBytes, const uint8_t* b, size_t bBytes) {
  const auto aStart = reinterpret_cast<uintptr_t>(a), bStart = reinterpret_cast<uintptr_t>(b);
  return aStart < bStart + bBytes && bStart < aStart + aBytes;
}
template <class T>
void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t elements) {
  const auto bytes = elements * sizeof(T);
  Require(bytes <= dst.bytes && bytes <= src.bytes, "activation copy overflow");
  for (auto& copy : model->mte1)
    Require(!Overlap(dst.data, bytes, copy.src, copy.bytes), "L1 overwrite before load completes");
  for (unsigned slot = 0; slot < 2; ++slot)
    if (model->flags[static_cast<unsigned>(HardEvent::MTE1_MTE2)][slot])
      for (auto& copy : model->l1Readers[slot])
        Require(!Overlap(dst.data, bytes, copy.src, copy.bytes), "L1 overwrite without MTE1->MTE2 backward fence");
  model->mte2.push_back({dst.data, src.data, bytes});
  ++model->stages;
}
inline void LoadData(LocalTensor<int4b_t> dst, LocalTensor<int4b_t> src, LoadData2DParams params) {
  Require(params.srcStride == 1 && !params.ifTranspose && params.startIndex == 0 && params.dstGap == 0,
          "operand interpretation changed");
  Require(params.repeatTimes == 2 || params.repeatTimes == 8, "unexpected fractal repeats");
  const size_t bytes = params.repeatTimes * 512;
  Require(bytes <= dst.bytes && bytes <= src.bytes, "operand load overflow");
  for (unsigned slot = 0; slot < 2; ++slot)
    if (model->busy[slot])
      Require(!Overlap(dst.data, bytes, model->a[slot], 2048) && !Overlap(dst.data, bytes, model->b[slot], 4096),
              "L0 overwrite before Cube releases operands");
  for (auto& copy : model->mte2)
    Require(!Overlap(src.data, bytes, copy.dst, copy.bytes), "read before activation copy completes");
  model->mte1.push_back({dst.data, src.data, bytes});
  if (params.repeatTimes == 2)
    ++model->activationLoads;
  else
    ++model->weightLoads;
}
template <HardEvent Kind>
void SetFlag(event_t slot) {
  Require(slot < 2, "event slot overflow");
  auto& flag = model->flags[static_cast<unsigned>(Kind)][slot];
  Require(!flag, "event set twice without wait");
  if constexpr (Kind == HardEvent::M_MTE1) Require(model->busy[slot], "release without Cube issue");
  if constexpr (Kind == HardEvent::MTE1_MTE2) model->l1Readers[slot] = model->mte1;
  flag = true;
}
template <HardEvent Kind>
void WaitFlag(event_t slot) {
  Require(slot < 2, "event slot overflow");
  auto& flag = model->flags[static_cast<unsigned>(Kind)][slot];
  Require(flag, "wait without matching event");
  flag = false;
  ++model->waits;
  if constexpr (Kind == HardEvent::M_MTE1) model->busy[slot] = false;
  auto finish = [](auto& copies) {
    for (auto& copy : copies) std::memcpy(copy.dst, copy.src, copy.bytes);
    copies.clear();
  };
  if constexpr (Kind == HardEvent::MTE2_MTE1) finish(model->mte2);
  if constexpr (Kind == HardEvent::MTE1_M || Kind == HardEvent::MTE1_MTE2) finish(model->mte1);
}
inline void CubeIssue(uint32_t slot) {
  WaitFlag<HardEvent::MTE1_M>(slot);
  Require(!model->busy[slot], "Cube issue without operand reuse wait");
  model->busy[slot] = true;
}
}  // namespace AscendC
