// SPDX-License-Identifier: Apache-2.0
// Concrete grouped operand producer; no kernel launch ABI or device qualification.
#ifndef QWEN_STREAMING_NEXT_OPERANDS_H
#define QWEN_STREAMING_NEXT_OPERANDS_H
#include "kernel_operator.h"
#include "qwen_streaming_next_contract.h"

namespace qwen_streaming_next {
using namespace AscendC;

__aicore__ constexpr uint32_t ActivationByte(uint32_t limb, uint32_t kb, uint32_t row, uint32_t byte) {
  return limb * (M * GROUP / 2) + (kb * M + row) * (K0 / 2) + byte;
}
__aicore__ constexpr uint32_t ActivationL0Byte(uint32_t limb, uint32_t kb, uint32_t row, uint32_t byte) {
  // [limb, M/16, G128/K64, 16, K64/2].
  return limb * (M * GROUP / 2) + ((row / BLOCK) * (GROUP / K0) + kb) * BLOCK * K0 / 2 + (row % BLOCK) * K0 / 2 + byte;
}
__aicore__ constexpr uint32_t WeightL1Byte(uint32_t nb, uint32_t group, uint32_t kb, uint32_t column, uint32_t byte,
                                           uint32_t groups) {
  // Resident checkpoint tile: [N/16, groups, G128/K64, 16, K64/2].
  return (((nb * groups + group) * (GROUP / K0) + kb) * BLOCK + column) * (K0 / 2) + byte;
}
__aicore__ constexpr uint32_t WeightL0Byte(uint32_t nb, uint32_t kb, uint32_t column, uint32_t byte) {
  // L0B group: [G128/K64, N/16, 16, K64/2].
  return ((kb * (N / BLOCK) + nb) * BLOCK + column) * (K0 / 2) + byte;
}

class OperandProducer {
 public:
  // Tensor arguments already denote arena subregions at contract *_OFFSET.
  // The owner keeps residentWeights alive for all row tiles of one expert/N tile.
  __aicore__ inline void Init(GlobalTensor<int8_t> low, GlobalTensor<int8_t> high, GlobalTensor<float> scale,
                              GlobalTensor<float> sums, LocalTensor<int8_t> packedUB, LocalTensor<float> metadataUB,
                              LocalTensor<int8_t> activationL1, LocalTensor<int8_t> activationL0,
                              LocalTensor<int8_t> weightL0, LocalTensor<int8_t> residentWeights, uint32_t k) {
    low_ = low;
    high_ = high;
    scale_ = scale;
    sums_ = sums;
    packed_ = packedUB;
    metadata_ = metadataUB;
    a1_ = activationL1;
    a2_ = activationL0;
    b2_ = weightL0;
    b1_ = residentWeights;
    k_ = k;
    groups_ = k / GROUP;
  }

  // Caller acquires this slot after preceding M_MTE1 and V_MTE2 completion,
  // and waits for MTE1_M before issuing MMAD. No live slot may be overwritten.
  __aicore__ inline void Produce(uint32_t slot, uint32_t metadataSlot, int64_t row, uint32_t group, uint32_t liveRows) {
    const uint32_t event = EventId(slot);
    // Caller has completed readback of the prior Cube using this physical
    // slot. Do not wait on the current Cube, which owns the other slot.
    SetFlag<HardEvent::V_MTE2>(event);
    WaitFlag<HardEvent::V_MTE2>(event);
    auto packed = packed_[slot * PACKED_ACTIVATION_SLOT_BYTES];
    auto metadata = metadata_[metadataSlot * ACTIVATION_METADATA_SLOT_BYTES / sizeof(float)];
    auto scale = metadata;
    auto sums = metadata[M * LANES];
    // Pad both limbs and metadata. Do not inherit stale bytes from a larger tile.
    Duplicate(packed.template ReinterpretCast<int16_t>(), static_cast<int16_t>(0), PACKED_ACTIVATION_SLOT_BYTES / 2);
    Duplicate(metadata, 0.0f, ACTIVATION_METADATA_SLOT_BYTES / sizeof(float));
    SetFlag<HardEvent::V_MTE2>(event);
    WaitFlag<HardEvent::V_MTE2>(event);
    DataCopyParams rows{static_cast<uint16_t>(liveRows), 1, static_cast<uint16_t>(k_ / K0 - 1), 0};
    if (liveRows > 0)
      for (uint32_t kb = 0; kb < GROUP / K0; ++kb) {
        const int64_t source = row * k_ / 2 + group * GROUP / 2 + kb * K0 / 2;
        DataCopy(packed[ActivationByte(0, kb, 0, 0)], low_[source], rows);
        DataCopy(packed[ActivationByte(1, kb, 0, 0)], high_[source], rows);
      }
    DataCopyParams meta{static_cast<uint16_t>(liveRows), 1, static_cast<uint16_t>(groups_ - 1), 0};
    const int64_t sourceMeta = (row * groups_ + group) * LANES;
    if (liveRows > 0) {
      DataCopy(scale, scale_[sourceMeta], meta);
      DataCopy(sums, sums_[sourceMeta], meta);
    }
    SetFlag<HardEvent::MTE2_V>(event);
    WaitFlag<HardEvent::MTE2_V>(event);
    // Metadata becomes vector-visible above; packed bytes travel through MTE3.
    SetFlag<HardEvent::V_MTE3>(event);
    WaitFlag<HardEvent::V_MTE3>(event);
    DataCopy(a1_[slot * ACTIVATION_L1_SLOT_BYTES], packed, PACKED_ACTIVATION_SLOT_BYTES);
    SetFlag<HardEvent::MTE3_MTE1>(event);
    WaitFlag<HardEvent::MTE3_MTE1>(event);
    SetFlag<HardEvent::MTE3_MTE2>(event);
    WaitFlag<HardEvent::MTE3_MTE2>(event);
    LoadData2DParams load;
    load.repeatTimes = GROUP / K0;
    load.srcStride = M / BLOCK;
    load.ifTranspose = false;
    for (uint32_t limb = 0; limb < 2; ++limb) {
      for (uint32_t mb = 0; mb < M / BLOCK; ++mb) {
        LoadData(a2_[slot * ACTIVATION_L0_SLOT_BYTES + limb * M * GROUP / 2 + mb * BLOCK * GROUP / 2]
                     .template ReinterpretCast<int4b_t>(),
                 a1_[slot * ACTIVATION_L1_SLOT_BYTES + limb * M * GROUP / 2 + mb * BLOCK * K0 / 2]
                     .template ReinterpretCast<int4b_t>(),
                 load);
      }
    }
    SetFlag<HardEvent::MTE1_MTE3>(event);
    WaitFlag<HardEvent::MTE1_MTE3>(event);
    load.repeatTimes = N / BLOCK;
    load.srcStride = k_ / K0;
    for (uint32_t kb = 0; kb < GROUP / K0; ++kb) {
      LoadData(b2_[slot * WEIGHT_L0_SLOT_BYTES + kb * N * K0 / 2].template ReinterpretCast<int4b_t>(),
               b1_[(group * GROUP + kb * K0) * BLOCK / 2].template ReinterpretCast<int4b_t>(), load);
    }
    SetFlag<HardEvent::MTE1_M>(event);
    WaitFlag<HardEvent::MTE1_M>(event);
  }

 private:
  GlobalTensor<int8_t> low_, high_;
  GlobalTensor<float> scale_, sums_;
  LocalTensor<int8_t> packed_, a1_, a2_, b1_, b2_;
  LocalTensor<float> metadata_;
  uint32_t k_ = 0, groups_ = 0;
};
static_assert(M % BLOCK == 0 && GROUP == 2 * K0 && SLOTS == 2, "unsupported paired operand layout");
static_assert(PACKED_ACTIVATION_SLOT_BYTES == 2 * M * GROUP / 2, "packed ABI mismatch");
static_assert(WEIGHT_L0_SLOT_BYTES == N * GROUP / 2, "weight ABI mismatch");
}  // namespace qwen_streaming_next
#endif
