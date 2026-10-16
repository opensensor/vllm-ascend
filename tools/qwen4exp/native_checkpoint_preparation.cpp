// SPDX-License-Identifier: Apache-2.0
// Fused device pre-copy decision and copy. No host acceptance observation.
#include "kernel_operator.h"
#define CHECKPOINT_INLINE __aicore__ inline
#include "checkpoint_preparation_integer.h"

using namespace AscendC;
namespace {
constexpr int64_t PADDED_ROWS = 8;
constexpr uint32_t TILE_WORDS = 4096;
constexpr int64_t ALIGN_WORDS = 8;
}  // namespace

__aicore__ inline void PrecopyState(GM_ADDR previous, GM_ADDR computed, GM_ADDR accepted, const __gm__ int64_t* cfg,
                                    GM_ADDR output) {
  const int64_t rows = cfg[2], columns = cfg[3], blocks = cfg[4];
  const int64_t words = cfg[5], conv_width = cfg[6], block_size = cfg[7];
  // Immutable geometry is checked by the owner before any submission.
  if (rows < 1 || rows > 6 || columns < 1 || blocks < 1 || words < ALIGN_WORDS || words % ALIGN_WORDS ||
      conv_width < 0 || conv_width % ALIGN_WORDS || block_size < 1)
    return;
  GlobalTensor<int32_t> sources, positions, counts, table, result;
  sources.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(previous));
  positions.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(computed));
  counts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(accepted));
  table.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cfg[1]));
  result.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output));
  GlobalTensor<uint32_t> bank;
  bank.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(cfg[0]));
  TPipe pipe;
  TBuf<TPosition::VECCALC> storage, receipt_storage;
  pipe.InitBuffer(storage, TILE_WORDS * sizeof(uint32_t));
  pipe.InitBuffer(receipt_storage, 2 * PADDED_ROWS * sizeof(int32_t));
  auto tile = storage.Get<uint32_t>();
  auto receipt = receipt_storage.Get<int32_t>();
  for (uint32_t i = 0; i < PADDED_ROWS; ++i) {
    receipt.SetValue(i, 1);
    receipt.SetValue(PADDED_ROWS + i, 0);
  }
  for (int64_t row = 0; row < rows; ++row) {
    const int32_t source = sources.GetValue(row);
    const auto plan = checkpoint_preparation::Plan(source, positions.GetValue(row), counts.GetValue(row), block_size);
    receipt.SetValue(row, plan.normalized);
    receipt.SetValue(PADDED_ROWS + row, plan.error);
    if (!plan.copy) continue;
    const int64_t source_index = static_cast<int64_t>(source) + (conv_width == 0 ? plan.bias : 0);
    if (source_index < 0 || source_index >= columns || plan.destination < 0 || plan.destination >= columns) {
      receipt.SetValue(PADDED_ROWS + row, 2);
      continue;
    }
    const int64_t src = table.GetValue(row * columns + source_index);
    const int64_t dst = table.GetValue(row * columns + plan.destination);
    const int64_t offset = conv_width * plan.bias;
    if (src < 0 || src >= blocks || dst < 0 || dst >= blocks || offset >= words) {
      receipt.SetValue(PADDED_ROWS + row, 3);
      continue;
    }
    // Preserve request order and memmove semantics for overlapping SD suffixes.
    const int64_t count = words - offset;
    for (int64_t start = 0; start < count; start += TILE_WORDS) {
      const uint32_t n = static_cast<uint32_t>(count - start < TILE_WORDS ? count - start : TILE_WORDS);
      DataCopy(tile, bank[src * words + offset + start], n);
      SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
      DataCopy(bank[dst * words + start], tile, n);
      SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
  }
  SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
  DataCopy(result, receipt, 2 * PADDED_ROWS);
  SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
  WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}

extern "C" __global__ __aicore__ void qwen_checkpoint_preparation_v2(GM_ADDR previous, GM_ADDR computed,
                                                                     GM_ADDR accepted, GM_ADDR descriptors,
                                                                     GM_ADDR output, GM_ADDR config) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  InitSocState();
  const int64_t states = reinterpret_cast<__gm__ int64_t*>(config)[0];
  if (states < 1 || states > 256) return;
  auto records = reinterpret_cast<__gm__ int64_t*>(descriptors);
  for (int64_t index = GetBlockIdx(); index < states; index += GetBlockNum()) {
    PrecopyState(previous, computed, accepted, records + index * 8, output + index * 2 * PADDED_ROWS * sizeof(int32_t));
  }
}
