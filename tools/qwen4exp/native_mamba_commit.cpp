// SPDX-License-Identifier: Apache-2.0
// Private-tensor prototype. One AICore computes decisions and copies one state
// bank in request order. No polling, peer reads, or host-observed decisions.
#include "kernel_operator.h"
#define MAMBA_COMMIT_INLINE __aicore__ inline
#include "mamba_commit_integer.h"

using namespace AscendC;
namespace {
constexpr int64_t PADDED_ROWS = 8;
constexpr uint32_t TILE_WORDS = 4096;
constexpr int64_t ALIGN_WORDS = 8;
}  // namespace

__aicore__ inline void CommitState(GM_ADDR state, GM_ADDR inputs, GM_ADDR block_table, GM_ADDR output,
                                   const __gm__ int64_t* cfg) {
  const int64_t rows = cfg[0], columns = cfg[1], blocks = cfg[2];
  const int64_t words = cfg[3], conv_width = cfg[4], block_size = cfg[5];
  // The wrapper admits immutable geometry before capture. Values in input and
  // block table remain dynamic and are checked before any state address use.
  if (rows < 1 || rows > PADDED_ROWS || columns < 1 || blocks < 1 || words < ALIGN_WORDS || words % ALIGN_WORDS ||
      conv_width < 0 || conv_width % ALIGN_WORDS || block_size < 1)
    return;
  GlobalTensor<int32_t> metadata, table, result;
  GlobalTensor<uint32_t> bank;
  metadata.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(inputs));
  table.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(block_table));
  bank.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(state));
  result.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output));
  TPipe pipe;
  TBuf<TPosition::VECCALC> storage, receipt_storage;
  pipe.InitBuffer(storage, TILE_WORDS * sizeof(uint32_t));
  pipe.InitBuffer(receipt_storage, 2 * PADDED_ROWS * sizeof(int32_t));
  auto tile = storage.Get<uint32_t>();
  auto receipt = receipt_storage.Get<int32_t>();
  for (uint32_t i = 0; i < 2 * PADDED_ROWS; ++i) receipt.SetValue(i, 0);
  for (int64_t row = 0; row < rows; ++row) {
    const int32_t source = metadata.GetValue(row);
    const int32_t computed = metadata.GetValue(PADDED_ROWS + row);
    const int32_t scheduled = metadata.GetValue(2 * PADDED_ROWS + row);
    const int32_t drafts = metadata.GetValue(3 * PADDED_ROWS + row);
    const int32_t accepted = metadata.GetValue(4 * PADDED_ROWS + row);
    receipt.SetValue(row, accepted);
    if (source < 0 || computed < 0 || drafts < 0 || scheduled <= drafts || accepted < 0 ||
        accepted > static_cast<int64_t>(drafts) + 1) {
      receipt.SetValue(PADDED_ROWS + row, 1);
      continue;
    }
    const auto plan = mamba_commit::Plan(source, computed, scheduled, drafts, accepted, block_size);
    receipt.SetValue(row, plan.next_count);
    if (!plan.copy) continue;
    const int64_t source_index = source + (conv_width == 0 ? plan.bias : 0);
    if (source_index < 0 || source_index >= columns || plan.destination < 0 || plan.destination >= columns ||
        plan.bias < 0 || plan.bias > drafts) {
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
    const int64_t count = words - offset;
    // SD convolution moves a suffix toward the beginning of the same block;
    // forward tiles preserve clone/copy semantics even when ranges overlap.
    // Different blocks are disjoint. Serial rows preserve inter-request order.
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

extern "C" __global__ __aicore__ void qwen_mamba_commit_v1(GM_ADDR state, GM_ADDR inputs, GM_ADDR block_table,
                                                           GM_ADDR output, GM_ADDR config) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  InitSocState();
  if (GetBlockIdx() != 0) return;
  CommitState(state, inputs, block_table, output, reinterpret_cast<__gm__ int64_t*>(config));
}

// Distinct banks may run concurrently; each bank preserves request order.
// Descriptors and all pointed-to banks/tables are retained by the graph owner.
extern "C" __global__ __aicore__ void qwen_mamba_commit_group_v1(GM_ADDR inputs, GM_ADDR descriptors, GM_ADDR output,
                                                                 GM_ADDR config) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  InitSocState();
  const int64_t states = reinterpret_cast<__gm__ int64_t*>(config)[0];
  if (states < 1 || states > 256) return;
  auto records = reinterpret_cast<__gm__ int64_t*>(descriptors);
  for (int64_t index = GetBlockIdx(); index < states; index += GetBlockNum()) {
    const auto record = records + index * 8;
    CommitState(reinterpret_cast<GM_ADDR>(record[0]), inputs, reinterpret_cast<GM_ADDR>(record[1]),
                output + index * 2 * PADDED_ROWS * sizeof(int32_t), record + 2);
  }
}
