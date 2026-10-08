// SPDX-License-Identifier: Apache-2.0
// Fused exact INT32 QSA metadata; no floating-point conversion or AI-CPU.
#include "kernel_operator.h"
namespace {
using namespace AscendC;
constexpr int64_t TILE = 64;
constexpr int64_t DMA_ELEMENTS = 8;
__aicore__ inline int32_t Floor(int32_t value, int32_t divisor) {
  return value / divisor - (value < 0 && value % divisor != 0);
}
__aicore__ inline void Store(GlobalTensor<int32_t>& output, LocalTensor<int32_t>& local, int64_t offset,
                             int64_t valid) {
  const int64_t aligned = (valid + DMA_ELEMENTS - 1) / DMA_ELEMENTS * DMA_ELEMENTS;
  for (int64_t i = valid; i < aligned; ++i) local.SetValue(i, 0);
  SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
  DataCopy(output[offset], local, static_cast<uint32_t>(aligned));
  SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
  WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}
}  // namespace
extern "C" __global__ __aicore__ void glm_qsa_metadata_v1(GM_ADDR ids, GM_ADDR positions, GM_ADDR table, GM_ADDR groups,
                                                          GM_ADDR metadata, GM_ADDR logical_table, GM_ADDR config) {
  AscendC::InitSocState();
  auto c = reinterpret_cast<__gm__ int64_t*>(config);
  // rows, requests, budget, token_start, ids row/column strides,
  // position stride/bytes, table width/row/column strides, split, metadata pitch.
  const int64_t rows = c[0], requests = c[1], budget = c[2], start = c[3];
  const int64_t columns = (c[8] + c[11] - 1) / c[11];
  GlobalTensor<int32_t> input_ids, input_table, output_groups, output_metadata, output_table;
  GlobalTensor<int32_t> positions32;
  GlobalTensor<int64_t> positions64;
  input_ids.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ids));
  input_table.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(table));
  positions32.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(positions));
  positions64.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(positions));
  output_groups.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(groups));
  output_metadata.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(metadata));
  output_table.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(logical_table));
  TPipe pipe;
  TBuf<TPosition::VECCALC> buffer;
  pipe.InitBuffer(buffer, TILE * sizeof(int32_t));
  auto local = buffer.Get<int32_t>();
  // Assign entire linear tiles, rather than padded rows, so an odd-width
  // logical table cannot overwrite the following row or race another core.
  for (int64_t offset = GetBlockIdx() * TILE; offset < rows * budget; offset += GetBlockNum() * TILE) {
    const int64_t valid = rows * budget - offset < TILE ? rows * budget - offset : TILE;
    for (int64_t i = 0; i < valid; ++i) {
      const int64_t row = (offset + i) / budget, column = (offset + i) % budget;
      local.SetValue(i, Floor(input_ids.GetValue((start + row) * c[4] + column * 4 * c[5]), 4));
    }
    Store(output_groups, local, offset, valid);
  }
  for (int64_t offset = GetBlockIdx() * TILE; offset < requests * columns; offset += GetBlockNum() * TILE) {
    const int64_t valid = requests * columns - offset < TILE ? requests * columns - offset : TILE;
    for (int64_t i = 0; i < valid; ++i) {
      const int64_t row = (offset + i) / columns, column = (offset + i) % columns;
      local.SetValue(i, Floor(input_table.GetValue(row * c[9] + column * c[11] * c[10]), c[11]));
    }
    Store(output_table, local, offset, valid);
  }
  if (GetBlockIdx() != 0) return;
  for (int64_t field = 0; field < 3; ++field) {
    for (int64_t offset = 0; offset < rows; offset += TILE) {
      const int64_t valid = rows - offset < TILE ? rows - offset : TILE;
      for (int64_t i = 0; i < valid; ++i) {
        const int64_t position_offset = (offset + i) * c[6];
        const int32_t position = c[7] == 4 ? positions32.GetValue(position_offset)
                                           : static_cast<int32_t>(positions64.GetValue(position_offset));
        const int32_t length = position + 1;
        const int32_t pools = Floor(length, 4);
        const bool dense = length <= budget * 4;
        int32_t value;
        if (field == 0)
          value = dense ? length : (pools < budget ? pools : budget);
        else if (field == 1)
          value = pools * 4;
        else
          value = dense ? -1 : length - pools * 4;
        local.SetValue(i, value);
      }
      Store(output_metadata, local, field * c[12] + offset, valid);
    }
  }
}
