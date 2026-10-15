// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#define QSA_GEOMETRY_INLINE __aicore__ inline
#include "qsa_geometry_integer.h"

using namespace AscendC;

namespace {
constexpr int64_t PADDED_ROWS = 8;
constexpr uint32_t OUTPUT_ELEMENTS = 3 * PADDED_ROWS;
}  // namespace

extern "C" __global__ __aicore__ void qwen_qsa_geometry_v1(GM_ADDR positions, GM_ADDR output, GM_ADDR config) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  InitSocState();
  if (GetBlockIdx() != 0) return;
  const int64_t rows = reinterpret_cast<__gm__ int64_t*>(config)[0];
  if (rows != 3 && rows != 6) return;
  GlobalTensor<int32_t> input, result;
  input.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(positions));
  result.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output));
  TPipe pipe;
  TBuf<TPosition::VECCALC> storage;
  pipe.InitBuffer(storage, OUTPUT_ELEMENTS * sizeof(int32_t));
  auto local = storage.Get<int32_t>();
  for (uint32_t i = 0; i < OUTPUT_ELEMENTS; ++i) local.SetValue(i, 0);
  // Scalar loads cover only actual input elements; the store is aligned96B.
  for (int64_t row = 0; row < rows; ++row) {
    const auto geometry = qwen_geometry::Position(input.GetValue(row));
    local.SetValue(row, geometry.groups);
    local.SetValue(PADDED_ROWS + row, geometry.start);
    local.SetValue(2 * PADDED_ROWS + row, geometry.count);
  }
  SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
  DataCopy(result, local, OUTPUT_ELEMENTS);
  SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
  WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}
