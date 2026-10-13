// SPDX-License-Identifier: Apache-2.0
// Gather token-quantized operands for local routes only, using device group ends.
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void qwen_local_route_gather_v1(GM_ADDR low, GM_ADDR high, GM_ADDR scale, GM_ADDR sum,
                                                                 GM_ADDR token_rows, GM_ADDR group_ends,
                                                                 GM_ADDR out_low, GM_ADDR out_high, GM_ADDR out_scale,
                                                                 GM_ADDR out_sum, GM_ADDR config) {
  AscendC::InitSocState();
  using namespace AscendC;
  auto c = reinterpret_cast<__gm__ int64_t*>(config);
  const int64_t packedWidth = c[0], metadataWidth = c[1], experts = c[2];
  GlobalTensor<int8_t> inputLow, inputHigh, outputLow, outputHigh;
  GlobalTensor<float> inputScale, inputSum, outputScale, outputSum;
  GlobalTensor<int32_t> rows;
  GlobalTensor<int64_t> ends;
  inputLow.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(low));
  inputHigh.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(high));
  outputLow.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(out_low));
  outputHigh.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(out_high));
  inputScale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));
  inputSum.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sum));
  outputScale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out_scale));
  outputSum.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out_sum));
  rows.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(token_rows));
  ends.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(group_ends));
  const int64_t activeRows = ends.GetValue(experts - 1);
  TPipe pipe;
  TBuf<TPosition::VECCALC> bytes, floats;
  // Four disjoint operand regions let MTE2 fill the complete row before one
  // producer/consumer handoff. Reusing one region for each limb forced four
  // independent DMA round trips and eight event waits per route.
  pipe.InitBuffer(bytes, 2 * packedWidth);
  pipe.InitBuffer(floats, 2 * metadataWidth * sizeof(float));
  auto localBytes = bytes.Get<int8_t>();
  auto localFloats = floats.Get<float>();
  for (int64_t row = GetBlockIdx(); row < activeRows; row += GetBlockNum()) {
    const int64_t source = rows.GetValue(row);
    DataCopy(localBytes, inputLow[source * packedWidth], packedWidth);
    DataCopy(localBytes[packedWidth], inputHigh[source * packedWidth], packedWidth);
    DataCopy(localFloats, inputScale[source * metadataWidth], metadataWidth);
    DataCopy(localFloats[metadataWidth], inputSum[source * metadataWidth], metadataWidth);
    SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    DataCopy(outputLow[row * packedWidth], localBytes, packedWidth);
    DataCopy(outputHigh[row * packedWidth], localBytes[packedWidth], packedWidth);
    DataCopy(outputScale[row * metadataWidth], localFloats, metadataWidth);
    DataCopy(outputSum[row * metadataWidth], localFloats[metadataWidth], metadataWidth);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
  }
}
