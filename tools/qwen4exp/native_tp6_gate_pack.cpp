// SPDX-License-Identifier: Apache-2.0
// Offline candidate. Pair one gate/up quantization group per core; preserve the
// original grouped accumulation and FP16 projection/activation boundaries.
#include "tp6_local_schedule.h"
#include "tp6_local_pack.h"
#include "qwen_w4_a8_int4_matmul_v310_tiling_data.h"
namespace {
using namespace AscendC;
constexpr uint32_t M = 16, N = 128, GROUPS = 5;
class GatePack : public native_int4::Schedule<M, N> {
 public:
  __aicore__ inline void Run(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR sums, GM_ADDR codes, GM_ADDR scale,
                             GM_ADDR offset, GM_ADDR ws, GM_ADDR ends, GM_ADDR outputLow, GM_ADDR outHigh,
                             GM_ADDR outScale, GM_ADDR outSum, __gm__ const QwenW4A8Int4KernelTilingData* td) {
    Init(low, high, xs, sums, codes, scale, offset, ws, ends, nullptr, outputLow, td);
    pipe_.InitBuffer(gate_, M * N * sizeof(half));
    pack_.Init(pipe_, outputLow, outHigh, outScale, outSum);
    int32_t expertIds[128];
    uint32_t counts[128], groupOfRow[128], starts[129];
    uint32_t routeGroups = 0;
    if (routed_) {
      auto cached = routeCache_.Get<int32_t>();
      const uint32_t aligned = rows_ / 8 * 8;
      if (aligned) DataCopy(cached, routeIds_[0], aligned);
      SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
      for (uint32_t row = aligned; row < rows_; ++row) cached.SetValue(row, routeIds_.GetValue(row));
      routeGroups = native_int4::BuildRouteGroups<false>(cached, rows_, experts_, broadcastFactor_, expertIds, counts,
                                                         groupOfRow, starts, routeRows_, sourceRows_);
    }
    for (uint32_t group = GetBlockIdx(); group < GROUPS; group += GetBlockNum()) {
      // Every output byte is initialized, including all-peer routes. Down
      // projection can use its ordinary consumer without a private tail ABI.
      auto gate = gate_.Get<half>();
      auto up = output_.Get<half>();
      Duplicate(gate, static_cast<half>(0), M * N);
      Duplicate(up, static_cast<half>(0), M * N);
      PipeBarrier<PIPE_V>();
      for (int64_t row = 0; row < rows_; row += M)
        pack_.Emit(gate, up, group, row, Min(M, rows_ - row), nullptr, false);
      const uint32_t groups = routed_ ? routeGroups : experts_;
      for (uint32_t g = 0; g < groups; ++g) {
        const int64_t expert = routed_ ? expertIds[g] : g;
        const int64_t begin = routed_ ? starts[g] : (g == 0 ? 0 : ends_.GetValue(g - 1));
        const int64_t end = routed_ ? starts[g + 1] : ends_.GetValue(g);
        if (begin < 0 || end < begin || end > rows_) {
          ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "invalid gate-pack boundaries"); });
          return;
        }
        for (int64_t row = begin; row < end; row += M) {
          const uint32_t count = Min(M, end - row);
          ProjectHalf(expert, group, row, count);
          auto output = output_.Get<half>();
          Adds(gate, output, static_cast<half>(0), M * N);
          PipeBarrier<PIPE_V>();
          ProjectHalf(expert, group + GROUPS, row, count);
          pack_.Emit(gate, output, group, row, count, routeRows_, routed_);
        }
      }
    }
  }

 private:
  __aicore__ inline void ProjectHalf(int64_t expert, int64_t tile, int64_t row, uint32_t count) {
    DataCopy(b1_.Get<int8_t>(), codes_[(expert * n_ + tile * N) * k_ / 2], N * k_ / 2);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    Project(expert, tile, row, count);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
  }
  TBuf<TPosition::VECCALC> gate_;
  tp6_pack::LocalPack pack_;
};
}  // namespace
extern "C" __global__ __aicore__ void qwen_tp6_gate_pack_v1(GM_ADDR low, GM_ADDR high, GM_ADDR xs, GM_ADDR sums,
                                                            GM_ADDR codes, GM_ADDR scale, GM_ADDR offset, GM_ADDR ws,
                                                            GM_ADDR ends, GM_ADDR outputLow, GM_ADDR outHigh,
                                                            GM_ADDR outScale, GM_ADDR outSum, GM_ADDR descriptor) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
  InitSocState();
  auto td = reinterpret_cast<__gm__ QwenW4A8Int4KernelTilingData*>(descriptor);
  if (td->numRows <= 0 || td->numRows > 20480 || td->numExperts < 85 || td->numExperts > 86 || td->nDim != 1280 ||
      td->kDim != 2560 || td->metadataLanes != 8 || (td->routed != 0 && td->routed != 1) ||
      (td->routed && td->numRows > 128) || (td->broadcastFactor != 1 && td->broadcastFactor != 10))
    return;
  GatePack op;
  op.Run(low, high, xs, sums, codes, scale, offset, ws, ends, outputLow, outHigh, outScale, outSum, td);
}
