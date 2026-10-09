// SPDX-License-Identifier: Apache-2.0
// Tile-local FP32 -> FP16 input conversion, FP16 Cube -> FP32 projection.
#include "kernel_operator.h"
namespace {
using namespace AscendC;
#ifndef GLM_MHC_PROJECTION_TILE_K
  #define GLM_MHC_PROJECTION_TILE_K 256
#endif
constexpr uint32_t M = 16, N = 32, K_TILE = GLM_MHC_PROJECTION_TILE_K, K = 16384, NZ = 16;
static_assert(K_TILE == 256 || K_TILE == 512 || K_TILE == 1024, "unqualified projection K tile");
constexpr uint32_t INPUT_ELEMENTS = M * K_TILE;
constexpr uint32_t PRODUCT_ELEMENTS = M * N;
constexpr IsResetLoad3dConfig LOAD_CONFIG = {true, true};
class MhcProjection {
 public:
  __aicore__ inline void Run(GM_ADDR input, GM_ADDR weight, GM_ADDR output, GM_ADDR config, GM_ADDR offsets) {
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
    w_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weight));
    y_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    auto descriptor = reinterpret_cast<__gm__ int64_t*>(config);
    const int64_t rows = descriptor[0];
    GlobalTensor<uint32_t> table;
    table.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(offsets));
    pipe_.InitBuffer(input_, INPUT_ELEMENTS * sizeof(float));
    pipe_.InitBuffer(halfInput_, INPUT_ELEMENTS * sizeof(half));
    pipe_.InitBuffer(packedInput_, INPUT_ELEMENTS * sizeof(half));
    pipe_.InitBuffer(indices_, (INPUT_ELEMENTS + PRODUCT_ELEMENTS) * sizeof(uint32_t));
    pipe_.InitBuffer(products_, 2 * PRODUCT_ELEMENTS * sizeof(float));
    pipe_.InitBuffer(a1_, INPUT_ELEMENTS * sizeof(half));
    pipe_.InitBuffer(a2_, INPUT_ELEMENTS * sizeof(half));
    pipe_.InitBuffer(b1_, N * K_TILE * sizeof(half));
    pipe_.InitBuffer(b2_, N * K_TILE * sizeof(half));
    pipe_.InitBuffer(c_, PRODUCT_ELEMENTS * sizeof(float));
    auto indices = indices_.Get<uint32_t>();
    DataCopy(indices, table, INPUT_ELEMENTS + PRODUCT_ELEMENTS);
    PipeBarrier<PIPE_ALL>();
    for (int64_t row = GetBlockIdx() * M; row < rows; row += GetBlockNum() * M) {
      const uint32_t count = rows - row < M ? rows - row : M;
      for (uint32_t k0 = 0; k0 < K; k0 += K_TILE) {
        auto x = input_.Get<float>();
        Duplicate(x, 0.0f, INPUT_ELEMENTS);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t m = 0; m < count; ++m) DataCopy(x[m * K_TILE], x_[(row + m) * K + k0], K_TILE);
        DataCopy(b1_.Get<half>(), w_[k0 * N], N * K_TILE);
        PipeBarrier<PIPE_ALL>();
        Cast(halfInput_.Get<half>(), x, RoundMode::CAST_NONE, INPUT_ELEMENTS);
        PipeBarrier<PIPE_V>();
        Gather(packedInput_.Get<half>(), halfInput_.Get<half>(), indices, static_cast<uint32_t>(0), INPUT_ELEMENTS);
        PipeBarrier<PIPE_ALL>();
        DataCopy(a1_.Get<half>(), packedInput_.Get<half>(), INPUT_ELEMENTS);
        PipeBarrier<PIPE_ALL>();
        LoadData3DParamsV2<half> loadA;
        loadA.l1H = M / NZ;
        loadA.l1W = NZ;
        loadA.channelSize = K_TILE;
        loadA.padList[0] = 0;
        loadA.padList[1] = 0;
        loadA.padList[2] = 0;
        loadA.padList[3] = 255;
        loadA.mExtension = M;
        loadA.kExtension = K_TILE;
        loadA.mStartPt = 0;
        loadA.kStartPt = 0;
        loadA.strideW = 1;
        loadA.strideH = 1;
        loadA.filterW = 1;
        loadA.filterSizeW = false;
        loadA.filterH = 1;
        loadA.filterSizeH = false;
        loadA.dilationFilterW = 1;
        loadA.dilationFilterH = 1;
        loadA.enTranspose = 0;
        loadA.fMatrixCtrl = 0;
        LoadData<half, LOAD_CONFIG>(a2_.Get<half>(), a1_.Get<half>(), loadA);
        LoadData2DParams loadB;
        loadB.startIndex = 0;
        loadB.repeatTimes = N / NZ * K_TILE / NZ;
        loadB.srcStride = 1;
        loadB.dstGap = 0;
        loadB.ifTranspose = false;
        LoadData(b2_.Get<half>(), b1_.Get<half>(), loadB);
        PipeBarrier<PIPE_ALL>();
        MmadParams mm;
        mm.m = M;
        mm.n = N;
        mm.k = K_TILE;
        mm.cmatrixInitVal = k0 == 0;
        Mmad(c_.Get<float>(), a2_.Get<half>(), b2_.Get<half>(), mm);
        PipeBarrier<PIPE_ALL>();
      }
      auto raw = products_.Get<float>();
      const DataCopyParams copy{N / NZ, M / NZ, 0, 0};
      DataCopyEnhancedParams enhanced;
      enhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
      DataCopy(raw, c_.Get<float>(), copy, enhanced);
      PipeBarrier<PIPE_V>();
      Gather(raw[PRODUCT_ELEMENTS], raw, indices[INPUT_ELEMENTS], static_cast<uint32_t>(0), PRODUCT_ELEMENTS);
      PipeBarrier<PIPE_ALL>();
      DataCopy(y_[row * N], raw[PRODUCT_ELEMENTS], count * N);
      PipeBarrier<PIPE_ALL>();
    }
  }

 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> input_, halfInput_, packedInput_, indices_, products_;
  TBuf<TPosition::A1> a1_;
  TBuf<TPosition::B1> b1_;
  TBuf<TPosition::A2> a2_;
  TBuf<TPosition::B2> b2_;
  TBuf<TPosition::CO1> c_;
  GlobalTensor<float> x_, y_;
  GlobalTensor<half> w_;
};
}  // namespace
extern "C" __global__ __aicore__ void glm_mhc_projection_fp32_v1(GM_ADDR input, GM_ADDR weight, GM_ADDR output,
                                                                 GM_ADDR config, GM_ADDR offsets) {
  InitSocState();
  MhcProjection operation;
  operation.Run(input, weight, output, config, offsets);
}
