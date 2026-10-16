// SPDX-License-Identifier: Apache-2.0
// Narrow dense INT8 Cube projection. Frozen weights are output-tile-major NZ.
#include "kernel_operator.h"

#ifndef RTSP_DENSE_N
  #define RTSP_DENSE_N 128
#endif
#ifndef RTSP_DENSE_K
  #define RTSP_DENSE_K 4096
#endif
#ifndef RTSP_DENSE_K_TILE
  #define RTSP_DENSE_K_TILE 256
#endif
#ifndef RTSP_DENSE_PIPELINE
  #define RTSP_DENSE_PIPELINE 0
#endif
#ifndef RTSP_DENSE_ND_SCATTER
  #define RTSP_DENSE_ND_SCATTER 0
#endif
#ifndef RTSP_DENSE_ACTIVE_EPILOGUE
  #define RTSP_DENSE_ACTIVE_EPILOGUE 0
#endif

namespace {
using namespace AscendC;
constexpr uint32_t ROWS = 16, COLUMNS = RTSP_DENSE_N, INNER = RTSP_DENSE_K;
constexpr uint32_t K_TILE = RTSP_DENSE_K_TILE, CELL_K = 32, CELL_N = 16;
constexpr uint32_t PRODUCTS = ROWS * COLUMNS;
constexpr uint32_t FP32_VECTOR_LANES = 64, VECTOR_ROWS = FP32_VECTOR_LANES / CELL_N;
constexpr uint32_t FP32_STRIP_BLOCKS = ROWS * CELL_N * sizeof(float) / 32;
constexpr uint32_t FP16_STRIP_BLOCKS = ROWS * CELL_N * sizeof(half) / 32;
static_assert(INNER % K_TILE == 0 && K_TILE % CELL_K == 0);
static_assert(COLUMNS == 64 || COLUMNS == 128 || COLUMNS == 256 || COLUMNS == 512);
static_assert(COLUMNS * K_TILE <= 65536, "L0B must fit the 310P buffer");

class DenseProjection {
 public:
  __aicore__ inline void Run(GM_ADDR input, GM_ADDR weight, GM_ADDR ws, GM_ADDR xs, GM_ADDR output, GM_ADDR geometry) {
    auto config = reinterpret_cast<__gm__ int64_t*>(geometry);
    rows_ = config[0];
    columns_ = config[1];
    nz_ = config[2] != 0;
    input_nz_ = config[3] != 0;
    scale_stride_ = config[4];
    if (rows_ < 1 || rows_ > ROWS || columns_ % COLUMNS != 0) return;
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(input));
    w_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(weight));
    ws_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws));
    xs_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(xs));
    y_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
    Allocate();
    PrepareInput();
    for (int64_t tile = GetBlockIdx(); tile < columns_ / COLUMNS; tile += GetBlockNum()) Project(tile);
  }

 private:
  __aicore__ inline void Allocate() {
    pipe_.InitBuffer(input_, ROWS * INNER);
    pipe_.InitBuffer(a1_, ROWS * INNER);
    pipe_.InitBuffer(a2_, ROWS * K_TILE);
    pipe_.InitBuffer(b1_, (RTSP_DENSE_PIPELINE ? 2 : 1) * COLUMNS * K_TILE);
    pipe_.InitBuffer(b2_, COLUMNS * K_TILE);
    pipe_.InitBuffer(c_, PRODUCTS * sizeof(int32_t));
    pipe_.InitBuffer(products_, PRODUCTS * sizeof(int32_t));
    pipe_.InitBuffer(values_, PRODUCTS * sizeof(float));
    pipe_.InitBuffer(scale_, COLUMNS * sizeof(float));
    pipe_.InitBuffer(output_, PRODUCTS * sizeof(half));
#if !RTSP_DENSE_ND_SCATTER
    pipe_.InitBuffer(nd_, PRODUCTS * sizeof(half));
#endif
  }

  __aicore__ inline void PrepareInput() {
    auto packed = input_.Get<int8_t>();
    if (input_nz_) {
      DataCopy(a1_.Get<int8_t>(), x_, ROWS * INNER);
    } else {
      Duplicate(packed.ReinterpretCast<int16_t>(), static_cast<int16_t>(0), ROWS * INNER / 2);
      PipeBarrier<PIPE_ALL>();
      const DataCopyParams copy{INNER / CELL_K, 1, 0, ROWS - 1};
      for (int64_t row = 0; row < rows_; ++row) DataCopy(packed[row * CELL_K], x_[row * INNER], copy);
      PipeBarrier<PIPE_ALL>();
      DataCopy(a1_.Get<int8_t>(), packed, ROWS * INNER);
    }
    PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline void Project(int64_t tile) {
#if RTSP_DENSE_PIPELINE
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    DataCopy(b1_.Get<int8_t>(), w_[tile * INNER * COLUMNS], COLUMNS * K_TILE);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
#endif
    for (uint32_t inner = 0; inner < INNER; inner += K_TILE) {
      const uint32_t slot = (inner / K_TILE) % 2;
#if RTSP_DENSE_PIPELINE
      const auto event = slot == 0 ? EVENT_ID0 : EVENT_ID1;
      WaitFlag<HardEvent::MTE2_MTE1>(event);
      WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
#else
      DataCopy(b1_.Get<int8_t>(), w_[tile * INNER * COLUMNS + inner * COLUMNS], COLUMNS * K_TILE);
      PipeBarrier<PIPE_ALL>();
#endif
      LoadData2DParams loadA;
      loadA.repeatTimes = K_TILE / CELL_K;
      loadA.srcStride = 1;
      loadA.ifTranspose = false;
      LoadData(a2_.Get<int8_t>(), a1_.Get<int8_t>()[inner * ROWS], loadA);
      LoadData2DParams loadB;
      loadB.repeatTimes = (K_TILE / CELL_K) * (COLUMNS / CELL_N);
      loadB.srcStride = 1;
      loadB.ifTranspose = false;
      LoadData(b2_.Get<int8_t>(), b1_.Get<int8_t>()[RTSP_DENSE_PIPELINE ? slot * COLUMNS * K_TILE : 0], loadB);
#if RTSP_DENSE_PIPELINE
      SetFlag<HardEvent::MTE1_MTE2>(event);
      SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
      WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
#else
      PipeBarrier<PIPE_ALL>();
#endif
      MmadParams mm;
      mm.m = ROWS;
      mm.n = COLUMNS;
      mm.k = K_TILE;
      mm.cmatrixInitVal = inner == 0;
      Mmad(c_.Get<int32_t>(), a2_.Get<int8_t>(), b2_.Get<int8_t>(), mm);
#if RTSP_DENSE_PIPELINE
      SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
      if (inner + K_TILE < INNER) {
        const uint32_t next = 1 - slot;
        const auto next_event = next == 0 ? EVENT_ID0 : EVENT_ID1;
        WaitFlag<HardEvent::MTE1_MTE2>(next_event);
        DataCopy(b1_.Get<int8_t>()[next * COLUMNS * K_TILE], w_[tile * INNER * COLUMNS + (inner + K_TILE) * COLUMNS],
                 COLUMNS * K_TILE);
        SetFlag<HardEvent::MTE2_MTE1>(next_event);
      }
#else
      PipeBarrier<PIPE_ALL>();
#endif
    }
#if RTSP_DENSE_PIPELINE
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    PipeBarrier<PIPE_ALL>();
#endif
    const DataCopyParams copy{COLUMNS / CELL_N, 1, 0, 0};
    DataCopyEnhancedParams enhanced;
    enhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(products_.Get<int32_t>(), c_.Get<int32_t>(), copy, enhanced);
    DataCopy(scale_.Get<float>(), ws_[tile * COLUMNS], COLUMNS);
    PipeBarrier<PIPE_ALL>();
    auto values = values_.Get<float>();
#if RTSP_DENSE_ACTIVE_EPILOGUE
    // NZ stores a physical 16-row strip. Convert only logical rows while
    // retaining its strip spacing and the original arithmetic boundaries.
    for (uint32_t row = 0; row < rows_; row += VECTOR_ROWS) {
      const uint64_t mask = (rows_ - row < VECTOR_ROWS ? rows_ - row : VECTOR_ROWS) * CELL_N;
      Cast(values[row * CELL_N], products_.Get<int32_t>()[row * CELL_N], RoundMode::CAST_NONE, mask, COLUMNS / CELL_N,
           {1, 1, FP32_STRIP_BLOCKS, FP32_STRIP_BLOCKS});
    }
#else
    Cast(values, products_.Get<int32_t>(), RoundMode::CAST_NONE, PRODUCTS);
#endif
    PipeBarrier<PIPE_V>();
    for (uint32_t strip = 0; strip < COLUMNS / CELL_N; ++strip)
      Mul(values[strip * ROWS * CELL_N], values[strip * ROWS * CELL_N], scale_.Get<float>()[strip * CELL_N], CELL_N,
          RTSP_DENSE_ACTIVE_EPILOGUE ? rows_ : ROWS, {1, 1, 1, 2, 2, 0});
    PipeBarrier<PIPE_V>();
    for (int64_t row = 0; row < rows_; ++row)
      Muls(values[row * CELL_N], values[row * CELL_N], xs_.GetValue(row * scale_stride_), CELL_N, COLUMNS / CELL_N,
           {1, 1, ROWS * CELL_N / 8, ROWS * CELL_N / 8});
    PipeBarrier<PIPE_V>();
    auto out = output_.Get<half>();
#if RTSP_DENSE_ACTIVE_EPILOGUE
    for (uint32_t row = 0; row < rows_; row += VECTOR_ROWS) {
      const uint64_t mask = (rows_ - row < VECTOR_ROWS ? rows_ - row : VECTOR_ROWS) * CELL_N;
      Cast(out[row * CELL_N], values[row * CELL_N], RoundMode::CAST_NONE, mask, COLUMNS / CELL_N,
           {1, 1, FP16_STRIP_BLOCKS, FP32_STRIP_BLOCKS});
    }
#else
    Cast(out, values, RoundMode::CAST_NONE, PRODUCTS);
#endif
    PipeBarrier<PIPE_ALL>();
    if (nz_) {
      DataCopy(y_[tile * PRODUCTS], out, PRODUCTS);
    } else {
#if RTSP_DENSE_ND_SCATTER
      // Each FP16 NZ cell is exactly one 32-byte DMA block. Scatter cells
      // into their final ND row, avoiding a UB transpose and vector Adds.
      // Consecutive source cells skip the other 15 rows; destination cells
      // are adjacent columns within this row, so the GM destination stride is zero.
      const DataCopyParams scatter{COLUMNS / CELL_N, 1, ROWS - 1, 0};
      for (int64_t row = 0; row < rows_; ++row)
        DataCopy(y_[row * columns_ + tile * COLUMNS], out[row * CELL_N], scatter);
#else
      auto row_major = nd_.Get<half>();
      for (uint32_t strip = 0; strip < COLUMNS / CELL_N; ++strip)
        Adds(row_major[strip * CELL_N], out[strip * ROWS * CELL_N], static_cast<half>(0), CELL_N, rows_,
             {1, 1, COLUMNS / CELL_N, 1});
      PipeBarrier<PIPE_ALL>();
      for (int64_t row = 0; row < rows_; ++row)
        DataCopy(y_[row * columns_ + tile * COLUMNS], row_major[row * COLUMNS], COLUMNS);
#endif
    }
    PipeBarrier<PIPE_ALL>();
  }

  int64_t rows_ = 0, columns_ = 0, scale_stride_ = 1;
  bool nz_ = false, input_nz_ = false;
  TPipe pipe_;
  GlobalTensor<int8_t> x_, w_;
  GlobalTensor<float> ws_, xs_;
  GlobalTensor<half> y_;
  TBuf<TPosition::VECCALC> input_, products_, values_, scale_, output_, nd_;
  TBuf<TPosition::A1> a1_;
  TBuf<TPosition::A2> a2_;
  TBuf<TPosition::B1> b1_;
  TBuf<TPosition::B2> b2_;
  TBuf<TPosition::CO1> c_;
};
}  // namespace

extern "C" __global__ __aicore__ void rtsp_dense_i8(GM_ADDR x, GM_ADDR w, GM_ADDR ws, GM_ADDR xs, GM_ADDR y,
                                                    GM_ADDR config) {
  DenseProjection op;
  op.Run(x, w, ws, xs, y, config);
}
