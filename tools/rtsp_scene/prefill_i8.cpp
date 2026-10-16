// SPDX-License-Identifier: Apache-2.0
// Large-row INT8 projection with bounded L1/UB storage and direct ND output.
#include "kernel_operator.h"

#ifndef RTSP_PREFILL_K
  #define RTSP_PREFILL_K 4096
#endif
#ifndef RTSP_PREFILL_M
  #define RTSP_PREFILL_M 64
#endif
#ifndef RTSP_PREFILL_N
  #define RTSP_PREFILL_N 128
#endif
namespace {
using namespace AscendC;
constexpr uint32_t INNER = RTSP_PREFILL_K, ROWS = RTSP_PREFILL_M, COLUMNS = RTSP_PREFILL_N;
constexpr uint32_t K_TILE = 256, CELL_K = 32, CELL_N = 16, PRODUCTS = ROWS * COLUMNS;
static_assert(ROWS == 32 || ROWS == 64);
static_assert(COLUMNS == 128 || COLUMNS == 256);
static_assert(ROWS * K_TILE <= 65536 && COLUMNS * K_TILE <= 65536);

class PrefillProjection {
 public:
  __aicore__ inline void Run(GM_ADDR x, GM_ADDR w, GM_ADDR ws, GM_ADDR xs, GM_ADDR y, GM_ADDR geometry) {
    const auto config = reinterpret_cast<__gm__ int64_t*>(geometry);
    rows_ = config[0];
    columns_ = config[1];
    if (rows_ <= 0 || columns_ % COLUMNS != 0 || config[2] != 0 || config[3] != 1 || config[4] != 1) return;
    padded_rows_ = (rows_ + CELL_N - 1) / CELL_N * CELL_N;
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x));
    w_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(w));
    ws_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws));
    xs_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(xs));
    y_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y));
    pipe_.InitBuffer(a1_, ROWS * K_TILE);
    pipe_.InitBuffer(a2_, ROWS * K_TILE);
    pipe_.InitBuffer(b1_, COLUMNS * K_TILE);
    pipe_.InitBuffer(b2_, COLUMNS * K_TILE);
    pipe_.InitBuffer(c_, PRODUCTS * sizeof(int32_t));
    pipe_.InitBuffer(products_, PRODUCTS * sizeof(int32_t));
    pipe_.InitBuffer(values_, PRODUCTS * sizeof(float));
    pipe_.InitBuffer(scales_, COLUMNS * sizeof(float));
    pipe_.InitBuffer(output_, PRODUCTS * sizeof(half));
    // Last M tile can own fewer physical rows than ROWS; copy only allocated
    // input cells and zero the remainder without reading beyond the NZ tensor.
    const int64_t column_tiles = columns_ / COLUMNS;
    const int64_t row_tiles = (rows_ + ROWS - 1) / ROWS;
    for (int64_t tile = GetBlockIdx(); tile < row_tiles * column_tiles; tile += GetBlockNum())
      Project(tile / column_tiles * ROWS, tile % column_tiles);
  }

 private:
  __aicore__ inline void Project(int64_t start_row, int64_t column_tile) {
    const uint32_t physical_rows = padded_rows_ - start_row < ROWS ? padded_rows_ - start_row : ROWS;
    const uint32_t valid_rows = rows_ - start_row < ROWS ? rows_ - start_row : ROWS;
    for (uint32_t inner = 0; inner < INNER; inner += K_TILE) {
      // NZ storage is [K/32, padded-M/16, 16, 32]. A tile retains that
      // ordering, including both row blocks when M exceeds sixteen.
      const DataCopyParams copy_a{K_TILE / CELL_K, static_cast<uint16_t>(physical_rows),
                                  static_cast<uint16_t>(padded_rows_ - physical_rows),
                                  static_cast<uint16_t>(ROWS - physical_rows)};
      if (physical_rows != ROWS) {
        // L1 cannot be vector-filled; copy a zeroed UB tile first.
        Duplicate(values_.Get<int16_t>(), static_cast<int16_t>(0), ROWS * K_TILE / 2);
        PipeBarrier<PIPE_ALL>();
        DataCopy(a1_.Get<int8_t>(), values_.Get<int8_t>(), ROWS * K_TILE);
        PipeBarrier<PIPE_ALL>();
      }
      DataCopy(a1_.Get<int8_t>(), x_[inner * padded_rows_ + start_row * CELL_K], copy_a);
      DataCopy(b1_.Get<int8_t>(), w_[column_tile * INNER * COLUMNS + inner * COLUMNS], COLUMNS * K_TILE);
      PipeBarrier<PIPE_ALL>();
      LoadData2DParams load_a;
      load_a.repeatTimes = K_TILE / CELL_K;
      load_a.srcStride = ROWS / CELL_N;
      load_a.ifTranspose = false;
      // L0A is row-block major, while GM/L1 NZ is K-block major.
      for (uint32_t row_block = 0; row_block < ROWS / CELL_N; ++row_block)
        LoadData(a2_.Get<int8_t>()[row_block * CELL_N * K_TILE], a1_.Get<int8_t>()[row_block * CELL_N * CELL_K],
                 load_a);
      LoadData2DParams load_b;
      load_b.repeatTimes = K_TILE / CELL_K * (COLUMNS / CELL_N);
      load_b.srcStride = 1;
      load_b.ifTranspose = false;
      LoadData(b2_.Get<int8_t>(), b1_.Get<int8_t>(), load_b);
      PipeBarrier<PIPE_ALL>();
      MmadParams mm;
      mm.m = ROWS;
      mm.n = COLUMNS;
      mm.k = K_TILE;
      mm.cmatrixInitVal = inner == 0;
      Mmad(c_.Get<int32_t>(), a2_.Get<int8_t>(), b2_.Get<int8_t>(), mm);
      PipeBarrier<PIPE_ALL>();
    }
    const DataCopyParams copy_c{PRODUCTS / (CELL_N * CELL_N), 1, 0, 0};
    DataCopyEnhancedParams enhanced;
    enhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(products_.Get<int32_t>(), c_.Get<int32_t>(), copy_c, enhanced);
    DataCopy(scales_.Get<float>(), ws_[column_tile * COLUMNS], COLUMNS);
    PipeBarrier<PIPE_ALL>();
    auto values = values_.Get<float>();
    Cast(values, products_.Get<int32_t>(), RoundMode::CAST_NONE, PRODUCTS);
    PipeBarrier<PIPE_V>();
    for (uint32_t strip = 0; strip < COLUMNS / CELL_N; ++strip)
      Mul(values[strip * ROWS * CELL_N], values[strip * ROWS * CELL_N], scales_.Get<float>()[strip * CELL_N], CELL_N,
          ROWS, {1, 1, 1, 2, 2, 0});
    PipeBarrier<PIPE_V>();
    for (uint32_t row = 0; row < valid_rows; ++row)
      Muls(values[row * CELL_N], values[row * CELL_N], xs_.GetValue(start_row + row), CELL_N, COLUMNS / CELL_N,
           {1, 1, ROWS * CELL_N / 8, ROWS * CELL_N / 8});
    PipeBarrier<PIPE_V>();
    auto output = output_.Get<half>();
    Cast(output, values, RoundMode::CAST_NONE, PRODUCTS);
    PipeBarrier<PIPE_ALL>();
    const DataCopyParams scatter{COLUMNS / CELL_N, 1, ROWS - 1, 0};
    for (uint32_t row = 0; row < valid_rows; ++row)
      DataCopy(y_[(start_row + row) * columns_ + column_tile * COLUMNS], output[row * CELL_N], scatter);
    PipeBarrier<PIPE_ALL>();
  }
  int64_t rows_ = 0, columns_ = 0, padded_rows_ = 0;
  TPipe pipe_;
  GlobalTensor<int8_t> x_, w_;
  GlobalTensor<float> ws_, xs_;
  GlobalTensor<half> y_;
  TBuf<TPosition::A1> a1_;
  TBuf<TPosition::A2> a2_;
  TBuf<TPosition::B1> b1_;
  TBuf<TPosition::B2> b2_;
  TBuf<TPosition::CO1> c_;
  TBuf<TPosition::VECCALC> products_, values_, scales_, output_;
};
}  // namespace
extern "C" __global__ __aicore__ void rtsp_prefill_i8(GM_ADDR x, GM_ADDR w, GM_ADDR ws, GM_ADDR xs, GM_ADDR y,
                                                      GM_ADDR config) {
  PrefillProjection op;
  op.Run(x, w, ws, xs, y, config);
}
