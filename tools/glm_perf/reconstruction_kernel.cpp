// SPDX-License-Identifier: Apache-2.0
// Compiled with the shared packed decoder from a hashed source snapshot.
#include "compat_310p.h"
#include "w2_blocked_dequant_matmul_v310.h"

extern "C" __global__ __aicore__ void glm_reconstruction_v1(GM_ADDR x, GM_ADDR codes, GM_ADDR scale, GM_ADDR ends,
                                                            GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  AscendC::InitSocState();
  int64_t config[7];
  const auto source = reinterpret_cast<__gm__ int64_t*>(tiling);
  for (unsigned i = 0; i < 7; ++i) config[i] = source[i];
  const int64_t rows = config[0], experts = config[1], n = config[2], k = config[3];
  const int64_t mode = config[4], residentRows = config[6];
  const bool nz = config[5] != 0;
  if (rows <= 0 || rows > 64 || experts <= 0 || n <= 0 || n % 128 != 0 || k < 256 || k > 4096 || k % 256 != 0 ||
      mode != 3 || !nz || residentRows < 0 || residentRows > 128)
    return;
  const int64_t packedK = k * NsW2::W3_BYTES_PER_GROUP / NsW2::W3_CODES_PER_GROUP;
  const int64_t scaleStride = (n / NsW2::W2_BLOCK_SIZE) * (k / NsW2::W2_BLOCK_SIZE);
  AscendC::GlobalTensor<int64_t> boundaries;
  boundaries.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(ends));
  NsW2::W2BlockedDequantMatmulV310Cube operation;
  bool tablesReady = false;
  int64_t first = 0;
  for (int64_t expert = 0; expert < experts; ++expert) {
    const int64_t end = boundaries.GetValue(expert);
    if (end < first || end > rows) return;
    if (end > first) {
      const bool resident = residentRows > 0 && end - first <= residentRows && k % NsW2::W2_L1_WIDE_K == 0;
      operation.InitGeometry(x + first * k * sizeof(half), codes + expert * n * packedK,
                             scale + expert * scaleStride * sizeof(float), y + first * n * sizeof(half), workspace,
                             end - first, n, k, mode, nz, resident);
      operation.Process(tablesReady);
      tablesReady = true;
      AscendC::PipeBarrier<PIPE_ALL>();
    }
    first = end;
  }
}
