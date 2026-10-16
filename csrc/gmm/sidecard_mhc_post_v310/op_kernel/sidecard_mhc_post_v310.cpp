#include "sidecard_mhc_post_v310.h"
extern "C" __global__ __aicore__ void sidecard_mhc_post_v310(GM_ADDR x, GM_ADDR residual, GM_ADDR post, GM_ADDR comb,
                                                             GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  REGISTER_TILING_DEFAULT(SidecardMhcPostV310TilingData);
  GET_TILING_DATA_WITH_STRUCT(SidecardMhcPostV310TilingData, tilingData, tiling);
  AscendC::TPipe pipe;
  NsSidecardMhcPostV310::Kernel kernel;
  kernel.Init(x, residual, post, comb, output, &tilingData, &pipe);
  kernel.Process();
}
