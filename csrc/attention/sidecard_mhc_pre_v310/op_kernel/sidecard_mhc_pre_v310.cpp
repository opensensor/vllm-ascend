#include "sidecard_mhc_pre_v310.h"
extern "C" __global__ __aicore__ void sidecard_mhc_pre_v310(GM_ADDR residual, GM_ADDR fn, GM_ADDR scale, GM_ADDR base,
                                                            GM_ADDR normWeight, GM_ADDR post, GM_ADDR logits,
                                                            GM_ADDR hidden, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  REGISTER_TILING_DEFAULT(SidecardMhcPreV310TilingData);
  GET_TILING_DATA_WITH_STRUCT(SidecardMhcPreV310TilingData, tilingData, tiling);
  AscendC::TPipe pipe;
  NsSidecardMhcPreV310::Kernel kernel;
  kernel.Init(residual, fn, scale, base, normWeight, post, logits, hidden, &tilingData, &pipe);
  kernel.Process();
}
