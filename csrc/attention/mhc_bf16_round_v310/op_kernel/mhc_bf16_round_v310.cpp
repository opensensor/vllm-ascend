#include "mhc_bf16_round_v310.h"

extern "C" __global__ __aicore__ void mhc_bf16_round_v310(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MhcBf16RoundV310TilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcBf16RoundV310TilingData, tilingData, tiling);
    AscendC::TPipe pipe;
    NsMhcBf16Round::MhcBf16RoundV310 op;
    op.Init(input, output, &tilingData, &pipe);
    op.Process();
}
