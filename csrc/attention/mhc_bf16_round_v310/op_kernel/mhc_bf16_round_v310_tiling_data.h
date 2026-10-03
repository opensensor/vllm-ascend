#ifndef MHC_BF16_ROUND_V310_TILING_DATA_H
#define MHC_BF16_ROUND_V310_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

struct MhcBf16RoundV310TilingData {
    int64_t numel;
    uint32_t blockCount;
};

#endif
