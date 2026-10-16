#ifndef SIDECARD_MHC_PRE_V310_TILING_H_DATA
#define SIDECARD_MHC_PRE_V310_TILING_H_DATA
#include "kernel_tiling/kernel_tiling.h"
struct SidecardMhcPreV310TilingData {
  int64_t rows;
  uint32_t blocks;
  float normEps;
  float hcEps;
};
#endif
