#include "qsa_indexer_score_v310.h"

extern "C" __global__ __aicore__ void qsa_indexer_score_v310(
    GM_ADDR query, GM_ADDR compressedKeyCache, GM_ADDR blockTable, GM_ADDR queryStartLoc,
    GM_ADDR positions, GM_ADDR scores, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(QsaIndexerScoreV310TilingData);
    GET_TILING_DATA_WITH_STRUCT(QsaIndexerScoreV310TilingData, tilingData, tiling);
    AscendC::TPipe pipe;
    NsQsaIndexerScore::QsaIndexerScoreV310 op;
    op.Init(query, compressedKeyCache, blockTable, queryStartLoc, positions, scores, &tilingData, &pipe);
    op.Process();
}
