#include "qsa_sparse_attention_v310.h"
#include "qsa_cube_sparse_attention_v310.h"

extern "C" __global__ __aicore__ void qsa_sparse_attention_v310(
    GM_ADDR query, GM_ADDR keyCache, GM_ADDR valueCache, GM_ADDR groupIndices, GM_ADDR groupCounts,
    GM_ADDR tailStarts, GM_ADDR tailCounts, GM_ADDR blockTable, GM_ADDR queryStartLoc, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    REGISTER_TILING_DEFAULT(QsaSparseAttentionV310TilingData);
    GET_TILING_DATA_WITH_STRUCT(QsaSparseAttentionV310TilingData, tilingData, tiling);
    AscendC::TPipe pipe;
    if (tilingData.headDim == NsQsaCubeSparseAttention::MAX_HEAD_DIM &&
        tilingData.headsPerTask <= NsQsaCubeSparseAttention::MAX_QUERY_HEADS &&
        tilingData.selectedGroupsWidth <= NsQsaCubeSparseAttention::MAX_GROUP_WIDTH) {
        NsQsaCubeSparseAttention::QsaCubeSparseAttentionV310 op;
        op.Init(query, keyCache, valueCache, groupIndices, groupCounts, tailStarts, tailCounts, blockTable,
                queryStartLoc, output, &tilingData, &pipe);
        op.Process();
    } else {
        NsQsaSparseAttention::QsaSparseAttentionV310 op;
        op.Init(query, keyCache, valueCache, groupIndices, groupCounts, tailStarts, tailCounts, blockTable,
                queryStartLoc, output, &tilingData, &pipe);
        op.Process();
    }
}
