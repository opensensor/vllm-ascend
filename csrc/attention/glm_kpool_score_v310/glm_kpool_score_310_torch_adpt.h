// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <limits>
#include "score_geometry.h"
namespace vllm_ascend {
at::Tensor npu_glm_kpool_score_310(const at::Tensor& query, const at::Tensor& weights, const at::Tensor& cache,
    const at::Tensor& table, const at::Tensor& boundaries, const at::Tensor& positions,
    int64_t pools, int64_t blocks, int64_t block_rows, int64_t block_stride, int64_t row_stride, int64_t cache_offset) {
    TORCH_CHECK(query.device().type()==c10::DeviceType::PrivateUse1 && query.scalar_type()==at::kHalf &&
        query.dim()==3 && query.size(1)==32 && query.size(2)==128, "query must be NPU FP16 [rows,32,128]");
    for (const auto& t : {query, weights, cache, table, boundaries, positions}) {
        TORCH_CHECK(t.device()==query.device() && t.is_contiguous(), "score inputs must be contiguous on the same NPU");
    }
    TORCH_CHECK(weights.scalar_type()==at::kFloat && weights.dim()==2 && weights.size(0)==query.size(0) &&
        weights.size(1)==32, "weights must be FP32 [rows,32]");
    TORCH_CHECK(cache.scalar_type()==at::kHalf && cache.dim()==1, "cache must be flat FP16 backing storage");
    TORCH_CHECK(table.scalar_type()==at::kInt && table.dim()==2 && table.size(0)>0 &&
        boundaries.scalar_type()==at::kInt && boundaries.dim()==1 && boundaries.numel()==table.size(0) &&
        positions.scalar_type()==at::kInt && positions.dim()==1 && positions.numel()==query.size(0),
        "expected INT32 page table, cumulative query ends and positions");
    TORCH_CHECK(NsGlmKpoolScore::ValidLayout(query.size(0),pools,blocks,block_rows,block_stride,row_stride,cache_offset,
        cache.numel()) && pools<=table.size(1)*block_rows, "invalid kpool cache geometry");
    const c10_npu::OptionalNPUGuard guard(query.device());
    // Fill is a separate stream-ordered operation. Each scoring core only
    // writes its own valid tiles, avoiding cross-core initialization races.
    auto output=at::full({query.size(0),pools},-std::numeric_limits<float>::infinity(),weights.options());
    EXEC_NPU_CMD(aclnnGlmKpoolScoreV310, query,weights,cache,table,boundaries,positions,
        pools,blocks,block_rows,block_stride,row_stride,cache_offset,output);
    return output;
}
}
