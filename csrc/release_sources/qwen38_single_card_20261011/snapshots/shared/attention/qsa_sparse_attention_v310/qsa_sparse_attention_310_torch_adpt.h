#ifndef QSA_SPARSE_ATTENTION_V310_TORCH_ADPT_H
#define QSA_SPARSE_ATTENTION_V310_TORCH_ADPT_H

#include <cmath>

namespace vllm_ascend {

constexpr double QSA_SCALE_Q24_FACTOR = 16777216.0;

at::Tensor npu_qsa_sparse_attention_310(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& group_indices,
    const at::Tensor& group_counts,
    const at::Tensor& tail_starts,
    const at::Tensor& tail_counts,
    const at::Tensor& block_table,
    const at::Tensor& query_start_loc,
    double scale,
    int64_t compress_ratio)
{
    TORCH_CHECK(query.dim() == 3, "query must be [T, Nq, D]");
    TORCH_CHECK(key_cache.dim() == 4 && value_cache.sizes() == key_cache.sizes(),
                "key/value cache must have identical NZ [blocks, HD/16, block, 16] shapes");
    TORCH_CHECK(group_indices.dim() == 2 && group_indices.size(0) == query.size(0),
                "group_indices must be [T, K]");
    TORCH_CHECK(group_indices.scalar_type() == at::kInt && group_counts.scalar_type() == at::kInt &&
                    tail_starts.scalar_type() == at::kInt && tail_counts.scalar_type() == at::kInt,
                "QSA selection tensors must be int32");
    TORCH_CHECK(block_table.scalar_type() == at::kInt && query_start_loc.scalar_type() == at::kInt,
                "block_table and query_start_loc must be int32");
    TORCH_CHECK(compress_ratio == 4, "310P QSA sparse attention requires compress_ratio=4");

    at::Tensor output = at::empty_like(query);
    int64_t scale_q24 = static_cast<int64_t>(std::llround(scale * QSA_SCALE_Q24_FACTOR));
    EXEC_NPU_CMD(aclnnQsaSparseAttentionV310,
                 query,
                 key_cache,
                 value_cache,
                 group_indices,
                 group_counts,
                 tail_starts,
                 tail_counts,
                 block_table,
                 query_start_loc,
                 scale_q24,
                 compress_ratio,
                 output);
    return output;
}

}  // namespace vllm_ascend

#endif
