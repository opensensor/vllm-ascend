// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#ifndef QSA_GATHER_VALUE_NZ_V310_TORCH_ADPT_H
#define QSA_GATHER_VALUE_NZ_V310_TORCH_ADPT_H

namespace vllm_ascend {

void qsa_gather_value_nz_310(const at::Tensor& value_cache,
                             const at::Tensor& group_indices,
                             const at::Tensor& group_counts,
                             const at::Tensor& tail_starts,
                             const at::Tensor& tail_counts,
                             const at::Tensor& block_table,
                             at::Tensor& value_nz,
                             int64_t num_kv_heads,
                             int64_t head_dim,
                             bool transpose_output)
{
    TORCH_CHECK(value_cache.dim() == 4 && value_cache.scalar_type() == at::kHalf,
                "value_cache must be a float16 NZ [blocks,channels,block,16] tensor");
    TORCH_CHECK(group_indices.dim() == 2 && group_indices.scalar_type() == at::kInt,
                "group_indices must be [T,K] int32");
    TORCH_CHECK(group_indices.size(0) > 0 && group_indices.size(1) > 0,
                "QSA gather requires at least one token and one selected group");
    TORCH_CHECK(group_counts.scalar_type() == at::kInt && tail_starts.scalar_type() == at::kInt &&
                    tail_counts.scalar_type() == at::kInt && block_table.scalar_type() == at::kInt,
                "QSA gather metadata must be int32");
    TORCH_CHECK(group_counts.sizes() == at::IntArrayRef({group_indices.size(0)}) &&
                    tail_starts.sizes() == group_counts.sizes() && tail_counts.sizes() == group_counts.sizes(),
                "QSA gather count/tail vectors must match the token count");
    TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == 1 && block_table.size(1) > 0,
                "QSA gather currently supports one request");
    TORCH_CHECK(head_dim > 0 && head_dim % 16 == 0 && num_kv_heads > 0 &&
                    value_cache.size(1) == num_kv_heads * head_dim / 16,
                "QSA gather head geometry is incompatible with the cache");
    TORCH_CHECK(value_cache.size(2) > 0 && value_cache.size(2) % 4 == 0 && value_cache.size(3) == 16,
                "QSA gather cache block size must be positive and divisible by four");
    const int64_t padded_tokens = ((group_indices.size(1) * 4 + 4 + 15) / 16) * 16;
    const bool shape_ok = transpose_output
        ? value_nz.sizes() == at::IntArrayRef({group_indices.size(0), num_kv_heads, head_dim, padded_tokens})
        : value_nz.sizes() == at::IntArrayRef({group_indices.size(0), num_kv_heads, padded_tokens, head_dim});
    TORCH_CHECK(shape_ok &&
                    value_nz.scalar_type() == at::kHalf,
                "value_nz must be float16 [T,H,S,D] or transposed [T,H,D,S]");
    EXEC_NPU_CMD(aclnnQsaGatherValueNzV310,
                 value_cache,
                 group_indices,
                 group_counts,
                 tail_starts,
                 tail_counts,
                 block_table,
                 num_kv_heads,
                 head_dim,
                 transpose_output,
                 value_nz);
}

}  // namespace vllm_ascend

#endif
