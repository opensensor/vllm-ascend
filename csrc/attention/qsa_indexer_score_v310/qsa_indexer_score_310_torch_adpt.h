#ifndef QSA_INDEXER_SCORE_V310_TORCH_ADPT_H
#define QSA_INDEXER_SCORE_V310_TORCH_ADPT_H

namespace vllm_ascend {

constexpr int64_t QSA_INDEX_CACHE_SCRATCH_ROWS = 3;

at::Tensor npu_qsa_indexer_score_310(const at::Tensor& query,
                                     const at::Tensor& compressed_key_cache,
                                     const at::Tensor& block_table,
                                     const at::Tensor& query_start_loc,
                                     const at::Tensor& positions,
                                     int64_t compress_ratio)
{
    TORCH_CHECK(query.dim() == 3, "query must be [T,H,D]");
    TORCH_CHECK(compressed_key_cache.dim() == 3, "compressed_key_cache must be [blocks,rows,D]");
    TORCH_CHECK(compressed_key_cache.size(1) > QSA_INDEX_CACHE_SCRATCH_ROWS,
                "compressed_key_cache must contain group rows and three scratch rows");
    TORCH_CHECK(block_table.dim() == 2 && block_table.scalar_type() == at::kInt,
                "block_table must be 2D int32");
    TORCH_CHECK(query_start_loc.scalar_type() == at::kInt && positions.scalar_type() == at::kInt,
                "query_start_loc and positions must be int32");
    TORCH_CHECK(compress_ratio == 4, "310P QSA index score requires compress_ratio=4");
    at::Tensor scores = at::empty(
        {query.size(0), block_table.size(1) *
                            (compressed_key_cache.size(1) - QSA_INDEX_CACHE_SCRATCH_ROWS)},
        query.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnQsaIndexerScoreV310,
                 query,
                 compressed_key_cache,
                 block_table,
                 query_start_loc,
                 positions,
                 compress_ratio,
                 scores);
    return scores;
}

}  // namespace vllm_ascend

#endif
