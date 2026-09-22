#ifndef QSA_INDEX_CACHE_UPDATE_V310_TORCH_ADPT_H
#define QSA_INDEX_CACHE_UPDATE_V310_TORCH_ADPT_H

namespace vllm_ascend {

void qsa_index_cache_update_310(at::Tensor& compressed_key_cache,
                                const at::Tensor& index_keys,
                                const at::Tensor& query_start_loc,
                                const at::Tensor& slot_mapping,
                                const at::Tensor& key_norm_weight,
                                const at::Tensor& rope_cos,
                                const at::Tensor& rope_sin,
                                int64_t block_size,
                                int64_t compress_ratio,
                                int64_t rotary_dim,
                                double norm_eps)
{
    TORCH_CHECK(compressed_key_cache.dim() == 3,
                "compressed_key_cache must be [blocks,rows,D]");
    TORCH_CHECK(index_keys.dim() == 2, "index_keys must be [T,D]");
    TORCH_CHECK(index_keys.size(0) == slot_mapping.size(0),
                "index_keys and slot_mapping must have the same token count");
    TORCH_CHECK(index_keys.size(1) == compressed_key_cache.size(2),
                "index_keys and cache must have the same head dimension");
    TORCH_CHECK(query_start_loc.scalar_type() == at::kInt &&
                    slot_mapping.scalar_type() == at::kInt,
                "query_start_loc and slot_mapping must be int32");
    TORCH_CHECK((block_size == 64 || block_size == 128) && compress_ratio == 4,
                "310P QSA cache update requires block_size 64/128 and compress_ratio=4");
    TORCH_CHECK(compressed_key_cache.size(1) == block_size / compress_ratio + compress_ratio - 1,
                "compressed cache has the wrong number of group/scratch rows");
    TORCH_CHECK(key_norm_weight.sizes() == at::IntArrayRef({index_keys.size(1)}),
                "key_norm_weight must be [D]");
    TORCH_CHECK(rope_cos.sizes() == at::IntArrayRef({index_keys.size(0), rotary_dim}) &&
                    rope_sin.sizes() == rope_cos.sizes(),
                "rope_cos and rope_sin must be [T, rotary_dim]");
    float norm_eps_value = static_cast<float>(norm_eps);
    EXEC_NPU_CMD(aclnnQsaIndexCacheUpdateV310,
                 compressed_key_cache,
                 index_keys,
                 query_start_loc,
                 slot_mapping,
                 key_norm_weight,
                 rope_cos,
                 rope_sin,
                 block_size,
                 compress_ratio,
                 rotary_dim,
                 norm_eps_value);
}

}  // namespace vllm_ascend

#endif
