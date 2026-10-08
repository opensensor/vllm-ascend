// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include "aclnn_torch_adapter/op_api_common.h"
#include "attention/glm_kpool_prefill_score_v310/glm_kpool_prefill_score_310_torch_adpt.h"
at::Tensor npu_glm_kpool_prefill_score_310_meta(const at::Tensor& query, const at::Tensor& weights,
    const at::Tensor& cache, const at::Tensor& table, const at::Tensor& boundaries, const at::Tensor& positions,
    int64_t pools, int64_t blocks, int64_t block_rows, int64_t block_stride, int64_t row_stride, int64_t cache_offset)
{
    return at::empty_symint({query.sym_size(0), c10::SymInt(pools)}, weights.options());
}

TORCH_LIBRARY_FRAGMENT(_C_ascend, m) {
    m.def("npu_glm_kpool_prefill_score_310(Tensor query, Tensor weights, Tensor cache, Tensor table, Tensor boundaries, Tensor positions, int pools, int blocks, int block_rows, int block_stride, int row_stride, int cache_offset) -> Tensor");
    m.impl("npu_glm_kpool_prefill_score_310", torch::kMeta, &npu_glm_kpool_prefill_score_310_meta);
    m.impl("npu_glm_kpool_prefill_score_310", torch::kPrivateUse1, &vllm_ascend::npu_glm_kpool_prefill_score_310);
}
