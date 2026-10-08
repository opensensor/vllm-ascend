// SPDX-License-Identifier: Apache-2.0
// Supplemental test binding; do not load alongside a full binding defining this op.
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include "aclnn_torch_adapter/op_api_common.h"
#include "gmm/glm_mhc_post_v310/glm_mhc_post_310_torch_adpt.h"
at::Tensor mhc_post_meta(const at::Tensor& x, const at::Tensor& residual,
                         const at::Tensor& post, const at::Tensor& comb) {
    return at::empty_symint(residual.sym_sizes(), residual.options());
}
TORCH_LIBRARY_FRAGMENT(_C_ascend, m) {
    m.def("npu_glm_mhc_post_310(Tensor x, Tensor residual, Tensor post_mix, Tensor comb_mix) -> Tensor");
    m.impl("npu_glm_mhc_post_310", torch::kMeta, &mhc_post_meta);
    m.impl("npu_glm_mhc_post_310", torch::kPrivateUse1, &vllm_ascend::npu_glm_mhc_post_310);
}
