// SPDX-License-Identifier: Apache-2.0
#ifndef GLM_MHC_POST_310_TORCH_ADPT_H
#define GLM_MHC_POST_310_TORCH_ADPT_H
#include "mhc_post_geometry.h"
namespace vllm_ascend {
at::Tensor npu_glm_mhc_post_310(const at::Tensor& x, const at::Tensor& residual,
                              const at::Tensor& post_mix, const at::Tensor& comb_mix) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 && x.scalar_type() == at::kHalf &&
                x.is_contiguous() && x.dim() == 2, "mHC post requires contiguous NPU FP16 x [tokens,width]");
    TORCH_CHECK(NsGlmMhcPost::ValidGeometry(x.size(0), x.size(1)), "invalid mHC post geometry");
    for (const auto& input : {residual, post_mix, comb_mix}) {
        TORCH_CHECK(input.device() == x.device() && input.scalar_type() == at::kFloat && input.is_contiguous() &&
                    input.dim() == 3 && input.size(0) == x.size(0) && input.size(1) == NsGlmMhcPost::STREAMS,
                    "mHC state must be contiguous FP32 [tokens,4,...] on the same NPU as x");
    }
    TORCH_CHECK(residual.size(2) == x.size(1) && post_mix.size(2) == 1 &&
                comb_mix.size(2) == NsGlmMhcPost::STREAMS, "invalid mHC residual or mixing shape");
    const c10_npu::OptionalNPUGuard guard(x.device());
    auto output = at::empty(residual.sizes(), residual.options());
    EXEC_NPU_CMD(aclnnGlmMhcPostV310, x, residual, post_mix, comb_mix, output);
    return output;
}
}  // namespace vllm_ascend
#endif
