// SPDX-License-Identifier: Apache-2.0
#ifndef SIDECARD_MHC_POST_310_TORCH_ADPT_H
#define SIDECARD_MHC_POST_310_TORCH_ADPT_H
namespace vllm_ascend {
// T11 schema: sidecard_mhc_post_310(Tensor x, Tensor residual, Tensor post, Tensor comb) -> Tensor
at::Tensor sidecard_mhc_post_310(const at::Tensor& x, const at::Tensor& residual, const at::Tensor& post,
                                 const at::Tensor& comb) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 && x.scalar_type() == at::kHalf && x.is_contiguous() &&
                  x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 32 && x.size(1) == 4096,
              "mHC post requires contiguous NPU FP16 x [1..32,4096]");
  for (const auto& input : {residual, post, comb}) {
    TORCH_CHECK(input.device() == x.device() && input.scalar_type() == at::kFloat && input.is_contiguous() &&
                    input.dim() == 3 && input.size(0) == x.size(0) && input.size(1) == 4,
                "mHC residual and mixers require contiguous FP32 [rows,4,...] on x device");
  }
  TORCH_CHECK(residual.size(2) == 4096 && post.size(2) == 1 && comb.size(2) == 4,
              "invalid mHC residual or mixer shape");
  const c10_npu::OptionalNPUGuard guard(x.device());
  auto output = at::empty_like(residual);
  EXEC_NPU_CMD(aclnnSidecardMhcPostV310, x, residual, post, comb, output);
  return output;
}
}  // namespace vllm_ascend
#endif
