// SPDX-License-Identifier: Apache-2.0
#ifndef SIDECARD_MHC_PRE_310_TORCH_ADPT_H
#define SIDECARD_MHC_PRE_310_TORCH_ADPT_H
#include <cmath>
namespace vllm_ascend {
// T11: include this header after EXEC_NPU_CMD/OptionalNPUGuard declarations.
// Register: sidecard_mhc_pre_310(Tensor residual, Tensor fn, Tensor scale,
// Tensor base, Tensor norm_weight, float norm_eps, float hc_eps) -> (Tensor, Tensor, Tensor)
std::tuple<at::Tensor, at::Tensor, at::Tensor> sidecard_mhc_pre_310(const at::Tensor& residual, const at::Tensor& fn,
                                                                    const at::Tensor& scale, const at::Tensor& base,
                                                                    const at::Tensor& normWeight, double normEps,
                                                                    double hcEps) {
  TORCH_CHECK(residual.device().type() == c10::DeviceType::PrivateUse1 && residual.scalar_type() == at::kFloat &&
                  residual.is_contiguous() && residual.dim() == 3 && residual.size(0) >= 1 && residual.size(0) <= 32 &&
                  residual.size(1) == 4 && residual.size(2) == 4096,
              "mHC pre requires contiguous NPU FP32 residual [1..32,4,4096]");
  for (const auto& input : {fn, scale, base}) {
    TORCH_CHECK(input.device() == residual.device() && input.scalar_type() == at::kFloat && input.is_contiguous(),
                "mHC coefficients must be contiguous FP32 on residual device");
  }
  TORCH_CHECK(fn.dim() == 2 && fn.size(0) == 24 && fn.size(1) == 16384 && scale.dim() == 1 && scale.size(0) == 3 &&
                  base.dim() == 1 && base.size(0) == 24,
              "mHC pre expects fn [24,16384], scale [3], base [24]");
  TORCH_CHECK(normWeight.device() == residual.device() && normWeight.scalar_type() == at::kHalf &&
                  normWeight.is_contiguous() && normWeight.dim() == 1 && normWeight.size(0) == 4096,
              "mHC norm weight must be contiguous FP16 [4096]");
  TORCH_CHECK(std::isfinite(normEps) && normEps > 0 && std::isfinite(hcEps) && hcEps > 0 &&
                  std::isfinite(static_cast<float>(normEps)) && static_cast<float>(normEps) > 0 &&
                  std::isfinite(static_cast<float>(hcEps)) && static_cast<float>(hcEps) > 0,
              "mHC epsilons must be representable positive finite FP32");
  const c10_npu::OptionalNPUGuard guard(residual.device());
  const auto rows = residual.size(0);
  auto post = at::empty({rows, 4, 1}, residual.options());
  auto logits = at::empty({rows, 4, 4}, residual.options());
  auto hidden = at::empty({rows, 4096}, normWeight.options());
  EXEC_NPU_CMD(aclnnSidecardMhcPreV310, residual, fn, scale, base, normWeight, normEps, hcEps, post, logits, hidden);
  return {post, logits, hidden};
}
}  // namespace vllm_ascend
#endif
