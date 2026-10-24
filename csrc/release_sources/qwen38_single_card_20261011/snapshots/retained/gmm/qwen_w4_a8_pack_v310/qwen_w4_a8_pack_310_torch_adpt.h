// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef QWEN_W4_A8_PACK_TORCH_ADPT_H
#define QWEN_W4_A8_PACK_TORCH_ADPT_H
namespace vllm_ascend {
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_qwen_w4_a8_pack_310(const at::Tensor& x) {
  constexpr int64_t MAX_ROUTES = 5120, MIN_K = 256, MAX_K = 2560, GROUP_SIZE = 128, LANES = 8;
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 && x.scalar_type() == at::kHalf && x.is_contiguous() &&
                  x.dim() == 2,
              "W4A8 pack requires contiguous NPU FP16 [R,K]");
  const int64_t rows = x.size(0), k = x.size(1);
  TORCH_CHECK(rows > 0 && rows <= MAX_ROUTES && k >= MIN_K && k <= MAX_K && k % GROUP_SIZE == 0,
              "unsupported W4A8 pack dimensions");
  const c10_npu::OptionalNPUGuard guard(x.device());
  auto low = at::empty({rows, k / 2}, x.options().dtype(at::kChar));
  auto high = at::empty_like(low);
  auto scale = at::empty({rows, k / GROUP_SIZE, LANES}, x.options().dtype(at::kFloat));
  auto sum = at::empty_like(scale);
  EXEC_NPU_CMD(aclnnQwenW4A8PackV310, x, low, high, scale, sum);
  return {low, high, scale, sum};
}
}  // namespace vllm_ascend
#endif
