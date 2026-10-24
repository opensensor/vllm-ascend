// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef QWEN_W4A8_INT4_MATMUL_TORCH_ADPT_H
#define QWEN_W4A8_INT4_MATMUL_TORCH_ADPT_H
namespace vllm_ascend {
at::Tensor npu_qwen_w4_a8_int4_matmul_310(const at::Tensor& low, const at::Tensor& high,
                                          const at::Tensor& activation_scale, const at::Tensor& activation_sum,
                                          const at::Tensor& codes, const at::Tensor& scale, const at::Tensor& offset,
                                          const at::Tensor& weight_sum, const at::Tensor& group_ends) {
  constexpr int64_t GROUP = 128, MAX_ROUTES = 20480, MIN_K = 256, MAX_K = 2560;
  constexpr int64_t MAX_N = 2 * MAX_K;
  constexpr int64_t DECODE_ROUTE_LIMIT = 128;
  TORCH_CHECK(low.device().type() == c10::DeviceType::PrivateUse1, "native INT4 requires NPU");
  for (const auto& tensor :
       {low, high, activation_scale, activation_sum, codes, scale, offset, weight_sum, group_ends}) {
    TORCH_CHECK(tensor.device() == low.device() && tensor.is_contiguous(),
                "native INT4 inputs must be contiguous on one NPU");
  }
  TORCH_CHECK(low.scalar_type() == at::kChar && high.scalar_type() == at::kChar && codes.scalar_type() == at::kChar &&
                  activation_scale.scalar_type() == at::kFloat && activation_sum.scalar_type() == at::kFloat &&
                  scale.scalar_type() == at::kHalf && offset.scalar_type() == at::kHalf &&
                  weight_sum.scalar_type() == at::kHalf &&
                  (group_ends.scalar_type() == at::kLong || group_ends.scalar_type() == at::kInt),
              "invalid native INT4 input dtypes");
  TORCH_CHECK(low.dim() == 2 && codes.dim() == 3 && scale.dim() == 3 && group_ends.dim() == 1 &&
                  (activation_scale.dim() == 2 || activation_scale.dim() == 3),
              "invalid native INT4 input ranks");
  const bool routed = group_ends.scalar_type() == at::kInt;
  const int64_t rows = low.size(0), k = low.size(1) * 2, experts = codes.size(0), n = codes.size(1);
  const int64_t output_rows = routed ? group_ends.size(0) : rows;
  TORCH_CHECK(rows > 0 && rows <= MAX_ROUTES && output_rows >= rows && output_rows <= MAX_ROUTES &&
                  output_rows % rows == 0 && k >= MIN_K && k <= MAX_K && k % GROUP == 0 && n > 0 && n <= MAX_N &&
                  n % GROUP == 0 && experts > 0 && codes.size(2) * 2 == k && high.sizes() == low.sizes() &&
                  (routed || group_ends.size(0) == experts),
              "unsupported native INT4 dimensions");
  TORCH_CHECK(activation_scale.size(0) == rows && activation_scale.size(1) == k / GROUP &&
                  activation_sum.sizes() == activation_scale.sizes() && scale.size(0) == experts &&
                  scale.size(1) == n && scale.size(2) == k / GROUP && offset.sizes() == scale.sizes() &&
                  weight_sum.sizes() == scale.sizes(),
              "native INT4 metadata shape mismatch");
  TORCH_CHECK(activation_scale.dim() == 2 || activation_scale.size(2) == 8,
              "native INT4 broadcast metadata requires eight lanes");
  TORCH_CHECK(!routed || (output_rows <= DECODE_ROUTE_LIMIT && activation_scale.dim() == 3),
              "native routed decode requires <=128 rows and broadcast metadata");
  const c10_npu::OptionalNPUGuard guard(low.device());
  auto out = at::empty({output_rows, n}, low.options().dtype(at::kHalf));
  EXEC_NPU_CMD(aclnnQwenW4A8Int4MatmulV310, low, high, activation_scale, activation_sum, codes, scale, offset,
               weight_sum, group_ends, out);
  return out;
}
}  // namespace vllm_ascend
#endif
