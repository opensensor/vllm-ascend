// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef QWEN_W8_ROUTE_COMBINE_310_TORCH_ADPT_H
#define QWEN_W8_ROUTE_COMBINE_310_TORCH_ADPT_H

namespace vllm_ascend {
at::Tensor npu_qwen_w8_route_combine_310(const at::Tensor& sorted_routes,
                                          const at::Tensor& inverse_order,
                                          const at::Tensor& route_weights,
                                          const at::Tensor& local_route_mask) {
  constexpr int64_t MAX_TOKENS = 8;
  constexpr int64_t MAX_TOP_K = 16;
  constexpr int64_t MAX_HIDDEN_SIZE = 8192;
  constexpr int64_t HIDDEN_ALIGNMENT = 16;
  TORCH_CHECK(sorted_routes.device().type() == c10::DeviceType::PrivateUse1 &&
                  sorted_routes.scalar_type() == at::kHalf && sorted_routes.is_contiguous() &&
                  sorted_routes.dim() == 2,
              "W8 route combine requires contiguous NPU FP16 sorted routes [R,H]");
  TORCH_CHECK(inverse_order.device() == sorted_routes.device() && inverse_order.scalar_type() == at::kInt &&
                  inverse_order.is_contiguous() && inverse_order.dim() == 1,
              "W8 route combine requires contiguous NPU INT32 inverse order [R]");
  TORCH_CHECK(route_weights.device() == sorted_routes.device() && route_weights.scalar_type() == at::kFloat &&
                  route_weights.is_contiguous() && route_weights.dim() == 2,
              "W8 route combine requires contiguous NPU FP32 route weights [T,K]");
  TORCH_CHECK(local_route_mask.device() == sorted_routes.device() && local_route_mask.scalar_type() == at::kBool &&
                  local_route_mask.is_contiguous() && local_route_mask.sizes() == route_weights.sizes(),
              "W8 route combine requires contiguous NPU BOOL local route mask [T,K]");
  const int64_t num_tokens = route_weights.size(0), top_k = route_weights.size(1);
  const int64_t routes = num_tokens * top_k, hidden = sorted_routes.size(1);
  TORCH_CHECK(num_tokens > 0 && num_tokens <= MAX_TOKENS && top_k > 0 && top_k <= MAX_TOP_K &&
                  hidden > 0 && hidden <= MAX_HIDDEN_SIZE && hidden % HIDDEN_ALIGNMENT == 0,
              "unsupported W8 route-combine dimensions");
  TORCH_CHECK(sorted_routes.size(0) == routes && inverse_order.size(0) == routes,
              "W8 route-combine inputs disagree on route count");
  const c10_npu::OptionalNPUGuard guard(sorted_routes.device());
  auto output = at::empty({num_tokens, hidden}, sorted_routes.options());
  EXEC_NPU_CMD(aclnnQwenW8RouteCombineV310, sorted_routes, inverse_order, route_weights,
               local_route_mask, output);
  return output;
}
}  // namespace vllm_ascend
#endif
