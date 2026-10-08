// SPDX-License-Identifier: Apache-2.0
#ifndef W2_ROUTE_COMBINE_310_TORCH_ADPT_H
#define W2_ROUTE_COMBINE_310_TORCH_ADPT_H
#include "route_combine_geometry.h"
namespace vllm_ascend {
at::Tensor npu_w2_route_combine_310(const at::Tensor& routed, const at::Tensor& inverse_order,
                                  const at::Tensor& route_weights, const at::Tensor& group_ends) {
    TORCH_CHECK(routed.device().type() == c10::DeviceType::PrivateUse1 &&
                routed.scalar_type() == at::kHalf && routed.is_contiguous() && routed.dim() == 2,
                "route combine requires contiguous NPU FP16 [routes,hidden]");
    TORCH_CHECK(inverse_order.device() == routed.device() && inverse_order.scalar_type() == at::kLong &&
                inverse_order.is_contiguous() && inverse_order.dim() == 1,
                "inverse_order must be contiguous INT64 [routes] on the same device");
    TORCH_CHECK(route_weights.device() == routed.device() && route_weights.scalar_type() == at::kFloat &&
                route_weights.is_contiguous() && route_weights.dim() == 2,
                "route_weights must be contiguous FP32 [tokens,top_k] on the same device");
    TORCH_CHECK(group_ends.device() == routed.device() && group_ends.scalar_type() == at::kLong &&
                group_ends.is_contiguous() && group_ends.dim() == 1,
                "group_ends must be contiguous INT64 [experts] on the same device");
    TORCH_CHECK(NsW2Combine::ValidGeometry(routed.size(0), routed.size(1), route_weights.size(0),
                                         route_weights.size(1), group_ends.size(0)) &&
                inverse_order.numel() == routed.size(0), "invalid route-combine geometry");
    const c10_npu::OptionalNPUGuard guard(routed.device());
    auto output = at::empty({route_weights.size(0), routed.size(1)}, routed.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnW2RouteCombineV310, routed, inverse_order, route_weights, group_ends, output);
    return output;
}
}  // namespace vllm_ascend
#endif
