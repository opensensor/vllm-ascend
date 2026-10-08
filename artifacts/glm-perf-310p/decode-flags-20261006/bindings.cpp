// SPDX-License-Identifier: Apache-2.0
// Test deployment glue using the unchanged repository adapters and schemas.
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include "aclnn_torch_adapter/op_api_common.h"
#include "gmm/w2_swiglu_v310/w2_swiglu_310_torch_adpt.h"
#include "gmm/w2_route_combine_v310/w2_route_combine_310_torch_adpt.h"
at::Tensor swiglu_meta(const at::Tensor& gate_up) {
    return at::empty_symint(c10::SymDimVector{gate_up.sym_size(0), gate_up.sym_size(1) / 2}, gate_up.options());
}
at::Tensor combine_meta(const at::Tensor& routed, const at::Tensor& inverse,
                        const at::Tensor& weights, const at::Tensor& ends) {
    return at::empty_symint(c10::SymDimVector{weights.sym_size(0), routed.sym_size(1)},
                            routed.options().dtype(at::kFloat));
}
TORCH_LIBRARY_FRAGMENT(_C_ascend, m) {
    m.def("npu_w2_swiglu_310(Tensor gate_up) -> Tensor");
    m.impl("npu_w2_swiglu_310", torch::kMeta, &swiglu_meta);
    m.impl("npu_w2_swiglu_310", torch::kPrivateUse1, &vllm_ascend::npu_w2_swiglu_310);
    m.def("npu_w2_route_combine_310(Tensor routed, Tensor inverse_order, Tensor route_weights, Tensor group_ends) -> Tensor");
    m.impl("npu_w2_route_combine_310", torch::kMeta, &combine_meta);
    m.impl("npu_w2_route_combine_310", torch::kPrivateUse1, &vllm_ascend::npu_w2_route_combine_310);
}
