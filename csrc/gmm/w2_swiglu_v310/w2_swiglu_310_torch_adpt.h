// SPDX-License-Identifier: Apache-2.0
#ifndef W2_SWIGLU_310_TORCH_ADPT_H
#define W2_SWIGLU_310_TORCH_ADPT_H
#include "swiglu_geometry.h"
namespace vllm_ascend {
at::Tensor npu_w2_swiglu_310(const at::Tensor& gate_up) {
    TORCH_CHECK(gate_up.device().type() == c10::DeviceType::PrivateUse1 &&
                gate_up.scalar_type() == at::kHalf && gate_up.is_contiguous() && gate_up.dim() == 2,
                "SwiGLU requires contiguous NPU FP16 [routes,2*intermediate]");
    TORCH_CHECK(NsW2Swiglu::ValidGeometry(gate_up.size(0), gate_up.size(1)), "invalid SwiGLU geometry");
    const c10_npu::OptionalNPUGuard guard(gate_up.device());
    auto output = at::empty({gate_up.size(0), gate_up.size(1) / 2}, gate_up.options());
    EXEC_NPU_CMD(aclnnW2SwigluV310, gate_up, output);
    return output;
}
}  // namespace vllm_ascend
#endif
