// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef QWEN_W4_A8_SWIGLU_PACK_310_TORCH_ADPT_H
#define QWEN_W4_A8_SWIGLU_PACK_310_TORCH_ADPT_H
namespace vllm_ascend {
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_qwen_w4_a8_swiglu_pack_310(
    const at::Tensor& gate_up) {
  constexpr int64_t MAX_ROUTES = 128, MIN_K = 256, MAX_K = 2560, GROUP_SIZE = 128, LANES = 8;
  TORCH_CHECK(gate_up.device().type() == c10::DeviceType::PrivateUse1 &&
                  gate_up.scalar_type() == at::kHalf && gate_up.is_contiguous() && gate_up.dim() == 2,
              "W4A8 SwiGLU pack requires contiguous NPU FP16 [R,2K]");
  const int64_t rows = gate_up.size(0), gate_up_width = gate_up.size(1);
  TORCH_CHECK(gate_up_width % 2 == 0, "W4A8 SwiGLU pack requires an even gate/up width");
  const int64_t k = gate_up_width / 2;
  TORCH_CHECK(rows > 0 && rows <= MAX_ROUTES && k >= MIN_K && k <= MAX_K && k % GROUP_SIZE == 0,
              "unsupported W4A8 SwiGLU pack dimensions");
  const c10_npu::OptionalNPUGuard guard(gate_up.device());
  auto low = at::empty({rows, k / 2}, gate_up.options().dtype(at::kChar));
  auto high = at::empty_like(low);
  auto scale = at::empty({rows, k / GROUP_SIZE, LANES}, gate_up.options().dtype(at::kFloat));
  auto sum = at::empty_like(scale);
  EXEC_NPU_CMD(aclnnQwenW4A8SwigluPackV310, gate_up, low, high, scale, sum);
  return {low, high, scale, sum};
}
}  // namespace vllm_ascend
#endif
