#ifndef SIDECARD_KDA_310_TORCH_ADPT_H
#define SIDECARD_KDA_310_TORCH_ADPT_H
#include <cstdint>
namespace vllm_ascend {
// T11: include after EXEC_NPU_CMD and c10_npu::OptionalNPUGuard declarations.
at::Tensor sidecard_kda_decode_310(const at::Tensor& query, const at::Tensor& key,
                                  const at::Tensor& value, const at::Tensor& gate,
                                  const at::Tensor& beta, at::Tensor& state)
{
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1,
                "sidecard KDA requires NPU inputs");
    TORCH_CHECK(query.dim() == 3 && query.size(0) == 1 && query.size(2) == 128,
                "sidecard KDA requires [1,H,128]");
    const int64_t heads = query.size(1);
    TORCH_CHECK(heads >= 8 && heads <= 128 && heads % 8 == 0, "unsupported sidecard KDA head count");
    const at::Tensor* inputs[] = {&query, &key, &value, &gate, &beta, &state};
    for (const auto* input : inputs) {
        TORCH_CHECK(input->is_contiguous() && input->device() == query.device(),
                    "sidecard KDA inputs must be contiguous on one device");
    }
    TORCH_CHECK(query.scalar_type() == at::kHalf && key.scalar_type() == at::kHalf &&
                value.scalar_type() == at::kHalf && gate.scalar_type() == at::kFloat &&
                beta.scalar_type() == at::kFloat && state.scalar_type() == at::kFloat,
                "sidecard KDA requires FP16 q/k/v and FP32 gate/beta/carry");
    TORCH_CHECK(key.sizes() == query.sizes() && value.sizes() == query.sizes() &&
                gate.sizes() == query.sizes(), "sidecard KDA q/k/v/gate shape mismatch");
    TORCH_CHECK(beta.dim() == 2 && beta.size(0) == 1 && beta.size(1) == heads &&
                state.dim() == 3 && state.size(0) == heads && state.size(1) == 128 && state.size(2) == 128,
                "sidecard KDA beta/carry shape mismatch");
    const auto stateBegin = reinterpret_cast<uintptr_t>(state.data_ptr());
    const auto stateEnd = stateBegin + state.numel() * state.element_size();
    for (const auto* input : {&query, &key, &value, &gate, &beta}) {
        const auto begin = reinterpret_cast<uintptr_t>(input->data_ptr());
        const auto end = begin + input->numel() * input->element_size();
        TORCH_CHECK(end <= stateBegin || begin >= stateEnd,
                    "sidecard KDA read-only input must not overlap mutable carry");
    }
    // Select the owner context even when a direct torch.ops caller currently
    // has the other rank's NPU selected. Restore that context after dispatch.
    const c10_npu::OptionalNPUGuard guard(query.device());
    auto output = at::empty_like(value);
    // The CANN op has a distinct state_out port. Safe aliasing is intentional:
    // each work group reads/writes disjoint [head,value,:] rows, without peers.
    EXEC_NPU_CMD(aclnnSidecardKdaV310, query, key, value, gate, beta, state, output, state);
    return output;
}
}  // namespace vllm_ascend
#endif
