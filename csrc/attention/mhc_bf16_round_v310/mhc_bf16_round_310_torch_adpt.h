#ifndef MHC_BF16_ROUND_310_TORCH_ADPT_H
#define MHC_BF16_ROUND_310_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor mhc_bf16_round_310(const at::Tensor& input)
{
    TORCH_CHECK(input.scalar_type() == at::kFloat && input.is_contiguous() && input.numel() > 0,
                "310P mHC BF16 rounding requires a nonempty contiguous float32 tensor");
    at::Tensor output = at::empty_like(input);
    EXEC_NPU_CMD(aclnnMhcBf16RoundV310, input, output);
    return output;
}

}  // namespace vllm_ascend

#endif
