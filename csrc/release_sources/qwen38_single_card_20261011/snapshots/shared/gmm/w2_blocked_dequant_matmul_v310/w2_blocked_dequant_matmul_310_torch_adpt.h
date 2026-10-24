/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef W2_BLOCKED_DEQUANT_MATMUL_V310_TORCH_ADPT_H
#define W2_BLOCKED_DEQUANT_MATMUL_V310_TORCH_ADPT_H
namespace vllm_ascend {

// out[T, N] (fp16) = x[T, K] (fp16) @ W^T,
// where W[n, k] = unpack_signed(codes)[n, k] * block_scale[n/32, k/32].
// codes is packed uint8 [N, K/4] for W2 or [N, K/2] for W4, little-endian
// by field. N = codes.size(0), K = x.size(1).
at::Tensor npu_w2_blocked_dequant_matmul_310(
    const at::Tensor& x,
    const at::Tensor& codes,
    const at::Tensor& block_scale)
{
    at::Tensor out = at::empty({x.size(0), codes.size(0)}, x.options());
    EXEC_NPU_CMD(aclnnW2BlockedDequantMatmulV310,
                 x,
                 codes,
                 block_scale,
                 out);
    return out;
}

}  // namespace vllm_ascend
#endif  // W2_BLOCKED_DEQUANT_MATMUL_V310_TORCH_ADPT_H
