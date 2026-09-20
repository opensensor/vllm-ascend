/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GDN_GATING_V310_TORCH_ADPT_H
#define GDN_GATING_V310_TORCH_ADPT_H
namespace vllm_ascend {

// g[T, H] (fp32)       = -exp(A_log[h]) * softplus(a[t, h] + dt_bias[h], beta)
// beta_out[T, H] (fp16) = sigmoid(b[t, h])
//
// neg_exp_a_log_tiled / dt_bias_tiled are [32, H] fp32 broadcasts of the
// weight-only constants; see gdn_gating_v310_def.cpp for why they arrive
// pre-tiled. Outputs are padded to a multiple of 32 rows so the kernel can use
// aligned UB->GM DataCopy stores; this adapter narrows them back to T as views.
std::tuple<at::Tensor, at::Tensor> npu_gdn_gating_310(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& neg_exp_a_log_tiled,
    const at::Tensor& dt_bias_tiled,
    double beta)
{
    // tileRows is a function of the head count (smallest value keeping
    // tileRows*H a multiple of 16 elements); the tiled constants already carry
    // it, so read it from there rather than duplicating the rule.
    const int64_t tile_rows = neg_exp_a_log_tiled.size(0);
    auto output_sizes = a.sizes().vec();
    output_sizes[0] = (output_sizes[0] + tile_rows - 1) / tile_rows * tile_rows;
    at::Tensor g = at::empty(output_sizes, a.options().dtype(at::kFloat));
    at::Tensor beta_out = at::empty(output_sizes, a.options());
    EXEC_NPU_CMD(aclnnGdnGatingV310,
                 a,
                 b,
                 neg_exp_a_log_tiled,
                 dt_bias_tiled,
                 beta,
                 g,
                 beta_out
                );

    const int64_t num_rows = a.size(0);
    return std::make_tuple(g.narrow(0, 0, num_rows), beta_out.narrow(0, 0, num_rows));
}

}
#endif
