// SPDX-License-Identifier: Apache-2.0
// A private addon, loaded alongside the original qualified base extension.
// Never compile the baseline torch_binding.cpp into this library.
#include <torch/library.h>
#include <torch/types.h>
#include <array>
#include <climits>
#include <unistd.h>
#include <string>
using std::string;
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "aclnn_torch_adapter/op_api_common.h"
#include "attention/sidecard_kda_v310/sidecard_kda_310_torch_adpt.h"
#include "attention/sidecard_mhc_pre_v310/sidecard_mhc_pre_310_torch_adpt.h"
#include "gmm/sidecard_mhc_post_v310/sidecard_mhc_post_310_torch_adpt.h"
#include "meta.h"

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
  ops.def(
      "sidecard_kda_decode_310(Tensor query, Tensor key, Tensor value, Tensor gate, Tensor beta, "
      "Tensor(a!) state) -> Tensor");
  ops.def(
      "sidecard_mhc_pre_310(Tensor residual, Tensor fn, Tensor scale, Tensor base, "
      "Tensor norm_weight, float norm_eps, float hc_eps) -> (Tensor, Tensor, Tensor)");
  ops.def("sidecard_mhc_post_310(Tensor x, Tensor residual, Tensor post, Tensor comb) -> Tensor");
}

TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, ops) {
  ops.impl("sidecard_kda_decode_310", &vllm_ascend::sidecard_kda_decode_310);
  ops.impl("sidecard_mhc_pre_310", &vllm_ascend::sidecard_mhc_pre_310);
  ops.impl("sidecard_mhc_post_310", &vllm_ascend::sidecard_mhc_post_310);
}

TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops) {
  ops.impl("sidecard_kda_decode_310", &sidecard_execution::KdaMeta);
  ops.impl("sidecard_mhc_pre_310", &sidecard_execution::PreMeta);
  ops.impl("sidecard_mhc_post_310", &sidecard_execution::PostMeta);
}
